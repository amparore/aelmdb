// mtest_perf.c
// Benchmark insert/delete performance with and without LMDB aggregate extension.
//
// Build (example, in the directory containing lmdb.h, mdb.c, midl.c):
//   cc -O3 -DNDEBUG -std=c11 -Wall -Wextra -pedantic 
//      mtest_perf.c mdb.c midl.c -lpthread -o mtest_perf
//
// Usage example:
//   ./mtest_perf --path /tmp/lmdb_bench --csv out.csv --n 200000 
//       --order asc,rand,desc --value-size 128 --txn-batch 5000 --nosync
// 
// 
//   ./mtest_perf --path /tmp/lmdb_bench_envs --csv bench.csv --n 500000 --value-size 128 --txn-batch 10000 --order asc,rand,desc --nosync
//   ./mtest_perf --path /tmp/lmdb_bench_envs --csv bench_hashsum.csv --n 500000 --value-size 128 --txn-batch 10000 --order rand --with-hashsum --hash-offset 0 --nosync
//
// Notes:
// - By default this runs two modes: plain (no aggregates) and agg (ENTRIES|KEYS).
// - Add --with-hashsum to also test ENTRIES|KEYS|HASHSUM.
//
// This program targets Linux (RSS from /proc). It will still run elsewhere,
// but RSS will be reported as 0.

#define _GNU_SOURCE 1

#include <assert.h>
#include <errno.h>
#include <inttypes.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#include "lmdb.h"

#ifndef ARRAY_LEN
#define ARRAY_LEN(x) (sizeof(x) / sizeof((x)[0]))
#endif

// ------------------------- utilities -------------------------

static void die(const char *fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  vfprintf(stderr, fmt, ap);
  va_end(ap);
  fputc('\n', stderr);
  exit(1);
}

static char *xasprintf(const char *fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  char *s = NULL;
  int rc = vasprintf(&s, fmt, ap);
  va_end(ap);
  if (rc < 0 || !s) {
    die("vasprintf failed");
  }
  return s;
}

static void mdb_chk(int rc, const char *what) {
  if (rc != MDB_SUCCESS) {
    die("%s: %s (%d)", what, mdb_strerror(rc), rc);
  }
}

static uint64_t now_ns(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

static uint64_t rss_kb_linux(void) {
#ifdef __linux__
  FILE *f = fopen("/proc/self/status", "r");
  if (!f) return 0;
  char line[512];
  while (fgets(line, sizeof(line), f)) {
    if (strncmp(line, "VmRSS:", 6) == 0) {
      char *p = line + 6;
      while (*p == ' ' || *p == '\t') p++;
      uint64_t kb = strtoull(p, NULL, 10);
      fclose(f);
      return kb;
    }
  }
  fclose(f);
  return 0;
#else
  return 0;
#endif
}

static void mkdir_p(const char *path) {
  if (mkdir(path, 0700) == 0) return;
  if (errno == EEXIST) return;
  die("mkdir(%s): %s", path, strerror(errno));
}

static void rm_rf(const char *path) {
  char *p = xasprintf("%s/data.mdb", path);
  unlink(p);
  free(p);
  p = xasprintf("%s/lock.mdb", path);
  unlink(p);
  free(p);
  rmdir(path);
}

static void encode_be64(uint64_t x, uint8_t out[8]) {
  out[0] = (uint8_t)(x >> 56);
  out[1] = (uint8_t)(x >> 48);
  out[2] = (uint8_t)(x >> 40);
  out[3] = (uint8_t)(x >> 32);
  out[4] = (uint8_t)(x >> 24);
  out[5] = (uint8_t)(x >> 16);
  out[6] = (uint8_t)(x >> 8);
  out[7] = (uint8_t)(x);
}

static uint64_t splitmix64(uint64_t *state) {
  uint64_t z = (*state += 0x9e3779b97f4a7c15ull);
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
  return z ^ (z >> 31);
}

static void shuffle_u64(uint64_t *a, size_t n, uint64_t seed) {
  uint64_t st = seed ? seed : 0x123456789abcdef0ull;
  for (size_t i = n; i > 1; i--) {
    uint64_t r = splitmix64(&st);
    size_t j = (size_t)(r % i);
    uint64_t tmp = a[i - 1];
    a[i - 1] = a[j];
    a[j] = tmp;
  }
}

static bool file_is_empty_or_missing(const char *path) {
  FILE *f = fopen(path, "r");
  if (!f) return true;
  int c = fgetc(f);
  fclose(f);
  return c == EOF;
}

static const char *order_to_str(int o) {
  switch (o) {
    case 0: return "asc";
    case 1: return "desc";
    case 2: return "rand";
    default: return "?";
  }
}

static int parse_order_token(const char *t) {
  if (strcmp(t, "asc") == 0) return 0;
  if (strcmp(t, "desc") == 0) return 1;
  if (strcmp(t, "rand") == 0 || strcmp(t, "random") == 0) return 2;
  return -1;
}

// ------------------------- benchmark core -------------------------

typedef struct {
  const char *label;
  unsigned dbi_flags;   // aggregate flags, etc, passed to mdb_dbi_open
  bool use_hashsum;
  unsigned hash_offset;
} bench_mode_t;

typedef struct {
  const char *base_path;
  const char *csv_path;
  const char *db_name;
  uint64_t n;
  size_t value_size;
  uint64_t txn_batch;
  uint64_t seed;
  unsigned env_flags;
  int *orders;
  size_t orders_len;
  int delete_order; // -1 => same as insert; else 0/1/2
  bool do_delete;
  bool keep_env;
  bool verify;
  bool run_hashsum_mode;
  unsigned hash_offset;
} cfg_t;

static void fill_value(uint8_t *buf, size_t sz, uint64_t key, bool need_hashsum_bytes) {
  if (sz == 0) return;
  memset(buf, 0, sz);

  if (!need_hashsum_bytes) {
    if (sz >= 8) {
      uint64_t x = key * 11400714819323198485ull;
      memcpy(buf, &x, 8);
    } else {
      buf[0] = (uint8_t)key;
    }
    return;
  }

  uint8_t *p = buf;
  size_t need = (size_t)MDB_HASH_SIZE;
  if (need > sz) need = sz;
  uint64_t st = key ^ 0x6a09e667f3bcc909ull;
  size_t off = 0;
  while (off < need) {
    uint64_t x = splitmix64(&st);
    size_t chunk = (need - off) < 8 ? (need - off) : 8;
    memcpy(p + off, &x, chunk);
    off += chunk;
  }
}

static void maybe_verify_totals(MDB_env *env, MDB_dbi dbi, uint64_t expected_entries, uint64_t expected_keys) {
  MDB_txn *rtxn = NULL;
  mdb_chk(mdb_txn_begin(env, NULL, MDB_RDONLY, &rtxn), "mdb_txn_begin(read)");

  MDB_agg agg;
  int rc = mdb_agg_totals(rtxn, dbi, &agg);
  if (rc == MDB_INCOMPATIBLE) {
    mdb_txn_abort(rtxn);
    return;
  }
  mdb_chk(rc, "mdb_agg_totals");

  if ((agg.mv_flags & MDB_AGG_ENTRIES) && agg.mv_agg_entries != expected_entries) {
    die("verify failed: entries got=%" PRIu64 " expected=%" PRIu64, agg.mv_agg_entries, expected_entries);
  }
  if ((agg.mv_flags & MDB_AGG_KEYS) && agg.mv_agg_keys != expected_keys) {
    die("verify failed: keys got=%" PRIu64 " expected=%" PRIu64, agg.mv_agg_keys, expected_keys);
  }

  mdb_txn_abort(rtxn);
}

static void write_csv_row(FILE *csv,
                          const char *run_id,
                          const char *mode_label,
                          unsigned dbi_flags,
                          unsigned env_flags,
                          const char *order_ins,
                          const char *order_del,
                          const char *phase,
                          uint64_t n,
                          size_t value_size,
                          uint64_t txn_batch,
                          uint64_t seed,
                          uint64_t ops,
                          double elapsed_ms,
                          uint64_t rss_before_kb,
                          uint64_t rss_after_kb,
                          const MDB_stat *st,
                          const MDB_envinfo *ei) {
  double ops_per_sec = elapsed_ms > 0.0 ? (double)ops / (elapsed_ms / 1000.0) : 0.0;

  fprintf(csv,
          "%s,%s,0x%x,0x%x,%s,%s,%s,"
          "%" PRIu64 ",%zu,%" PRIu64 ",%" PRIu64 ","
          "%" PRIu64 ",%.3f,%.3f,"
          "%" PRIu64 ",%" PRIu64 ","
          "%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ","
          "%" PRIu64 ",%" PRIu64 "\n",
          run_id,
          mode_label,
          dbi_flags,
          env_flags,
          order_ins,
          order_del,
          phase,
          n,
          value_size,
          txn_batch,
          seed,
          ops,
          elapsed_ms,
          ops_per_sec,
          rss_before_kb,
          rss_after_kb,
          st ? (uint64_t)st->ms_branch_pages : (uint64_t)0,
          st ? (uint64_t)st->ms_leaf_pages : (uint64_t)0,
          st ? (uint64_t)st->ms_overflow_pages : (uint64_t)0,
          st ? (uint64_t)st->ms_entries : (uint64_t)0,
          ei ? (uint64_t)ei->me_last_pgno : (uint64_t)0,
          ei ? (uint64_t)ei->me_mapsize : (uint64_t)0);
}

static void get_stats(MDB_env *env, MDB_dbi dbi, MDB_stat *st_out, MDB_envinfo *ei_out) {
  MDB_txn *rtxn = NULL;
  mdb_chk(mdb_txn_begin(env, NULL, MDB_RDONLY, &rtxn), "mdb_txn_begin(read stats)");
  mdb_chk(mdb_stat(rtxn, dbi, st_out), "mdb_stat");
  mdb_chk(mdb_env_info(env, ei_out), "mdb_env_info");
  mdb_txn_abort(rtxn);
}

static void run_one(MDB_env *env, MDB_dbi dbi, const bench_mode_t *mode, const cfg_t *cfg,
                    const char *run_id, int ins_order, FILE *csv) {
  const uint64_t N = cfg->n;

  uint64_t *keys = (uint64_t *)malloc((size_t)N * sizeof(uint64_t));
  if (!keys) die("malloc keys (%" PRIu64 ") failed", N);
  for (uint64_t i = 0; i < N; i++) keys[i] = i;

  if (ins_order == 1) {
    for (uint64_t i = 0; i < N / 2; i++) {
      uint64_t tmp = keys[i];
      keys[i] = keys[N - 1 - i];
      keys[N - 1 - i] = tmp;
    }
  } else if (ins_order == 2) {
    shuffle_u64(keys, (size_t)N, cfg->seed ? cfg->seed : 1);
  }

  const char *ins_order_s = order_to_str(ins_order);
  int del_order = (cfg->delete_order >= 0) ? cfg->delete_order : ins_order;
  const char *del_order_s = order_to_str(del_order);

  uint8_t *valbuf = NULL;
  if (cfg->value_size) {
    valbuf = (uint8_t *)malloc(cfg->value_size);
    if (!valbuf) die("malloc valbuf (%zu) failed", cfg->value_size);
  }

  // INSERT phase
  uint64_t rss0 = rss_kb_linux();
  uint64_t t0 = now_ns();

  MDB_txn *txn = NULL;
  uint64_t in_txn = 0;
  mdb_chk(mdb_txn_begin(env, NULL, 0, &txn), "mdb_txn_begin(write insert)");

  for (uint64_t i = 0; i < N; i++) {
    uint8_t kbuf[8];
    encode_be64(keys[i], kbuf);
    MDB_val k = { .mv_size = 8, .mv_data = kbuf };

    if (cfg->value_size) {
      fill_value(valbuf, cfg->value_size, keys[i], mode->use_hashsum);
    }
    MDB_val v = { .mv_size = cfg->value_size, .mv_data = valbuf };

    int rc = mdb_put(txn, dbi, &k, &v, MDB_NOOVERWRITE);
    if (rc != MDB_SUCCESS) {
      fprintf(stderr, "mdb_put failed at i=%" PRIu64 " key=%" PRIu64 ": %s (%d)\n",
              i, keys[i], mdb_strerror(rc), rc);
      mdb_txn_abort(txn);
      free(valbuf);
      free(keys);
      exit(2);
    }

    in_txn++;
    if (cfg->txn_batch && in_txn >= cfg->txn_batch && i + 1 < N) {
      mdb_chk(mdb_txn_commit(txn), "mdb_txn_commit(insert)");
      mdb_chk(mdb_txn_begin(env, NULL, 0, &txn), "mdb_txn_begin(insert next)");
      in_txn = 0;
    }
  }

  mdb_chk(mdb_txn_commit(txn), "mdb_txn_commit(insert final)");

  uint64_t t1 = now_ns();
  uint64_t rss1 = rss_kb_linux();

  MDB_stat st_ins;
  MDB_envinfo ei_ins;
  get_stats(env, dbi, &st_ins, &ei_ins);

  double ins_ms = (double)(t1 - t0) / 1e6;
  write_csv_row(csv, run_id, mode->label, mode->dbi_flags, cfg->env_flags,
                ins_order_s, del_order_s, "insert",
                N, cfg->value_size, cfg->txn_batch, cfg->seed,
                N, ins_ms, rss0, rss1, &st_ins, &ei_ins);
  fflush(csv);

  if (cfg->verify) {
    maybe_verify_totals(env, dbi, N, N);
  }

  // DELETE phase
  if (cfg->do_delete) {
    uint64_t *del_keys = keys;
    uint64_t *del_keys_owned = NULL;

    if (del_order == 0) {
      if (ins_order != 0) {
        del_keys_owned = (uint64_t *)malloc((size_t)N * sizeof(uint64_t));
        if (!del_keys_owned) die("malloc del_keys failed");
        for (uint64_t i = 0; i < N; i++) del_keys_owned[i] = i;
        del_keys = del_keys_owned;
      }
    } else if (del_order == 1) {
      del_keys_owned = (uint64_t *)malloc((size_t)N * sizeof(uint64_t));
      if (!del_keys_owned) die("malloc del_keys failed");
      for (uint64_t i = 0; i < N; i++) del_keys_owned[i] = (N - 1 - i);
      del_keys = del_keys_owned;
    } else if (del_order == 2) {
      del_keys_owned = (uint64_t *)malloc((size_t)N * sizeof(uint64_t));
      if (!del_keys_owned) die("malloc del_keys failed");
      for (uint64_t i = 0; i < N; i++) del_keys_owned[i] = i;
      shuffle_u64(del_keys_owned, (size_t)N, (cfg->seed ? cfg->seed : 1) ^ 0xfeedbeefULL);
      del_keys = del_keys_owned;
    }

    rss0 = rss_kb_linux();
    t0 = now_ns();

    txn = NULL;
    in_txn = 0;
    mdb_chk(mdb_txn_begin(env, NULL, 0, &txn), "mdb_txn_begin(write delete)");

    for (uint64_t i = 0; i < N; i++) {
      uint8_t kbuf[8];
      encode_be64(del_keys[i], kbuf);
      MDB_val k = { .mv_size = 8, .mv_data = kbuf };

      int rc = mdb_del(txn, dbi, &k, NULL);
      if (rc != MDB_SUCCESS) {
        fprintf(stderr, "mdb_del failed at i=%" PRIu64 " key=%" PRIu64 ": %s (%d)\n",
                i, del_keys[i], mdb_strerror(rc), rc);
        mdb_txn_abort(txn);
        free(del_keys_owned);
        free(valbuf);
        free(keys);
        exit(3);
      }

      in_txn++;
      if (cfg->txn_batch && in_txn >= cfg->txn_batch && i + 1 < N) {
        mdb_chk(mdb_txn_commit(txn), "mdb_txn_commit(delete)");
        mdb_chk(mdb_txn_begin(env, NULL, 0, &txn), "mdb_txn_begin(delete next)");
        in_txn = 0;
      }
    }

    mdb_chk(mdb_txn_commit(txn), "mdb_txn_commit(delete final)");

    t1 = now_ns();
    rss1 = rss_kb_linux();

    MDB_stat st_del;
    MDB_envinfo ei_del;
    get_stats(env, dbi, &st_del, &ei_del);

    double del_ms = (double)(t1 - t0) / 1e6;
    write_csv_row(csv, run_id, mode->label, mode->dbi_flags, cfg->env_flags,
                  ins_order_s, del_order_s, "delete",
                  N, cfg->value_size, cfg->txn_batch, cfg->seed,
                  N, del_ms, rss0, rss1, &st_del, &ei_del);
    fflush(csv);

    free(del_keys_owned);
  }

  free(valbuf);
  free(keys);
}

static uint64_t parse_u64(const char *s, const char *what) {
  errno = 0;
  char *end = NULL;
  unsigned long long v = strtoull(s, &end, 10);
  if (errno || !end || *end) die("bad %s: %s", what, s);
  return (uint64_t)v;
}

static size_t parse_size(const char *s, const char *what) {
  errno = 0;
  char *end = NULL;
  unsigned long long v = strtoull(s, &end, 10);
  if (errno || !end) die("bad %s: %s", what, s);
  unsigned long long mul = 1;
  if (*end) {
    if (end[1] != '\0') die("bad %s: %s", what, s);
    if (*end == 'k' || *end == 'K') mul = 1024ull;
    else if (*end == 'm' || *end == 'M') mul = 1024ull * 1024ull;
    else if (*end == 'g' || *end == 'G') mul = 1024ull * 1024ull * 1024ull;
    else die("bad %s suffix: %s", what, s);
  }
  return (size_t)(v * mul);
}

static unsigned parse_env_flags(int argc, char **argv) {
  unsigned f = 0;
  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "--nosync") == 0) f |= MDB_NOSYNC;
    else if (strcmp(argv[i], "--nometasync") == 0) f |= MDB_NOMETASYNC;
    else if (strcmp(argv[i], "--writemap") == 0) f |= MDB_WRITEMAP;
    else if (strcmp(argv[i], "--mapasync") == 0) f |= MDB_MAPASYNC;
    else if (strcmp(argv[i], "--notls") == 0) f |= MDB_NOTLS;
    else if (strcmp(argv[i], "--nordahead") == 0) f |= MDB_NORDAHEAD;
  }
  return f;
}

static void usage(const char *argv0) {
  fprintf(stderr,
          "Usage: %s [options]\n\n"
          "Required:\n"
          "  --path DIR            Base directory for per-run LMDB envs\n"
          "\nMain params:\n"
          "  --csv FILE            Output CSV path (default: bench.csv)\n"
          "  --n N                 Number of records (default: 100000)\n"
          "  --value-size BYTES    Value size, supports k/m/g suffix (default: 128)\n"
          "  --txn-batch N         Records per write txn commit (default: 10000)\n"
          "  --order LIST          Comma list: asc,desc,rand (default: asc)\n"
          "  --delete-order ORD    same|asc|desc|rand (default: same)\n"
          "  --no-delete           Skip delete phase\n"
          "\nAggregate modes:\n"
          "  --with-hashsum        Also benchmark ENTRIES|KEYS|HASHSUM\n"
          "  --hash-offset N       Hash offset for HASHSUM (default: 0)\n"
          "\nEnv flags (optional):\n"
          "  --nosync --nometasync --writemap --mapasync --notls --nordahead\n"
          "\nOther:\n"
          "  --seed N              Seed for random order (default: 1)\n"
          "  --keep-env            Do not delete per-run directories\n"
          "  --verify              Light correctness check with mdb_agg_totals\n\n",
          argv0);
  exit(2);
}

int main(int argc, char **argv) {
  cfg_t cfg;
  memset(&cfg, 0, sizeof(cfg));

  cfg.base_path = NULL;
  cfg.csv_path = "bench.csv";
  cfg.db_name = "bench";
  cfg.n = 100000;
  cfg.value_size = 128;
  cfg.txn_batch = 10000;
  cfg.seed = 1;
  cfg.env_flags = 0;
  cfg.delete_order = -1; // same
  cfg.do_delete = true;
  cfg.keep_env = false;
  cfg.verify = false;
  cfg.run_hashsum_mode = false;
  cfg.hash_offset = 0;

  int default_orders[1] = {0};
  cfg.orders = default_orders;
  cfg.orders_len = 1;

  for (int i = 1; i < argc; i++) {
    const char *a = argv[i];
    if (strcmp(a, "--path") == 0 && i + 1 < argc) {
      cfg.base_path = argv[++i];
    } else if (strcmp(a, "--csv") == 0 && i + 1 < argc) {
      cfg.csv_path = argv[++i];
    } else if (strcmp(a, "--n") == 0 && i + 1 < argc) {
      cfg.n = parse_u64(argv[++i], "n");
    } else if (strcmp(a, "--value-size") == 0 && i + 1 < argc) {
      cfg.value_size = parse_size(argv[++i], "value-size");
    } else if (strcmp(a, "--txn-batch") == 0 && i + 1 < argc) {
      cfg.txn_batch = parse_u64(argv[++i], "txn-batch");
    } else if (strcmp(a, "--seed") == 0 && i + 1 < argc) {
      cfg.seed = parse_u64(argv[++i], "seed");
    } else if (strcmp(a, "--order") == 0 && i + 1 < argc) {
      const char *list = argv[++i];
      char *tmp = strdup(list);
      if (!tmp) die("strdup order");
      int *orders = (int *)malloc(3 * sizeof(int));
      if (!orders) die("malloc orders");
      size_t cnt = 0;
      char *save = NULL;
      for (char *tok = strtok_r(tmp, ",", &save); tok; tok = strtok_r(NULL, ",", &save)) {
        int o = parse_order_token(tok);
        if (o < 0) die("bad --order token: %s", tok);
        bool dup = false;
        for (size_t k = 0; k < cnt; k++) if (orders[k] == o) dup = true;
        if (!dup) orders[cnt++] = o;
      }
      if (cnt == 0) die("empty --order");
      cfg.orders = orders;
      cfg.orders_len = cnt;
      free(tmp);
    } else if (strcmp(a, "--delete-order") == 0 && i + 1 < argc) {
      const char *o = argv[++i];
      if (strcmp(o, "same") == 0) cfg.delete_order = -1;
      else {
        int v = parse_order_token(o);
        if (v < 0) die("bad --delete-order: %s", o);
        cfg.delete_order = v;
      }
    } else if (strcmp(a, "--no-delete") == 0) {
      cfg.do_delete = false;
    } else if (strcmp(a, "--keep-env") == 0) {
      cfg.keep_env = true;
    } else if (strcmp(a, "--verify") == 0) {
      cfg.verify = true;
    } else if (strcmp(a, "--with-hashsum") == 0) {
      cfg.run_hashsum_mode = true;
    } else if (strcmp(a, "--hash-offset") == 0 && i + 1 < argc) {
      cfg.hash_offset = (unsigned)parse_u64(argv[++i], "hash-offset");
    } else if (strcmp(a, "--help") == 0 || strcmp(a, "-h") == 0) {
      usage(argv[0]);
    }
  }

  if (!cfg.base_path) usage(argv[0]);
  cfg.env_flags = parse_env_flags(argc, argv);

  mkdir_p(cfg.base_path);

  if (cfg.run_hashsum_mode && cfg.value_size < (size_t)(cfg.hash_offset + MDB_HASH_SIZE)) {
    die("--with-hashsum requires value-size >= hash-offset + MDB_HASH_SIZE (%u)",
        (unsigned)(cfg.hash_offset + MDB_HASH_SIZE));
  }

  bench_mode_t modes[3];
  size_t modes_len = 0;
  modes[modes_len++] = (bench_mode_t){ .label = "plain", .dbi_flags = 0u, .use_hashsum = false, .hash_offset = 0 };
  modes[modes_len++] = (bench_mode_t){ .label = "agg", .dbi_flags = (MDB_AGG_ENTRIES | MDB_AGG_KEYS), .use_hashsum = false, .hash_offset = 0 };
  if (cfg.run_hashsum_mode) {
    modes[modes_len++] = (bench_mode_t){ .label = "agg_hash", .dbi_flags = (MDB_AGG_ENTRIES | MDB_AGG_KEYS | MDB_AGG_HASHSUM),
                                  .use_hashsum = true, .hash_offset = cfg.hash_offset };
  }

  bool need_header = file_is_empty_or_missing(cfg.csv_path);
  FILE *csv = fopen(cfg.csv_path, "a");
  if (!csv) die("open csv %s: %s", cfg.csv_path, strerror(errno));
  if (need_header) {
    fprintf(csv,
            "run_id,mode,dbi_flags,env_flags,insert_order,delete_order,phase,"
            "n,value_size,txn_batch,seed,ops,elapsed_ms,ops_per_sec,"
            "rss_before_kb,rss_after_kb,branch_pages,leaf_pages,overflow_pages,stat_entries,"
            "env_last_pgno,env_mapsize\n");
    fflush(csv);
  }

  for (size_t mi = 0; mi < modes_len; mi++) {
    const bench_mode_t *mode = &modes[mi];
    for (size_t oi = 0; oi < cfg.orders_len; oi++) {
      int ord = cfg.orders[oi];

      char *envdir = xasprintf("%s/%s_n%" PRIu64 "_%s", cfg.base_path, mode->label, cfg.n, order_to_str(ord));

      rm_rf(envdir);
      mkdir_p(envdir);

      MDB_env *env = NULL;
      mdb_chk(mdb_env_create(&env), "mdb_env_create");
      mdb_chk(mdb_env_set_maxdbs(env, 8), "mdb_env_set_maxdbs");

      uint64_t approx_per = 128ull + (uint64_t)cfg.value_size + 64ull;
      uint64_t approx_total = approx_per * cfg.n;
      uint64_t mapsize = approx_total + (256ull << 20);
      if (mapsize < (512ull << 20)) mapsize = (512ull << 20);
      mdb_chk(mdb_env_set_mapsize(env, (mdb_size_t)mapsize), "mdb_env_set_mapsize");

      mdb_chk(mdb_env_open(env, envdir, cfg.env_flags, 0600), "mdb_env_open");

      MDB_txn *txn = NULL;
      mdb_chk(mdb_txn_begin(env, NULL, 0, &txn), "mdb_txn_begin(create db)");
      MDB_dbi dbi;
      mdb_chk(mdb_dbi_open(txn, cfg.db_name, MDB_CREATE | mode->dbi_flags, &dbi), "mdb_dbi_open");
      if (mode->use_hashsum) {
        mdb_chk(mdb_set_hash_offset(txn, dbi, mode->hash_offset), "mdb_set_hash_offset");
      }
      mdb_chk(mdb_txn_commit(txn), "mdb_txn_commit(create db)");

      char run_id[256];
      time_t wall = time(NULL);
      struct tm tm;
      localtime_r(&wall, &tm);
      snprintf(run_id, sizeof(run_id), "%04d%02d%02dT%02d%02d%02d_%s_%s",
               tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday,
               tm.tm_hour, tm.tm_min, tm.tm_sec,
               mode->label, order_to_str(ord));

      run_one(env, dbi, mode, &cfg, run_id, ord, csv);

      mdb_dbi_close(env, dbi);
      mdb_env_close(env);

      if (!cfg.keep_env) {
        rm_rf(envdir);
      }

      free(envdir);
    }
  }

  fclose(csv);

  if (cfg.orders != default_orders) free(cfg.orders);

  fprintf(stderr, "Wrote CSV: %s\n", cfg.csv_path);
  return 0;
}
