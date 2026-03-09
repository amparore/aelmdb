/* mtest_unit_keyhash.c - deterministic unit tests for LMDB multi-aggregate extension
 *
 * This suite is dedicated to the KEY-SOURCED HASHSUM modality:
 *   - MDB_AGG_HASHSUM with MDB_AGG_HASHSOURCE_FROM_KEY
 *
 * Goals:
 *   1) Exercise aggregate correctness (totals, prefix, range, rank/select)
 *      when hash bytes come from the key.
 *   2) Stress non-zero (and varying) md_hash_offset values heavily.
 *
 * Notes:
 *   - DUPSORT databases are intentionally not used (and are expected to be incompatible).
 *   - Values are varied aggressively (including overflow values) to ensure hashsum is
 *     independent of data in key-hash mode.
 *
 * Build (example):
 *   cc -O2 -std=c99 -I. mtest_unit_keyhash.c mdb.c midl.c -o mtest_unit_keyhash
 *   cc -g -std=c99 -DMDB_DEBUG_AGG_PRINT -DMDB_DEBUG_AGG_INTEGRITY -DMDB_DEBUG -I. mtest_unit_keyhash.c mdb.c midl.c -o mtest_unit_keyhash
 */

#include "lmdb.h"

#include <errno.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#define ENV_BASE "/tmp/testdb_agg_keyhash"

#define CHECK(rc, msg) do {                                                     \
    if ((rc) != MDB_SUCCESS) {                                                  \
        fprintf(stderr, "%s: %s (%d)\n", (msg), mdb_strerror((rc)), (rc));       \
        exit(EXIT_FAILURE);                                                     \
    }                                                                           \
} while (0)

#define REQUIRE(cond, msg) do {                                                 \
    if (!(cond)) {                                                              \
        fprintf(stderr, "%s\n", (msg));                                         \
        exit(EXIT_FAILURE);                                                     \
    }                                                                           \
} while (0)

#ifdef MDB_DEBUG_AGG_INTEGRITY
#define DBG_CHECK(txn, dbi, msg) do { CHECK(mdb_dbg_check_agg_db((txn), (dbi)), (msg)); } while (0)
#else
#define DBG_CHECK(txn, dbi, msg) do { (void)(txn); (void)(dbi); (void)(msg); } while (0)
#endif

static void fatal_errno(const char *msg)
{
    fprintf(stderr, "%s: %s\n", msg, strerror(errno));
    exit(EXIT_FAILURE);
}

static void expect_rc(int rc, int expect, const char *msg)
{
    if (rc != expect) {
        fprintf(stderr, "%s: expected %s (%d) got %s (%d)\n",
                msg, mdb_strerror(expect), expect, mdb_strerror(rc), rc);
        exit(EXIT_FAILURE);
    }
}

static void ensure_dir(const char *path)
{
    if (mkdir(path, 0775) && errno != EEXIST)
        fatal_errno("mkdir");
    if (chmod(path, 0775) && errno != EPERM)
        fatal_errno("chmod");
}

static void unlink_env_files(const char *dir)
{
    char path[512];
    snprintf(path, sizeof(path), "%s/data.mdb", dir);
    unlink(path);
    snprintf(path, sizeof(path), "%s/lock.mdb", dir);
    unlink(path);
}

/* Make per-test directory under ENV_BASE. */
static void make_test_dir(char *out, size_t cap, const char *name)
{
    ensure_dir(ENV_BASE);
    snprintf(out, cap, "%s/%s", ENV_BASE, name);
    ensure_dir(out);
}

/* ------------------------------ deterministic RNG helpers ------------------------------ */

static uint64_t fnv1a64(const char *s)
{
    uint64_t h = 1469598103934665603ULL;
    for (; *s; s++) {
        h ^= (unsigned char)*s;
        h *= 1099511628211ULL;
    }
    return h;
}

static uint64_t rng_next_u64(uint64_t *st)
{
    /* xorshift64* */
    uint64_t x = *st;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *st = x;
    return x * 2685821657736338717ULL;
}

static int rng_bit(uint64_t *st) { return (int)(rng_next_u64(st) & 1ULL); }
static size_t rng_index(uint64_t *st, size_t n) { return (size_t)(rng_next_u64(st) % (n ? n : 1)); }

/* ------------------------------ key/value builders ------------------------------ */

static void fill_hash_prefix(uint8_t out[MDB_HASH_SIZE], uint64_t x)
{
    /* portable 8-byte little-endian fill (MDB_HASH_SIZE expected 8) */
    for (unsigned i = 0; i < (unsigned)MDB_HASH_SIZE; i++) {
        out[i] = (uint8_t)(x & 0xFFu);
        x >>= 8;
    }
}

static void fill_prefix_bytes(uint8_t *dst, size_t n, uint64_t seed)
{
    /* deterministic non-trivial bytes, can include zeros (good for stress) */
    uint64_t x = seed ? seed : 0x9E3779B97F4A7C15ULL;
    for (size_t i = 0; i < n; i++) {
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        dst[i] = (uint8_t)(x & 0xFFu) ^ (uint8_t)(i * 13u);
    }
}

static void make_key_with_hash_offset_salted(uint8_t *buf, size_t cap, size_t off,
                                             const char *tag, unsigned id, uint64_t salt,
                                             MDB_val *out)
{
    REQUIRE(off > 0, "keyhash tests require non-zero offsets");
    REQUIRE(cap >= off + (size_t)MDB_HASH_SIZE + 64u, "buffer too small for key");

    /* Prefix controls ordering/split patterns. */
    fill_prefix_bytes(buf, off, (((uint64_t)id) << 1) ^ 0xBADC0FFEE0DDF00DULL ^ salt);

    /* Hash bytes at offset. */
    fill_hash_prefix(buf + off, (((uint64_t)id << 32) ^ 0xA55AA55Au) ^ salt);

    /* Suffix makes keys unique & size-stable. Include NUL in key size. */
    int n = snprintf((char *)buf + off + (size_t)MDB_HASH_SIZE,
                     cap - (off + (size_t)MDB_HASH_SIZE),
                     "/%s/%08u/%016" PRIx64, tag, id, (uint64_t)salt);
    REQUIRE(n > 0, "snprintf failed");
    out->mv_data = buf;
    out->mv_size = off + (size_t)MDB_HASH_SIZE + (size_t)n + 1u; /* include NUL */
}

static void make_small_val(uint8_t *buf, size_t cap, const char *tag, unsigned id, uint64_t salt, MDB_val *out)
{
    REQUIRE(cap >= 64u, "buffer too small for value");
    int n = snprintf((char *)buf, cap, "VAL/%s/%08u/%016" PRIx64, tag, id, (uint64_t)salt);
    REQUIRE(n > 0, "snprintf failed");
    out->mv_data = buf;
    out->mv_size = (size_t)n + 1u; /* include NUL */
}

static void make_big_val(uint8_t *buf, size_t cap, const char *tag, unsigned id, uint64_t salt, size_t target, MDB_val *out)
{
    REQUIRE(cap >= target, "buffer too small for big value");
    /* place a readable header then fill the rest deterministically */
    int n = snprintf((char *)buf, cap, "BIG/%s/%08u/%016" PRIx64 "/", tag, id, (uint64_t)salt);
    REQUIRE(n > 0, "snprintf failed");
    size_t pos = (size_t)n + 1u; /* include NUL */
    if (pos < target) {
        fill_prefix_bytes(buf + pos, target - pos, ((uint64_t)id << 7) ^ salt ^ 0x1234567890ABCDEFULL);
    }
    out->mv_data = buf;
    out->mv_size = target;
}

/* ------------------------------ environment helpers ------------------------------ */

static MDB_env *open_env_dir_flags(const char *dir, int maxdbs, size_t mapsize, int fresh, unsigned env_flags)
{
    MDB_env *env = NULL;

    ensure_dir(ENV_BASE);
    ensure_dir(dir);

    if (fresh)
        unlink_env_files(dir);

    CHECK(mdb_env_create(&env), "env_create");
    CHECK(mdb_env_set_maxdbs(env, maxdbs), "env_set_maxdbs");
    CHECK(mdb_env_set_mapsize(env, mapsize), "env_set_mapsize");
    CHECK(mdb_env_open(env, dir, env_flags, 0664), "env_open");

    return env;
}

/* ------------------------------ oracle (cursor enumeration) ------------------------------ */

typedef struct {
    unsigned char *k;
    size_t klen;
    unsigned char *d;
    size_t dlen;
} Rec;

typedef struct {
    Rec *v;
    size_t len;
    size_t cap;
} RecVec;

static void recvec_init(RecVec *vv)
{
    vv->v = NULL;
    vv->len = 0;
    vv->cap = 0;
}

static void rec_free(Rec *r)
{
    free(r->k);
    free(r->d);
    r->k = NULL; r->d = NULL;
    r->klen = r->dlen = 0;
}

static void recvec_free(RecVec *vv)
{
    if (!vv) return;
    for (size_t i = 0; i < vv->len; i++)
        rec_free(&vv->v[i]);
    free(vv->v);
    vv->v = NULL;
    vv->len = vv->cap = 0;
}

static void recvec_push_copy(RecVec *vv, const MDB_val *k, const MDB_val *d)
{
    if (vv->len == vv->cap) {
        size_t ncap = vv->cap ? vv->cap * 2u : 64u;
        Rec *nv = (Rec *)realloc(vv->v, ncap * sizeof(Rec));
        REQUIRE(nv != NULL, "realloc failed");
        vv->v = nv;
        vv->cap = ncap;
    }
    Rec *r = &vv->v[vv->len++];
    memset(r, 0, sizeof(*r));

    r->klen = k->mv_size;
    r->k = (unsigned char *)malloc(r->klen);
    REQUIRE(r->k != NULL, "malloc key failed");
    memcpy(r->k, k->mv_data, r->klen);

    r->dlen = d->mv_size;
    r->d = (unsigned char *)malloc(r->dlen ? r->dlen : 1u);
    REQUIRE(r->d != NULL, "malloc data failed");
    if (r->dlen)
        memcpy(r->d, d->mv_data, r->dlen);
}

static void oracle_build(MDB_txn *txn, MDB_dbi dbi, RecVec *out)
{
    MDB_cursor *cur = NULL;
    MDB_val k, d;
    int rc;

    recvec_init(out);
    CHECK(mdb_cursor_open(txn, dbi, &cur), "oracle: cursor_open");

    rc = mdb_cursor_get(cur, &k, &d, MDB_FIRST);
    while (rc == MDB_SUCCESS) {
        recvec_push_copy(out, &k, &d);
        rc = mdb_cursor_get(cur, &k, &d, MDB_NEXT);
    }
    if (rc != MDB_NOTFOUND)
        CHECK(rc, "oracle: cursor walk");

    mdb_cursor_close(cur);
}

static void oracle_hashsum_total_key(const RecVec *vv, size_t off, uint8_t out[MDB_HASH_SIZE])
{
    memset(out, 0, MDB_HASH_SIZE);
    for (size_t i = 0; i < vv->len; i++) {
        const Rec *r = &vv->v[i];
        REQUIRE(r->klen >= off + (size_t)MDB_HASH_SIZE, "oracle: key too small for hashsum");
        mdb_hashsum_add(out, (const uint8_t *)r->k + off);
    }
}

static int rec_key_cmp(MDB_txn *txn, MDB_dbi dbi, const Rec *r, const MDB_val *b)
{
    MDB_val rk;
    rk.mv_size = r->klen;
    rk.mv_data = r->k;
    return mdb_cmp(txn, dbi, &rk, (MDB_val *)b);
}

static int key_equal_rec(const Rec *a, const Rec *b)
{
    return a->klen == b->klen && memcmp(a->k, b->k, a->klen) == 0;
}

static void oracle_range_agg_keyhash(MDB_txn *txn, MDB_dbi dbi, const RecVec *vv,
                                     const MDB_val *low_key, const MDB_val *high_key,
                                     unsigned flags, size_t off, MDB_agg *out)
{
    const int lincl = (flags & MDB_RANGE_LOWER_INCL) != 0;
    const int hincl = (flags & MDB_RANGE_UPPER_INCL) != 0;

    memset(out, 0, sizeof(*out));
    out->mv_flags = 0; /* filled by caller if needed */

    uint8_t hacc[MDB_HASH_SIZE];
    memset(hacc, 0, MDB_HASH_SIZE);

    uint64_t entries = 0, keys = 0;
    int have_last = 0;
    const Rec *last = NULL;

    for (size_t i = 0; i < vv->len; i++) {
        const Rec *r = &vv->v[i];

        int cl = low_key ? rec_key_cmp(txn, dbi, r, low_key) : 1;
        int ch = high_key ? rec_key_cmp(txn, dbi, r, high_key) : -1;

        int ok_low = low_key ? (cl > 0 || (lincl && cl == 0)) : 1;
        int ok_high = high_key ? (ch < 0 || (hincl && ch == 0)) : 1;

        if (ok_low && ok_high) {
            entries++;
            REQUIRE(r->klen >= off + (size_t)MDB_HASH_SIZE, "oracle: key too small for range hashsum");
            mdb_hashsum_add(hacc, (const uint8_t *)r->k + off);

            if (!have_last || !key_equal_rec(r, last)) {
                keys++;
                have_last = 1;
                last = r;
            }
        }
    }

    out->mv_agg_entries = entries;
    out->mv_agg_keys = keys;
    memcpy(out->mv_agg_hashes, hacc, MDB_HASH_SIZE);
}

static void oracle_prefix_agg_keyhash(MDB_txn *txn, MDB_dbi dbi, const RecVec *vv,
                                      const MDB_val *key, unsigned flags, size_t off,
                                      MDB_agg *out)
{
    const int incl = (flags & MDB_AGG_PREFIX_INCL) != 0;

    memset(out, 0, sizeof(*out));
    out->mv_flags = 0; /* caller */

    uint8_t hacc[MDB_HASH_SIZE];
    memset(hacc, 0, MDB_HASH_SIZE);

    uint64_t entries = 0, keys = 0;
    int have_last = 0;
    const Rec *last = NULL;

    for (size_t i = 0; i < vv->len; i++) {
        const Rec *r = &vv->v[i];
        int c = rec_key_cmp(txn, dbi, r, key);

        if (c < 0 || (incl && c == 0)) {
            entries++;
            REQUIRE(r->klen >= off + (size_t)MDB_HASH_SIZE, "oracle: key too small for prefix hashsum");
            mdb_hashsum_add(hacc, (const uint8_t *)r->k + off);

            if (!have_last || !key_equal_rec(r, last)) {
                keys++;
                have_last = 1;
                last = r;
            }
        }
    }

    out->mv_agg_entries = entries;
    out->mv_agg_keys = keys;
    memcpy(out->mv_agg_hashes, hacc, MDB_HASH_SIZE);
}

/* ------------------------------ aggregate comparison helpers ------------------------------ */

static void agg_expect_flags(const char *label, unsigned got, unsigned expect)
{
    if (got != expect) {
        fprintf(stderr, "%s: agg flags mismatch: got=0x%x expect=0x%x\n", label, got, expect);
        exit(EXIT_FAILURE);
    }
}

static void agg_require_equal(unsigned schema, const MDB_agg *a, const MDB_agg *b, const char *msg)
{
    if (a->mv_flags != schema || b->mv_flags != schema) {
        fprintf(stderr, "%s: schema mismatch (a=0x%x b=0x%x expected=0x%x)\n",
                msg, a->mv_flags, b->mv_flags, schema);
        exit(EXIT_FAILURE);
    }
    if (schema & MDB_AGG_ENTRIES) {
        if (a->mv_agg_entries != b->mv_agg_entries) {
            fprintf(stderr, "%s: entries mismatch got=%" PRIu64 " exp=%" PRIu64 "\n",
                    msg, a->mv_agg_entries, b->mv_agg_entries);
            exit(EXIT_FAILURE);
        }
    }
    if (schema & MDB_AGG_KEYS) {
        if (a->mv_agg_keys != b->mv_agg_keys) {
            fprintf(stderr, "%s: keys mismatch got=%" PRIu64 " exp=%" PRIu64 "\n",
                    msg, a->mv_agg_keys, b->mv_agg_keys);
            exit(EXIT_FAILURE);
        }
    }
    if (schema & MDB_AGG_HASHSUM) {
        if (memcmp(a->mv_agg_hashes, b->mv_agg_hashes, MDB_HASH_SIZE) != 0) {
            fprintf(stderr, "%s: hashsum mismatch\n", msg);
            exit(EXIT_FAILURE);
        }
    }
}

/* ------------------------------ verification against oracle ------------------------------ */

#define AGG_SCHEMA_KEYHASH (MDB_AGG_ENTRIES|MDB_AGG_KEYS|MDB_AGG_HASHSUM|MDB_AGG_HASHSOURCE_FROM_KEY)

static void verify_rank_select_keyonly(MDB_txn *txn, MDB_dbi dbi, const RecVec *vv, const char *label)
{
    /* sample a handful of ranks (including edges) */
    const uint64_t samples[] = {0, 1, 2, 3, 7, 11, 31, 63, 127, 255, 511, 1023};
    for (size_t i = 0; i < sizeof(samples)/sizeof(samples[0]); i++) {
        uint64_t r = samples[i];
        if (vv->len == 0) {
            MDB_val k = {0, NULL}, d = {0, NULL};
            uint64_t di = UINT64_MAX;
            int rc = mdb_agg_select(txn, dbi, MDB_AGG_WEIGHT_ENTRIES, 0, &k, &d, &di);
            expect_rc(rc, MDB_NOTFOUND, "select on empty");
            return;
        }
        r %= (uint64_t)vv->len;

        const Rec *or = &vv->v[(size_t)r];
        MDB_val k = {0, NULL}, d = {0, NULL};
        uint64_t di = UINT64_MAX;

        CHECK(mdb_agg_select(txn, dbi, MDB_AGG_WEIGHT_ENTRIES, r, &k, &d, &di), "mdb_agg_select");
        if (k.mv_size != or->klen || memcmp(k.mv_data, or->k, or->klen) != 0) {
            fprintf(stderr, "%s: select(entries) key mismatch at r=%" PRIu64 "\n", label, r);
            exit(EXIT_FAILURE);
        }
        if (d.mv_size != or->dlen || (or->dlen && memcmp(d.mv_data, or->d, or->dlen) != 0)) {
            fprintf(stderr, "%s: select(entries) data mismatch at r=%" PRIu64 "\n", label, r);
            exit(EXIT_FAILURE);
        }
        if (di != 0) {
            fprintf(stderr, "%s: select(entries) dup_index expected 0 got %" PRIu64 "\n", label, di);
            exit(EXIT_FAILURE);
        }

        /* exact rank for returned (k,d) in key-only mode: pass data=NULL/0 */
        MDB_val qk = { or->klen, or->k };
        MDB_val qd = { 0, NULL };
        uint64_t rr = UINT64_MAX, rdi = UINT64_MAX;
        CHECK(mdb_agg_rank(txn, dbi, &qk, &qd, MDB_AGG_WEIGHT_ENTRIES, MDB_AGG_RANK_EXACT, &rr, &rdi),
              "mdb_agg_rank exact");
        if (rr != r) {
            fprintf(stderr, "%s: rank(entries exact) expected=%" PRIu64 " got=%" PRIu64 "\n", label, r, rr);
            exit(EXIT_FAILURE);
        }
        if (rdi != 0) {
            fprintf(stderr, "%s: rank dup_index expected 0 got %" PRIu64 "\n", label, rdi);
            exit(EXIT_FAILURE);
        }
    }
}

static void verify_against_oracle_keyhash(MDB_txn *txn, MDB_dbi dbi, const char *label, size_t off)
{
    unsigned schema = 0;
    CHECK(mdb_agg_info(txn, dbi, &schema), "mdb_agg_info");

    /* Must be in key-hash mode and include our expected schema bits. */
    REQUIRE((schema & MDB_AGG_HASHSUM) != 0, "schema missing MDB_AGG_HASHSUM");
    REQUIRE((schema & MDB_AGG_HASHSOURCE_FROM_KEY) != 0, "schema missing MDB_AGG_HASHSOURCE_FROM_KEY");

    /* Build oracle view (in-order). */
    RecVec vv;
    oracle_build(txn, dbi, &vv);

    /* Totals */
    MDB_agg got, exp;
    CHECK(mdb_agg_totals(txn, dbi, &got), "mdb_agg_totals");
    agg_expect_flags(label, got.mv_flags, schema);

    uint8_t exp_hash[MDB_HASH_SIZE];
    oracle_hashsum_total_key(&vv, off, exp_hash);

    memset(&exp, 0, sizeof(exp));
    exp.mv_flags = schema;
    exp.mv_agg_entries = (uint64_t)vv.len;
    exp.mv_agg_keys = (uint64_t)vv.len; /* no dupsort */
    memcpy(exp.mv_agg_hashes, exp_hash, MDB_HASH_SIZE);

    agg_require_equal(schema, &got, &exp, "totals");

    /* Randomized (but deterministic) prefix and range checks. */
    uint64_t st = fnv1a64(label) ^ ((uint64_t)vv.len * 0x9E3779B97F4A7C15ULL) ^ (uint64_t)off;
    if (!st) st = 1;
    size_t trials = 150;
    if (vv.len < 40) trials = 50;
    if (vv.len > 5000) trials = 100;

    for (size_t t = 0; t < trials; t++) {
        if (vv.len == 0)
            break;

        const int do_prefix = rng_bit(&st);

        if (do_prefix) {
            const Rec *r = &vv.v[rng_index(&st, vv.len)];
            MDB_val k = { r->klen, r->k };
            unsigned pflags = rng_bit(&st) ? MDB_AGG_PREFIX_INCL : 0;

            MDB_agg gotp, expp;
            CHECK(mdb_agg_prefix(txn, dbi, &k, NULL, pflags, &gotp), "mdb_agg_prefix");
            agg_expect_flags("prefix flags", gotp.mv_flags, schema);

            oracle_prefix_agg_keyhash(txn, dbi, &vv, &k, pflags, off, &expp);
            expp.mv_flags = schema;
            agg_require_equal(schema, &gotp, &expp, "prefix");
        } else {
            const Rec *a = &vv.v[rng_index(&st, vv.len)];
            const Rec *b = &vv.v[rng_index(&st, vv.len)];
            MDB_val ka = { a->klen, a->k };
            MDB_val kb = { b->klen, b->k };

            /* ensure ka <= kb */
            if (mdb_cmp(txn, dbi, &ka, &kb) > 0) {
                MDB_val tmp = ka; ka = kb; kb = tmp;
            }

            unsigned rflags = 0;
            if (rng_bit(&st)) rflags |= MDB_RANGE_LOWER_INCL;
            if (rng_bit(&st)) rflags |= MDB_RANGE_UPPER_INCL;

            MDB_agg gotr, expr;
            CHECK(mdb_agg_range(txn, dbi, &ka, NULL, &kb, NULL, rflags, &gotr), "mdb_agg_range");
            agg_expect_flags("range flags", gotr.mv_flags, schema);

            oracle_range_agg_keyhash(txn, dbi, &vv, &ka, &kb, rflags, off, &expr);
            expr.mv_flags = schema;
            agg_require_equal(schema, &gotr, &expr, "range");
        }
    }

    /* Rank/select checks (key-only). */
    verify_rank_select_keyonly(txn, dbi, &vv, label);

    recvec_free(&vv);
}

/* ------------------------------ tests ------------------------------ */

static void open_keyhash_db(MDB_txn *txn, const char *name, MDB_dbi *dbi_out, size_t off)
{
    MDB_dbi dbi;
    CHECK(mdb_dbi_open(txn, name, MDB_CREATE | AGG_SCHEMA_KEYHASH, &dbi), "dbi_open keyhash");
    CHECK(mdb_set_hash_offset(txn, dbi, off), "mdb_set_hash_offset");
    *dbi_out = dbi;
}

/* Insert N records with deterministic keys at given offset. */
static void insert_keys(MDB_txn *txn, MDB_dbi dbi, const char *tag, size_t off,
                        unsigned start, unsigned count, int descending,
                        int big_values, size_t big_size, int use_reserve)
{
    uint8_t kbuf[512];
    uint8_t vbuf[1 << 20]; /* up to 1MB value buffer for overflow tests */
    REQUIRE(sizeof(kbuf) >= off + (size_t)MDB_HASH_SIZE + 64u, "kbuf too small");

    for (unsigned j = 0; j < count; j++) {
        unsigned i = descending ? (start + count - 1u - j) : (start + j);
        uint64_t salt = ((uint64_t)i * 0xD6E8FEB86659FD93ULL) ^ 0x0123456789ABCDEFULL;

        MDB_val k, d;
        make_key_with_hash_offset_salted(kbuf, sizeof(kbuf), off, tag, i, salt, &k);

        if (use_reserve) {
            /* reserve path should be compatible in key-hash mode */
            d.mv_size = 96u;
            d.mv_data = NULL;
            int rc = mdb_put(txn, dbi, &k, &d, MDB_RESERVE);
            CHECK(rc, "mdb_put RESERVE");
            REQUIRE(d.mv_data != NULL, "reserve returned NULL");
            fill_prefix_bytes((uint8_t *)d.mv_data, d.mv_size, salt ^ 0xA5A5A5A5A5A5A5A5ULL);
        } else if (big_values) {
            make_big_val(vbuf, sizeof(vbuf), tag, i, salt, big_size, &d);
            CHECK(mdb_put(txn, dbi, &k, &d, 0), "mdb_put big");
        } else {
            make_small_val(vbuf, sizeof(vbuf), tag, i, salt, &d);
            CHECK(mdb_put(txn, dbi, &k, &d, 0), "mdb_put small");
        }

        DBG_CHECK(txn, dbi, "dbg agg after put");
    }
}

static void test_keyhash_rightsplit_offset17(void)
{
    char dir[256];
    make_test_dir(dir, sizeof(dir), "t01_rightsplit_off17");
    MDB_env *env = open_env_dir_flags(dir, 8, 128u * 1024u * 1024u, 1, 0);

    MDB_txn *txn = NULL;
    MDB_dbi dbi;
    const size_t off = 17;

    CHECK(mdb_txn_begin(env, NULL, 0, &txn), "txn_begin");
    open_keyhash_db(txn, "kh", &dbi, off);

    insert_keys(txn, dbi, "R", off, 0, 900, 0, 0, 0, 0);

    verify_against_oracle_keyhash(txn, dbi, "rightsplit_off17", off);

    CHECK(mdb_txn_commit(txn), "commit");
    mdb_dbi_close(env, dbi);
    mdb_env_close(env);
}

static void test_keyhash_leftsplit_offset33(void)
{
    char dir[256];
    make_test_dir(dir, sizeof(dir), "t02_leftsplit_off33");
    MDB_env *env = open_env_dir_flags(dir, 8, 128u * 1024u * 1024u, 1, 0);

    MDB_txn *txn = NULL;
    MDB_dbi dbi;
    const size_t off = 33;

    CHECK(mdb_txn_begin(env, NULL, 0, &txn), "txn_begin");
    open_keyhash_db(txn, "kh", &dbi, off);

    insert_keys(txn, dbi, "L", off, 0, 900, 1, 0, 0, 0);

    verify_against_oracle_keyhash(txn, dbi, "leftsplit_off33", off);

    CHECK(mdb_txn_commit(txn), "commit");
    mdb_dbi_close(env, dbi);
    mdb_env_close(env);
}

static void test_keyhash_delete_root_collapse_offset7(void)
{
    char dir[256];
    make_test_dir(dir, sizeof(dir), "t03_delete_collapse_off7");
    MDB_env *env = open_env_dir_flags(dir, 8, 128u * 1024u * 1024u, 1, 0);

    MDB_txn *txn = NULL;
    MDB_dbi dbi;
    const size_t off = 7;

    CHECK(mdb_txn_begin(env, NULL, 0, &txn), "txn_begin");
    open_keyhash_db(txn, "kh", &dbi, off);

    insert_keys(txn, dbi, "D", off, 0, 1200, 0, 0, 0, 0);
    verify_against_oracle_keyhash(txn, dbi, "delete_collapse_before", off);

    /* Delete most keys in a pattern to force merges/collapse. */
    uint8_t kbuf[512], vbuf[128];
    MDB_val k, d;
    for (unsigned i = 0; i < 1200; i++) {
        if ((i % 3u) == 1u || i < 50u || i > 1140u) {
            uint64_t salt = ((uint64_t)i * 0xD6E8FEB86659FD93ULL) ^ 0x0123456789ABCDEFULL;
            make_key_with_hash_offset_salted(kbuf, sizeof(kbuf), off, "D", i, salt, &k);
            make_small_val(vbuf, sizeof(vbuf), "X", i, salt ^ 0x55ULL, &d);

            /* use del by key only */
            int rc = mdb_del(txn, dbi, &k, NULL);
            CHECK(rc, "mdb_del");
            DBG_CHECK(txn, dbi, "dbg agg after del");
        }
    }

    verify_against_oracle_keyhash(txn, dbi, "delete_collapse_after", off);

    CHECK(mdb_txn_commit(txn), "commit");
    mdb_dbi_close(env, dbi);
    mdb_env_close(env);
}

static void test_keyhash_overwrite_no_hash_delta_offset31(void)
{
    char dir[256];
    make_test_dir(dir, sizeof(dir), "t04_overwrite_off31");
    MDB_env *env = open_env_dir_flags(dir, 8, 128u * 1024u * 1024u, 1, 0);

    MDB_txn *txn = NULL;
    MDB_dbi dbi;
    const size_t off = 31;

    CHECK(mdb_txn_begin(env, NULL, 0, &txn), "txn_begin");
    open_keyhash_db(txn, "kh", &dbi, off);

    insert_keys(txn, dbi, "O", off, 0, 400, 0, 0, 0, 0);
    verify_against_oracle_keyhash(txn, dbi, "overwrite_before", off);

    /* Overwrite data for some existing keys with wildly different sizes. */
    uint8_t kbuf[512];
    uint8_t vsmall[128];
    uint8_t vbig[4096];
    for (unsigned i = 0; i < 400; i += 7) {
        uint64_t salt = ((uint64_t)i * 0xD6E8FEB86659FD93ULL) ^ 0x0123456789ABCDEFULL;
        MDB_val k, d;
        make_key_with_hash_offset_salted(kbuf, sizeof(kbuf), off, "O", i, salt, &k);
        if ((i / 7u) & 1u) {
            make_small_val(vsmall, sizeof(vsmall), "NEW", i, salt ^ 0x1111ULL, &d);
        } else {
            make_big_val(vbig, sizeof(vbig), "NEWBIG", i, salt ^ 0x2222ULL, 3000u, &d);
        }
        CHECK(mdb_put(txn, dbi, &k, &d, 0), "overwrite put");
        DBG_CHECK(txn, dbi, "dbg agg after overwrite");
    }

    /* Also test MDB_CURRENT update via cursor. */
    MDB_cursor *cur = NULL;
    CHECK(mdb_cursor_open(txn, dbi, &cur), "cursor_open");
    MDB_val ck, cd;
    int rc = mdb_cursor_get(cur, &ck, &cd, MDB_FIRST);
    if (rc == MDB_SUCCESS) {
        /* update first entry's data */
        MDB_val nd;
        make_big_val(vbig, sizeof(vbig), "CUR", 0, 0x3333ULL, 3500u, &nd);
        CHECK(mdb_cursor_put(cur, &ck, &nd, MDB_CURRENT), "cursor_put MDB_CURRENT");
        DBG_CHECK(txn, dbi, "dbg agg after cursor current");
    } else {
        expect_rc(rc, MDB_NOTFOUND, "cursor_get first");
    }
    mdb_cursor_close(cur);

    /* Hashsum must be unchanged for a given key set; verify full agg. */
    verify_against_oracle_keyhash(txn, dbi, "overwrite_after", off);

    CHECK(mdb_txn_commit(txn), "commit");
    mdb_dbi_close(env, dbi);
    mdb_env_close(env);
}

static void test_keyhash_big_values_offset64(void)
{
    char dir[256];
    make_test_dir(dir, sizeof(dir), "t05_bigvals_off64");
    MDB_env *env = open_env_dir_flags(dir, 8, 256u * 1024u * 1024u, 1, 0);

    MDB_txn *txn = NULL;
    MDB_dbi dbi;
    const size_t off = 64;

    CHECK(mdb_txn_begin(env, NULL, 0, &txn), "txn_begin");
    open_keyhash_db(txn, "kh", &dbi, off);

    /* Insert overflow values; hashsum must still match key bytes. */
    insert_keys(txn, dbi, "B", off, 0, 260, 0, 1, 70000u, 0);

    verify_against_oracle_keyhash(txn, dbi, "bigvals_before_del", off);

    /* Delete all. */
    uint8_t kbuf[512];
    MDB_val k;
    for (unsigned i = 0; i < 260; i++) {
        uint64_t salt = ((uint64_t)i * 0xD6E8FEB86659FD93ULL) ^ 0x0123456789ABCDEFULL;
        make_key_with_hash_offset_salted(kbuf, sizeof(kbuf), off, "B", i, salt, &k);
        CHECK(mdb_del(txn, dbi, &k, NULL), "mdb_del bigvals");
        DBG_CHECK(txn, dbi, "dbg agg after del bigval");
    }

    verify_against_oracle_keyhash(txn, dbi, "bigvals_after_del", off);

    CHECK(mdb_txn_commit(txn), "commit");
    mdb_dbi_close(env, dbi);
    mdb_env_close(env);
}

static void test_keyhash_offsets_matrix(void)
{
    char dir[256];
    make_test_dir(dir, sizeof(dir), "t06_offsets_matrix");
    MDB_env *env = open_env_dir_flags(dir, 16, 256u * 1024u * 1024u, 1, 0);

    const size_t offs[] = {1, 5, 9, 17, 29, 73, 127, 191};
    for (size_t t = 0; t < sizeof(offs)/sizeof(offs[0]); t++) {
        MDB_txn *txn = NULL;
        MDB_dbi dbi;
        char name[32];
        const size_t off = offs[t];
        snprintf(name, sizeof(name), "db_off_%zu", off);

        CHECK(mdb_txn_begin(env, NULL, 0, &txn), "txn_begin");
        open_keyhash_db(txn, name, &dbi, off);

        insert_keys(txn, dbi, name, off, 1000u * (unsigned)t, 220, 0, 0, 0, 0);
        verify_against_oracle_keyhash(txn, dbi, name, off);

        CHECK(mdb_txn_commit(txn), "commit");
        mdb_dbi_close(env, dbi);
    }

    mdb_env_close(env);
}

static void test_keyhash_reserve_allowed_offset15(void)
{
    char dir[256];
    make_test_dir(dir, sizeof(dir), "t07_reserve_ok_off15");
    MDB_env *env = open_env_dir_flags(dir, 8, 128u * 1024u * 1024u, 1, 0);

    MDB_txn *txn = NULL;
    MDB_dbi dbi;
    const size_t off = 15;

    CHECK(mdb_txn_begin(env, NULL, 0, &txn), "txn_begin");
    open_keyhash_db(txn, "kh", &dbi, off);

    insert_keys(txn, dbi, "RSV", off, 0, 500, 0, 0, 0, 1);

    verify_against_oracle_keyhash(txn, dbi, "reserve_ok", off);

    CHECK(mdb_txn_commit(txn), "commit");
    mdb_dbi_close(env, dbi);
    mdb_env_close(env);
}

static void test_keyhash_writemap_allowed_offset23(void)
{
    char dir[256];
    make_test_dir(dir, sizeof(dir), "t08_writemap_ok_off23");
    MDB_env *env = open_env_dir_flags(dir, 8, 128u * 1024u * 1024u, 1, MDB_WRITEMAP);

    MDB_txn *txn = NULL;
    MDB_dbi dbi;
    const size_t off = 23;

    CHECK(mdb_txn_begin(env, NULL, 0, &txn), "txn_begin");
    open_keyhash_db(txn, "kh", &dbi, off);

    insert_keys(txn, dbi, "WM", off, 0, 400, 0, 0, 0, 0);

    verify_against_oracle_keyhash(txn, dbi, "writemap_ok", off);

    CHECK(mdb_txn_commit(txn), "commit");
    mdb_dbi_close(env, dbi);
    mdb_env_close(env);
}

static void test_keyhash_dupsort_incompatible(void)
{
    char dir[256];
    make_test_dir(dir, sizeof(dir), "t09_dupsort_incompat");
    MDB_env *env = open_env_dir_flags(dir, 8, 64u * 1024u * 1024u, 1, 0);

    MDB_txn *txn = NULL;
    MDB_dbi dbi = 0;
    CHECK(mdb_txn_begin(env, NULL, 0, &txn), "txn_begin");

    /* Attempt to create an incompatible DBI: DUPSORT + key-hash mode. */
    int rc = mdb_dbi_open(txn, "bad",
                          MDB_CREATE | MDB_DUPSORT | MDB_AGG_HASHSUM | MDB_AGG_HASHSOURCE_FROM_KEY,
                          &dbi);
    /* The patch plans for MDB_INCOMPATIBLE; accept EINVAL too if your build maps it. */
    if (rc != MDB_INCOMPATIBLE && rc != EINVAL) {
        fprintf(stderr, "dupsort_incompat: expected MDB_INCOMPATIBLE/EINVAL got %s (%d)\n",
                mdb_strerror(rc), rc);
        exit(EXIT_FAILURE);
    }
    mdb_txn_abort(txn);
    mdb_env_close(env);
}

/* ------------------------------ runner ------------------------------ */

typedef void (*test_fn)(void);

typedef struct {
    const char *name;
    test_fn fn;
} Test;

static void run_test(const Test *t)
{
    fprintf(stdout, "[RUN] %s\n", t->name);
    t->fn();
    fprintf(stdout, "[OK ] %s\n", t->name);
}

int main(void)
{
    const Test tests[] = {
        {"keyhash: plain right splits (off=17)", test_keyhash_rightsplit_offset17},
        {"keyhash: plain left splits (off=33)", test_keyhash_leftsplit_offset33},
        {"keyhash: delete/merge/root collapse (off=7)", test_keyhash_delete_root_collapse_offset7},
        {"keyhash: overwrite + MDB_CURRENT no delta (off=31)", test_keyhash_overwrite_no_hash_delta_offset31},
        {"keyhash: overflow values do not affect hashsum (off=64)", test_keyhash_big_values_offset64},
        {"keyhash: offsets matrix (many non-zero)", test_keyhash_offsets_matrix},
        {"keyhash: MDB_RESERVE allowed (off=15)", test_keyhash_reserve_allowed_offset15},
        {"keyhash: MDB_WRITEMAP allowed (off=23)", test_keyhash_writemap_allowed_offset23},
        {"keyhash: DUPSORT incompatible", test_keyhash_dupsort_incompatible},
    };

    for (size_t i = 0; i < sizeof(tests)/sizeof(tests[0]); i++)
        run_test(&tests[i]);

    fprintf(stdout, "All key-hash unit tests passed.\n");
    return 0;
}
