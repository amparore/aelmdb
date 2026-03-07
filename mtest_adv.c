/* mtest_agg_adv.c - aggressive tests for LMDB aggregate (COUNTED+) databases
 *
 * Build (example):
 *   cc -O2 -std=c99 -I. mtest_agg_adv.c mdb.c midl.c -o mtest_agg_adv
 *
 * Uses LMDB aggregate API:
 *   - MDB_AGG_ENTRIES / MDB_AGG_KEYS / MDB_AGG_HASHSUM
 *   - mdb_agg_info(), mdb_agg_totals(), mdb_agg_range()
 *   - mdb_agg_select(), mdb_agg_rank()
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

#define ENV_DIR "/tmp/testdb_agg_adv"

#define AGG_SCHEMA (MDB_AGG_ENTRIES | MDB_AGG_KEYS | MDB_AGG_HASHSUM)

#define CHECK(rc, msg) do {                                                     \
    if ((rc) != MDB_SUCCESS) {                                                  \
        fprintf(stderr, "%s: %s (%d)\n", (msg), mdb_strerror(rc), (rc));       \
        exit(EXIT_FAILURE);                                                     \
    }                                                                           \
} while (0)

#define REQUIRE(cond, msg) do {                                                 \
    if (!(cond)) {                                                              \
        fprintf(stderr, "%s\n", (msg));                                       \
        exit(EXIT_FAILURE);                                                     \
    }                                                                           \
} while (0)

static void fatal_errno(const char *msg)
{
    fprintf(stderr, "%s: %s\n", msg, strerror(errno));
    exit(EXIT_FAILURE);
}

static void expect_rc(int rc, int expect, const char *msg)
{
    if (rc != expect) {
        fprintf(stderr, "%s: expected %s (%d), got %s (%d)\n",
                msg, mdb_strerror(expect), expect, mdb_strerror(rc), rc);
        exit(EXIT_FAILURE);
    }
}

static void expect_u64_eq(uint64_t got, uint64_t want, const char *msg)
{
    if (got != want) {
        fprintf(stderr, "%s: expected %" PRIu64 ", got %" PRIu64 "\n", msg, want, got);
        exit(EXIT_FAILURE);
    }
}

/* SplitMix64: deterministic RNG for test data. */
static uint64_t splitmix64_next(uint64_t *x)
{
    uint64_t z = (*x += UINT64_C(0x9E3779B97F4A7C15));
    z = (z ^ (z >> 30)) * UINT64_C(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94D049BB133111EB);
    return z ^ (z >> 31);
}

/* ------------------------------ env helpers ------------------------------ */

static void unlink_env_files(const char *dir)
{
    char path[512];
    snprintf(path, sizeof(path), "%s/data.mdb", dir);
    unlink(path);
    snprintf(path, sizeof(path), "%s/lock.mdb", dir);
    unlink(path);
}

static MDB_env *open_env_dir(const char *dir, int maxdbs, size_t mapsize, int fresh)
{
    MDB_env *env = NULL;

    if (mkdir(dir, 0775) && errno != EEXIST)
        fatal_errno("mkdir envdir");
    if (chmod(dir, 0775) && errno != EPERM)
        fatal_errno("chmod envdir");

    if (fresh)
        unlink_env_files(dir);

    CHECK(mdb_env_create(&env), "mdb_env_create");
    CHECK(mdb_env_set_maxdbs(env, maxdbs), "mdb_env_set_maxdbs");
    CHECK(mdb_env_set_mapsize(env, mapsize), "mdb_env_set_mapsize");

    /* Fast test defaults: no lock + reduced sync. */
    CHECK(mdb_env_open(env, dir, MDB_NOLOCK | MDB_NOSYNC | MDB_NOMETASYNC, 0664), "mdb_env_open");
    return env;
}

static void assert_depth_at_least(MDB_txn *txn, MDB_dbi dbi, unsigned want_depth, const char *msg)
{
    MDB_stat st;
    CHECK(mdb_stat(txn, dbi, &st), "mdb_stat");
    if (st.ms_depth < want_depth) {
        fprintf(stderr, "%s: expected depth >= %u, got %u\n", msg, want_depth, st.ms_depth);
        exit(EXIT_FAILURE);
    }
}

/* ------------------------------ key/value generators ------------------------------ */

static void make_string_key(char *buf, size_t cap, unsigned flavor, unsigned id)
{
    /* Heavy shared prefixes to stress internal separator keys & page splits. */
    const char *pfx =
        (flavor % 4 == 0) ? "aaaaaaaaaaaaaaaa/common/prefix/" :
        (flavor % 4 == 1) ? "aaaaaaaaaaaaaaaa/common/prefix/zz/" :
        (flavor % 4 == 2) ? "aaaaaaaaaaaaaaaa/common/other/" :
                            "bbbbbbbbbbbbbbbb/other/prefix/";

    if ((flavor & 8u) != 0)
        snprintf(buf, cap, "%ssection/%02u/key_%09u", pfx, flavor & 31u, id);
    else
        snprintf(buf, cap, "%skey_%09u_%02u", pfx, id, flavor & 31u);
}

static size_t choose_value_size(uint64_t *rng)
{
    /* HASHSUM requires values to have at least MDB_HASH_SIZE bytes. */
    uint64_t r = splitmix64_next(rng);
    switch (r % 10u) {
        case 0: return (size_t)MDB_HASH_SIZE;
        case 1: return 64;
        case 2: return 96;
        case 3: return 128;
        case 4: return 512;
        case 5: return 2048;
        default: return (size_t)MDB_HASH_SIZE + 32;
    }
}

static void fill_value(unsigned char *buf, size_t sz, uint64_t seed)
{
    uint64_t x = seed;
    for (size_t i = 0; i < sz; ++i) {
        if ((i & 7u) == 0)
            x = splitmix64_next(&seed);
        buf[i] = (unsigned char)(x ^ (uint64_t)i);
    }
}

static void fill_fixed_elem(unsigned char *buf, size_t elsz, uint64_t tag)
{
    /* Deterministic fixed-size element content. */
    uint64_t x = tag;
    for (size_t i = 0; i < elsz; ++i) {
        x = x * UINT64_C(6364136223846793005) + UINT64_C(1442695040888963407);
        buf[i] = (unsigned char)(x >> 56);
    }
}

/* ------------------------------ naive reference implementations ------------------------------ */

static int val_eq(const MDB_val *a, const MDB_val *b)
{
    if (a->mv_size != b->mv_size) return 0;
    if (a->mv_size == 0) return 1;
    return memcmp(a->mv_data, b->mv_data, a->mv_size) == 0;
}

struct agg_oracle {
    uint64_t entries;
    uint64_t keys;
    uint8_t  hashsum[MDB_HASH_SIZE];
};

static void agg_oracle_zero(struct agg_oracle *o)
{
    o->entries = 0;
    o->keys = 0;
    memset(o->hashsum, 0, sizeof(o->hashsum));
}

static void agg_add_hash(struct agg_oracle *o, const MDB_val *data)
{
    /* Only valid when DB schema includes MDB_AGG_HASHSUM, and values satisfy size >= MDB_HASH_SIZE. */
    const uint8_t *p = (const uint8_t *)data->mv_data;
    mdb_hashsum_add(o->hashsum, p);
}

static int naive_agg_totals(MDB_txn *txn, MDB_dbi dbi, struct agg_oracle *out)
{
    MDB_cursor *cur = NULL;
    MDB_val k, v;
    int rc;

    agg_oracle_zero(out);

    rc = mdb_cursor_open(txn, dbi, &cur);
    if (rc != MDB_SUCCESS) return rc;

    MDB_val lastk = {0, NULL};
    rc = mdb_cursor_get(cur, &k, &v, MDB_FIRST);
    while (rc == MDB_SUCCESS) {
        out->entries++;
        if (lastk.mv_size == 0 || !val_eq(&k, &lastk)) {
            out->keys++;
            lastk = k;
        }
        agg_add_hash(out, &v);
        rc = mdb_cursor_get(cur, &k, &v, MDB_NEXT);
    }
    if (rc == MDB_NOTFOUND) rc = MDB_SUCCESS;

    mdb_cursor_close(cur);
    return rc;
}

static int naive_agg_range(MDB_txn *txn, MDB_dbi dbi,
                           const MDB_val *low, const MDB_val *high,
                           unsigned flags, struct agg_oracle *out)
{
    MDB_cursor *cur = NULL;
    MDB_val k, v;
    int rc;

    const int lower_incl = (flags & MDB_RANGE_LOWER_INCL) != 0;
    const int upper_incl = (flags & MDB_RANGE_UPPER_INCL) != 0;

    agg_oracle_zero(out);

    if (low && high) {
        int c = mdb_cmp(txn, dbi, (MDB_val *)low, (MDB_val *)high);
        if (c > 0 || (c == 0 && !(lower_incl && upper_incl)))
            return MDB_SUCCESS;
    }

    rc = mdb_cursor_open(txn, dbi, &cur);
    if (rc != MDB_SUCCESS) return rc;

    if (low) {
        k = *low;
        rc = mdb_cursor_get(cur, &k, &v, MDB_SET_RANGE);
        if (rc == MDB_NOTFOUND) { mdb_cursor_close(cur); return MDB_SUCCESS; }
        if (rc != MDB_SUCCESS)  { mdb_cursor_close(cur); return rc; }
        if (!lower_incl && mdb_cmp(txn, dbi, &k, (MDB_val *)low) == 0) {
            rc = mdb_cursor_get(cur, &k, &v, MDB_NEXT_NODUP);
            if (rc == MDB_NOTFOUND) { mdb_cursor_close(cur); return MDB_SUCCESS; }
            if (rc != MDB_SUCCESS)  { mdb_cursor_close(cur); return rc; }
        }
    } else {
        rc = mdb_cursor_get(cur, &k, &v, MDB_FIRST);
        if (rc == MDB_NOTFOUND) { mdb_cursor_close(cur); return MDB_SUCCESS; }
        if (rc != MDB_SUCCESS)  { mdb_cursor_close(cur); return rc; }
    }

    MDB_val lastk = {0, NULL};
    for (;;) {
        if (high) {
            int c = mdb_cmp(txn, dbi, &k, (MDB_val *)high);
            if (c > 0 || (c == 0 && !upper_incl))
                break;
        }
        out->entries++;
        if (lastk.mv_size == 0 || !val_eq(&k, &lastk)) {
            out->keys++;
            lastk = k;
        }
        agg_add_hash(out, &v);

        rc = mdb_cursor_get(cur, &k, &v, MDB_NEXT);
        if (rc == MDB_NOTFOUND) { rc = MDB_SUCCESS; break; }
        if (rc != MDB_SUCCESS)  { mdb_cursor_close(cur); return rc; }
    }

    mdb_cursor_close(cur);
    return rc;
}

static int naive_select_entry(MDB_txn *txn, MDB_dbi dbi, uint64_t rank, MDB_val *key, MDB_val *data)
{
    MDB_cursor *cur = NULL;
    MDB_val k, v;
    int rc;

    rc = mdb_cursor_open(txn, dbi, &cur);
    if (rc != MDB_SUCCESS) return rc;

    rc = mdb_cursor_get(cur, &k, &v, MDB_FIRST);
    if (rc != MDB_SUCCESS) {
        mdb_cursor_close(cur);
        return rc;
    }

    for (uint64_t i = 0; i < rank; ++i) {
        rc = mdb_cursor_get(cur, &k, &v, MDB_NEXT);
        if (rc != MDB_SUCCESS) {
            mdb_cursor_close(cur);
            return rc;
        }
    }

    *key = k;
    *data = v;
    mdb_cursor_close(cur);
    return MDB_SUCCESS;
}

static int naive_select_key(MDB_txn *txn, MDB_dbi dbi, uint64_t rank, MDB_val *key, MDB_val *data)
{
    MDB_cursor *cur = NULL;
    MDB_val k, v;
    int rc;

    rc = mdb_cursor_open(txn, dbi, &cur);
    if (rc != MDB_SUCCESS) return rc;
    rc = mdb_cursor_get(cur, &k, &v, MDB_FIRST);
    if (rc != MDB_SUCCESS) { mdb_cursor_close(cur); return rc; }

    for (uint64_t i = 0; i < rank; ++i) {
        rc = mdb_cursor_get(cur, &k, &v, MDB_NEXT_NODUP);
        if (rc != MDB_SUCCESS) { mdb_cursor_close(cur); return rc; }
    }

    *key = k;
    *data = v; /* first duplicate */
    mdb_cursor_close(cur);
    return MDB_SUCCESS;
}

static int naive_rank_exact_entry(MDB_txn *txn, MDB_dbi dbi,
                                 const MDB_val *key, const MDB_val *data,
                                 uint64_t *rank, uint64_t *dup_index)
{
    MDB_cursor *cur = NULL;
    MDB_val k, v;
    int rc;
    uint64_t i = 0;
    uint64_t di = 0;
    int in_key = 0;

    rc = mdb_cursor_open(txn, dbi, &cur);
    if (rc != MDB_SUCCESS) return rc;

    rc = mdb_cursor_get(cur, &k, &v, MDB_FIRST);
    while (rc == MDB_SUCCESS) {
        if (!in_key || !val_eq(&k, key)) {
            in_key = val_eq(&k, key);
            di = 0;
        } else {
            di++;
        }

        if (val_eq(&k, key) && val_eq(&v, data)) {
            *rank = i;
            if (dup_index) *dup_index = in_key ? di : 0;
            mdb_cursor_close(cur);
            return MDB_SUCCESS;
        }
        i++;
        rc = mdb_cursor_get(cur, &k, &v, MDB_NEXT);
    }

    mdb_cursor_close(cur);
    if (rc == MDB_NOTFOUND) return MDB_NOTFOUND;
    return rc;
}

static int naive_rank_set_range_entry(MDB_txn *txn, MDB_dbi dbi, int is_dupsort,
                                      MDB_val *key_inout, MDB_val *data_inout,
                                      uint64_t *rank, uint64_t *dup_index)
{
    /*
     * Emulate mdb_agg_rank(..., MDB_AGG_RANK_SET_RANGE) using cursor semantics.
     *
     * For MDB_DUPSORT, set-range is in *record order* over (key,data). In LMDB,
     * MDB_GET_BOTH_RANGE requires an *existing* key, so it is not equivalent to
     * "first record >= (key,data)" when the key doesn't exist or when no duplicate
     * >= data exists for that key. In those cases, the correct result is the first
     * record of the next greater key.
     */
    MDB_cursor *cur = NULL;
    MDB_val qk = *key_inout;
    MDB_val qd = *data_inout;
    MDB_val k = qk;
    MDB_val v = qd;
    int rc;

    rc = mdb_cursor_open(txn, dbi, &cur);
    if (rc != MDB_SUCCESS)
        return rc;

    if (!is_dupsort) {
        rc = mdb_cursor_get(cur, &k, &v, MDB_SET_RANGE);
        if (rc != MDB_SUCCESS) {
            mdb_cursor_close(cur);
            return rc;
        }
    } else {
        /* 1) Find the first key >= qk (data becomes first dup for that key). */
        MDB_val sk = qk;
        MDB_val sv = {0, NULL};
        rc = mdb_cursor_get(cur, &sk, &sv, MDB_SET_RANGE);
        if (rc != MDB_SUCCESS) {
            mdb_cursor_close(cur);
            return rc;
        }

        /* 2) If we landed on the same key and have a data lower bound, refine within duplicates. */
        if (val_eq(&sk, &qk) && qd.mv_size != 0) {
            MDB_val tk = sk;
            MDB_val tv = qd;
            int rc2 = mdb_cursor_get(cur, &tk, &tv, MDB_GET_BOTH_RANGE);
            if (rc2 == MDB_SUCCESS) {
                sk = tk;
                sv = tv;
            } else if (rc2 == MDB_NOTFOUND) {
                /*
                 * No duplicate >= qd for key==qk, so the next record in record-order
                 * is the first record of the next greater key (if any).
                 */
                MDB_val nk = sk;
                MDB_val nv = sv;

                /* Reposition to the key (GET_BOTH_RANGE may leave the cursor unpositioned). */
                rc2 = mdb_cursor_get(cur, &nk, &nv, MDB_SET);
                if (rc2 != MDB_SUCCESS) {
                    mdb_cursor_close(cur);
                    return rc2;
                }
                rc2 = mdb_cursor_get(cur, &nk, &nv, MDB_LAST_DUP);
                if (rc2 != MDB_SUCCESS) {
                    mdb_cursor_close(cur);
                    return rc2;
                }
                rc2 = mdb_cursor_get(cur, &nk, &nv, MDB_NEXT_NODUP);
                if (rc2 != MDB_SUCCESS) {
                    mdb_cursor_close(cur);
                    return rc2;
                }
                sk = nk;
                sv = nv;
            } else {
                mdb_cursor_close(cur);
                return rc2;
            }
        }

        k = sk;
        v = sv;
    }

    /* Found record is (k,v). Compute its rank by scanning from the start. */
    MDB_val fk, fv;
    uint64_t i = 0;
    uint64_t di = 0;
    int in_key = 0;

    int rc2 = mdb_cursor_get(cur, &fk, &fv, MDB_FIRST);
    while (rc2 == MDB_SUCCESS) {
        if (!in_key || !val_eq(&fk, &k)) {
            in_key = val_eq(&fk, &k);
            di = 0;
        } else {
            di++;
        }

        if (val_eq(&fk, &k) && val_eq(&fv, &v)) {
            *rank = i;
            if (dup_index)
                *dup_index = in_key ? di : 0;
            *key_inout = k;
            *data_inout = v;
            mdb_cursor_close(cur);
            return MDB_SUCCESS;
        }
        i++;
        rc2 = mdb_cursor_get(cur, &fk, &fv, MDB_NEXT);
    }

    mdb_cursor_close(cur);
    if (rc2 == MDB_NOTFOUND)
        return MDB_CORRUPTED; /* should not happen */
    return rc2;
}

static int naive_rank_exact_key(MDB_txn *txn, MDB_dbi dbi, const MDB_val *key, uint64_t *rank)
{
    MDB_cursor *cur = NULL;
    MDB_val k, v;
    int rc;
    uint64_t i = 0;

    rc = mdb_cursor_open(txn, dbi, &cur);
    if (rc != MDB_SUCCESS) return rc;
    rc = mdb_cursor_get(cur, &k, &v, MDB_FIRST);
    while (rc == MDB_SUCCESS) {
        if (val_eq(&k, key)) {
            *rank = i;
            mdb_cursor_close(cur);
            return MDB_SUCCESS;
        }
        i++;
        rc = mdb_cursor_get(cur, &k, &v, MDB_NEXT_NODUP);
    }
    mdb_cursor_close(cur);
    if (rc == MDB_NOTFOUND) return MDB_NOTFOUND;
    return rc;
}

static int naive_rank_set_range_key(MDB_txn *txn, MDB_dbi dbi, MDB_val *key_inout,
                                    uint64_t *rank)
{
    MDB_cursor *cur = NULL;
    MDB_val k = *key_inout;
    MDB_val v = {0, NULL};
    int rc;

    rc = mdb_cursor_open(txn, dbi, &cur);
    if (rc != MDB_SUCCESS) return rc;

    rc = mdb_cursor_get(cur, &k, &v, MDB_SET_RANGE);
    if (rc != MDB_SUCCESS) { mdb_cursor_close(cur); return rc; }

    /* Compute key-rank by scanning distinct keys from the start. */
    MDB_val fk, fv;
    uint64_t i = 0;
    int rc2 = mdb_cursor_get(cur, &fk, &fv, MDB_FIRST);
    while (rc2 == MDB_SUCCESS) {
        if (val_eq(&fk, &k)) {
            *rank = i;
            *key_inout = k;
            mdb_cursor_close(cur);
            return MDB_SUCCESS;
        }
        i++;
        rc2 = mdb_cursor_get(cur, &fk, &fv, MDB_NEXT_NODUP);
    }
    mdb_cursor_close(cur);
    if (rc2 == MDB_NOTFOUND) return MDB_CORRUPTED;
    return rc2;
}

static void expect_val_eq(const MDB_val *got, const MDB_val *want, const char *msg)
{
    if (got->mv_size != want->mv_size || (got->mv_size && memcmp(got->mv_data, want->mv_data, got->mv_size) != 0)) {
        fprintf(stderr, "%s: value mismatch (size got=%zu want=%zu)\n",
                msg, (size_t)got->mv_size, (size_t)want->mv_size);
        exit(EXIT_FAILURE);
    }
}

static void expect_hash_eq(const uint8_t *got, const uint8_t *want, const char *msg)
{
    if (memcmp(got, want, MDB_HASH_SIZE) != 0) {
        fprintf(stderr, "%s: hashsum mismatch\n", msg);
        for (unsigned i = 0; i < MDB_HASH_SIZE; ++i)
            fprintf(stderr, "%02x", (unsigned)got[i]);
        fprintf(stderr, "\nwant: ");
        for (unsigned i = 0; i < MDB_HASH_SIZE; ++i)
            fprintf(stderr, "%02x", (unsigned)want[i]);
        fprintf(stderr, "\n");
        exit(EXIT_FAILURE);
    }
}

static void agg_get_all(const MDB_agg *a, struct agg_oracle *out)
{
    // uint64_t e = 0, k = 0;
    // const void *hp = NULL;
    // size_t hl = 0;
    // CHECK(mdb_agg_get_entries(a, &e), "mdb_agg_get_entries");
    // CHECK(mdb_agg_get_keys(a, &k), "mdb_agg_get_keys");
    // CHECK(mdb_agg_get_hashsum(a, &hp, &hl), "mdb_agg_get_hashsum");
    // REQUIRE(hl == MDB_HASH_SIZE, "hashsum length");
    out->entries = a->mv_agg_entries;
    out->keys = a->mv_agg_keys;
    memcpy(out->hashsum, a->mv_agg_hashes, MDB_HASH_SIZE);
}

static void validate_agg_db(MDB_txn *txn, MDB_dbi dbi, int is_dupsort, uint64_t *rng, const char *tag)
{
    (void)is_dupsort;
    struct agg_oracle got, exp;
    MDB_agg a;
    unsigned schema = 0;

    CHECK(mdb_agg_info(txn, dbi, &schema), "mdb_agg_info");
    REQUIRE((schema & AGG_SCHEMA) == AGG_SCHEMA, "validate_agg_db: schema must be full");

    CHECK(mdb_agg_totals(txn, dbi, &a), "mdb_agg_totals");
    agg_get_all(&a, &got);
    CHECK(naive_agg_totals(txn, dbi, &exp), "naive_agg_totals");

    if (got.entries != exp.entries || got.keys != exp.keys || memcmp(got.hashsum, exp.hashsum, MDB_HASH_SIZE) != 0) {
        fprintf(stderr, "%s: totals mismatch got(entries=%" PRIu64 ",keys=%" PRIu64 ") exp(entries=%" PRIu64 ",keys=%" PRIu64 ")\n",
                tag, got.entries, got.keys, exp.entries, exp.keys);
        expect_hash_eq(got.hashsum, exp.hashsum, "totals hashsum");
        exit(EXIT_FAILURE);
    }

    /* Range aggregate tests. */
    for (unsigned i = 0; i < 32; ++i) {
        unsigned flags = 0;
        if (splitmix64_next(rng) & 1u) flags |= MDB_RANGE_LOWER_INCL;
        if (splitmix64_next(rng) & 2u) flags |= MDB_RANGE_UPPER_INCL;

        MDB_val low = {0}, high = {0};
        MDB_val lk = {0}, ld = {0}, hk = {0}, hd = {0};
        int have_low = 0, have_high = 0;

        if (got.entries > 0 && (splitmix64_next(rng) % 4u) != 0u) {
            uint64_t r = splitmix64_next(rng) % got.entries;
            CHECK(mdb_agg_select(txn, dbi, MDB_AGG_WEIGHT_ENTRIES, r, &lk, &ld, NULL), "select for low");
            low = lk;
            have_low = 1;
        }
        if (got.entries > 0 && (splitmix64_next(rng) % 4u) != 0u) {
            uint64_t r = splitmix64_next(rng) % got.entries;
            CHECK(mdb_agg_select(txn, dbi, MDB_AGG_WEIGHT_ENTRIES, r, &hk, &hd, NULL), "select for high");
            high = hk;
            have_high = 1;
        }

        /* Occasionally force an empty range by swapping bounds. */
        if (have_low && have_high && ((splitmix64_next(rng) & 7u) == 0u)) {
            MDB_val tmp = low;
            low = high;
            high = tmp;
        }

        struct agg_oracle g2, e2;
        CHECK(mdb_agg_range(txn, dbi,
                            have_low ? &low : NULL,
                            NULL, have_high ? &high : NULL,
                            NULL, flags, &a),
              "mdb_agg_range");
        agg_get_all(&a, &g2);
        CHECK(naive_agg_range(txn, dbi,
                              have_low ? &low : NULL,
                              have_high ? &high : NULL,
                              flags, &e2),
              "naive_agg_range");
        if (g2.entries != e2.entries || g2.keys != e2.keys || memcmp(g2.hashsum, e2.hashsum, MDB_HASH_SIZE) != 0) {
            fprintf(stderr, "%s: range mismatch flags=%u got(e=%" PRIu64 ",k=%" PRIu64 ") exp(e=%" PRIu64 ",k=%" PRIu64 ")\n",
                    tag, flags, g2.entries, g2.keys, e2.entries, e2.keys);
            expect_hash_eq(g2.hashsum, e2.hashsum, "range hashsum");
            exit(EXIT_FAILURE);
        }
    }

    /* Select + rank tests. */
    if (got.entries > 0) {
        for (unsigned i = 0; i < 16; ++i) {
            uint64_t r = splitmix64_next(rng) % got.entries;
            MDB_val gk, gd, ek, ed;
            uint64_t gdi = 0, edi = 0;
            CHECK(mdb_agg_select(txn, dbi, MDB_AGG_WEIGHT_ENTRIES, r, &gk, &gd, &gdi), "mdb_agg_select entries");
            CHECK(naive_select_entry(txn, dbi, r, &ek, &ed), "naive_select_entry");
            expect_val_eq(&gk, &ek, "select(entries) key");
            expect_val_eq(&gd, &ed, "select(entries) data");

            /* rank(exact) for an existing record must return the same rank. */
            {
                MDB_val rk = gk;
				/*
				 * API contract (see lmdb.h):
				 * - Non-DUPSORT: data must be empty for rank(entries).
				 * - DUPSORT: data must identify an existing duplicate for EXACT.
				 */
				MDB_val rd = is_dupsort ? gd : (MDB_val){0, NULL};
                uint64_t rr = UINT64_MAX, rdi = 0;
                CHECK(mdb_agg_rank(txn, dbi, &rk, &rd, MDB_AGG_WEIGHT_ENTRIES, MDB_AGG_RANK_EXACT, &rr, &rdi),
                      "mdb_agg_rank exact entries");
                expect_u64_eq(rr, r, "rank(exact entries)==r");
				/* For non-DUPSORT, rank(entries exact) should also materialize the record data. */
				if (!is_dupsort) {
					expect_val_eq(&rk, &gk, "rank(exact entries) key");
					expect_val_eq(&rd, &gd, "rank(exact entries) data materialized");
					REQUIRE(rdi == 0, "rank(exact entries) dup_index==0 for non-dupsort");
					REQUIRE(gdi == 0, "select(entries) dup_index==0 for non-dupsort");
				}
                /* Cross-check with naive scan. */
                {
                    uint64_t nr = UINT64_MAX, ndi = 0;
                    CHECK(naive_rank_exact_entry(txn, dbi, &gk, &gd, &nr, &ndi), "naive_rank_exact_entry");
                    expect_u64_eq(nr, r, "naive_rank_exact_entry==r");
                    edi = ndi;
                }
                if (is_dupsort) {
                    REQUIRE(rdi == edi, "dup_index exact entries matches naive");
                    REQUIRE(gdi == edi, "dup_index select entries matches naive");
                }
            }
        }

        /* select out-of-range */
        {
            MDB_val k, d;
            int rc = mdb_agg_select(txn, dbi, MDB_AGG_WEIGHT_ENTRIES, got.entries, &k, &d, NULL);
            expect_rc(rc, MDB_NOTFOUND, "select(entries) rank==N => NOTFOUND");
        }

        /* rank(set-range) should match cursor set-range semantics. */
        for (unsigned i = 0; i < 16; ++i) {
            uint64_t base_r = splitmix64_next(rng) % got.entries;
            MDB_val bk, bd;
            CHECK(mdb_agg_select(txn, dbi, MDB_AGG_WEIGHT_ENTRIES, base_r, &bk, &bd, NULL), "select base for set-range");

            unsigned char kbuf[512];
            unsigned char dbuf[2048];
            MDB_val pk, pd;

            if (bk.mv_size > sizeof(kbuf)) {
                pk = bk;
            } else {
                memcpy(kbuf, bk.mv_data, bk.mv_size);
                if (bk.mv_size > 0) {
                    size_t last = bk.mv_size - 1;
                    kbuf[last] = (unsigned char)(kbuf[last] + 1u);
                }
                pk.mv_data = kbuf;
                pk.mv_size = bk.mv_size;
            }

            if (is_dupsort) {
                if (bd.mv_size > sizeof(dbuf)) {
                    pd = bd;
                } else {
                    memcpy(dbuf, bd.mv_data, bd.mv_size);
                    if (bd.mv_size > 0) {
                        size_t last = bd.mv_size - 1;
                        dbuf[last] = (unsigned char)(dbuf[last] + 1u);
                    }
                    pd.mv_data = dbuf;
                    pd.mv_size = bd.mv_size;
                }
            } else {
                pd.mv_data = NULL;
                pd.mv_size = 0;
            }

            MDB_val gk = pk, gd = pd;
            uint64_t gr = UINT64_MAX, gdi = 0;
            int grc = mdb_agg_rank(txn, dbi, &gk, &gd, MDB_AGG_WEIGHT_ENTRIES, MDB_AGG_RANK_SET_RANGE, &gr, &gdi);

            MDB_val ek = pk, ed = pd;
            uint64_t er = UINT64_MAX, edi = 0;
            int erc = naive_rank_set_range_entry(txn, dbi, is_dupsort, &ek, &ed, &er, &edi);

            if (grc != erc) {
                fprintf(stderr, "%s: set-range rc mismatch got=%s(%d) exp=%s(%d)\n",
                        tag, mdb_strerror(grc), grc, mdb_strerror(erc), erc);
                exit(EXIT_FAILURE);
            }
            if (grc == MDB_SUCCESS) {
                expect_u64_eq(gr, er, "set-range rank entries");
                expect_val_eq(&gk, &ek, "set-range key entries");
                expect_val_eq(&gd, &ed, "set-range data entries");
                if (is_dupsort) REQUIRE(gdi == edi, "set-range dup_index entries");

                MDB_val sk, sd;
                uint64_t sdi = 0;
                CHECK(mdb_agg_select(txn, dbi, MDB_AGG_WEIGHT_ENTRIES, gr, &sk, &sd, &sdi), "select after set-range");
                expect_val_eq(&sk, &gk, "select matches set-range key");
                expect_val_eq(&sd, &gd, "select matches set-range data");
                if (is_dupsort) REQUIRE(sdi == gdi, "select dup_index matches");
            }
        }
    }

    /* Key-weight select/rank tests (important for DUPSORT). */
    if (got.keys > 0) {
        for (unsigned i = 0; i < 12; ++i) {
            uint64_t r = splitmix64_next(rng) % got.keys;
            MDB_val gk, gd, ek, ed;
            uint64_t gdi = 7;
            CHECK(mdb_agg_select(txn, dbi, MDB_AGG_WEIGHT_KEYS, r, &gk, &gd, &gdi), "mdb_agg_select keys");
            CHECK(naive_select_key(txn, dbi, r, &ek, &ed), "naive_select_key");
            expect_val_eq(&gk, &ek, "select(keys) key");
            expect_val_eq(&gd, &ed, "select(keys) data(first) ");
            REQUIRE(gdi == 0, "select(keys) dup_index==0");

            MDB_val rk = gk;
            MDB_val rd = {0, NULL};
            uint64_t rr = UINT64_MAX, rdi = 9;
            CHECK(mdb_agg_rank(txn, dbi, &rk, &rd, MDB_AGG_WEIGHT_KEYS, MDB_AGG_RANK_EXACT, &rr, &rdi), "mdb_agg_rank exact keys");
            expect_u64_eq(rr, r, "rank(exact keys)==r");
            REQUIRE(rdi == 0, "rank(keys) dup_index==0");

            {
                uint64_t nr = UINT64_MAX;
                CHECK(naive_rank_exact_key(txn, dbi, &gk, &nr), "naive_rank_exact_key");
                expect_u64_eq(nr, r, "naive_rank_exact_key==r");
            }
        }

        /* select out-of-range */
        {
            MDB_val k, d;
            int rc = mdb_agg_select(txn, dbi, MDB_AGG_WEIGHT_KEYS, got.keys, &k, &d, NULL);
            expect_rc(rc, MDB_NOTFOUND, "select(keys) rank==K => NOTFOUND");
        }

        /* key-weight set-range */
        for (unsigned i = 0; i < 10; ++i) {
            uint64_t base_r = splitmix64_next(rng) % got.keys;
            MDB_val bk, bd;
            CHECK(mdb_agg_select(txn, dbi, MDB_AGG_WEIGHT_KEYS, base_r, &bk, &bd, NULL), "select base key for set-range(keys)");

            unsigned char kbuf[512];
            MDB_val pk;
            if (bk.mv_size > sizeof(kbuf)) {
                pk = bk;
            } else {
                memcpy(kbuf, bk.mv_data, bk.mv_size);
                if (bk.mv_size > 0) {
                    size_t last = bk.mv_size - 1;
                    kbuf[last] = (unsigned char)(kbuf[last] + 1u);
                }
                pk.mv_data = kbuf;
                pk.mv_size = bk.mv_size;
            }

            MDB_val gk = pk;
            MDB_val gd = {0, NULL};
            uint64_t gr = UINT64_MAX, gdi = 123;
            int grc = mdb_agg_rank(txn, dbi, &gk, &gd, MDB_AGG_WEIGHT_KEYS, MDB_AGG_RANK_SET_RANGE, &gr, &gdi);

            MDB_val ek = pk;
            uint64_t er = UINT64_MAX;
            int erc = naive_rank_set_range_key(txn, dbi, &ek, &er);

            if (grc != erc) {
                fprintf(stderr, "%s: set-range(keys) rc mismatch got=%s(%d) exp=%s(%d)\n",
                        tag, mdb_strerror(grc), grc, mdb_strerror(erc), erc);
                exit(EXIT_FAILURE);
            }
            if (grc == MDB_SUCCESS) {
                expect_u64_eq(gr, er, "set-range rank keys");
                expect_val_eq(&gk, &ek, "set-range key keys");
                REQUIRE(gdi == 0, "set-range(keys) dup_index==0");
            }
        }
    }
}

/* ------------------------------ snapshot helper ------------------------------ */

struct kv_copy {
    uint64_t rank;
    void *k;
    size_t ks;
    void *v;
    size_t vs;
};

static void kv_copy_free(struct kv_copy *c)
{
    if (!c) return;
    free(c->k);
    free(c->v);
    c->k = NULL;
    c->v = NULL;
    c->ks = c->vs = 0;
}

static struct kv_copy kv_copy_make(uint64_t rank, const MDB_val *k, const MDB_val *v)
{
    struct kv_copy c;
    c.rank = rank;
    c.ks = k->mv_size;
    c.vs = v->mv_size;
    c.k = malloc(c.ks ? c.ks : 1);
    c.v = malloc(c.vs ? c.vs : 1);
    if (!c.k || !c.v) fatal_errno("malloc kv_copy");
    if (c.ks) memcpy(c.k, k->mv_data, c.ks);
    if (c.vs) memcpy(c.v, v->mv_data, c.vs);
    return c;
}

/* ------------------------------ tests ------------------------------ */

static void test_empty_and_errors(void)
{
    fprintf(stderr, "\ntest_empty_and_errors\n");

    MDB_env *env = open_env_dir(ENV_DIR, 8, (size_t)256u * 1024u * 1024u, 1);
    MDB_txn *txn = NULL;
    MDB_dbi dbi_agg = 0, dbi_plain = 0;

    CHECK(mdb_txn_begin(env, NULL, 0, &txn), "empty: txn_begin");
    CHECK(mdb_dbi_open(txn, "agg", MDB_CREATE | AGG_SCHEMA, &dbi_agg), "empty: open agg");
    CHECK(mdb_dbi_open(txn, "plain", MDB_CREATE, &dbi_plain), "empty: open plain");
    CHECK(mdb_txn_commit(txn), "empty: commit");

    CHECK(mdb_txn_begin(env, NULL, MDB_RDONLY, &txn), "empty: ro begin");
    CHECK(mdb_dbi_open(txn, "agg", 0, &dbi_agg), "empty: ro open agg");
    CHECK(mdb_dbi_open(txn, "plain", 0, &dbi_plain), "empty: ro open plain");

    MDB_agg a;
    struct agg_oracle g;
    CHECK(mdb_agg_totals(txn, dbi_agg, &a), "empty: agg_totals");
    agg_get_all(&a, &g);
    expect_u64_eq(g.entries, 0, "empty entries==0");
    expect_u64_eq(g.keys, 0, "empty keys==0");
    {
        uint8_t z[MDB_HASH_SIZE];
        memset(z, 0, sizeof(z));
        expect_hash_eq(g.hashsum, z, "empty hashsum==0");
    }

    CHECK(mdb_agg_range(txn, dbi_agg, NULL, NULL, NULL, NULL, 0, &a), "empty: agg_range all");
    agg_get_all(&a, &g);
    expect_u64_eq(g.entries, 0, "empty range entries==0");
    expect_u64_eq(g.keys, 0, "empty range keys==0");

    {
        MDB_val k, d;
        uint64_t di = 0;
        int rc = mdb_agg_select(txn, dbi_agg, MDB_AGG_WEIGHT_ENTRIES, 0, &k, &d, &di);
        expect_rc(rc, MDB_NOTFOUND, "empty select(entries,0)");
        rc = mdb_agg_select(txn, dbi_agg, MDB_AGG_WEIGHT_KEYS, 0, &k, &d, &di);
        expect_rc(rc, MDB_NOTFOUND, "empty select(keys,0)");
    }

    {
        char kbuf[] = "k";
        MDB_val k = { sizeof(kbuf)-1, kbuf };
        MDB_val d = { 0, NULL };
        uint64_t r = 0;
        int rc = mdb_agg_rank(txn, dbi_agg, &k, &d, MDB_AGG_WEIGHT_ENTRIES, MDB_AGG_RANK_EXACT, &r, NULL);
        expect_rc(rc, MDB_NOTFOUND, "empty rank(exact entries)");
        rc = mdb_agg_rank(txn, dbi_agg, &k, &d, MDB_AGG_WEIGHT_KEYS, MDB_AGG_RANK_EXACT, &r, NULL);
        expect_rc(rc, MDB_NOTFOUND, "empty rank(exact keys)");
    }

    {
        /* Non-agg DBs must reject agg APIs. */
        expect_rc(mdb_agg_totals(txn, dbi_plain, &a), MDB_INCOMPATIBLE, "plain agg_totals incompatible");
        expect_rc(mdb_agg_range(txn, dbi_plain, NULL, NULL, NULL, NULL, 0, &a), MDB_INCOMPATIBLE, "plain agg_range incompatible");
        {
            MDB_val k, d;
            expect_rc(mdb_agg_select(txn, dbi_plain, MDB_AGG_WEIGHT_ENTRIES, 0, &k, &d, NULL), MDB_INCOMPATIBLE,
                      "plain select incompatible");
        }
        {
            char kbuf[] = "k";
            MDB_val k = { sizeof(kbuf)-1, kbuf };
            MDB_val d = { 0, NULL };
            uint64_t r = 0;
            expect_rc(mdb_agg_rank(txn, dbi_plain, &k, &d, MDB_AGG_WEIGHT_ENTRIES, MDB_AGG_RANK_EXACT, &r, NULL),
                      MDB_INCOMPATIBLE, "plain rank incompatible");
        }
    }

    {
        /* Invalid flags/args sanity. */
        expect_rc(mdb_agg_range(txn, dbi_agg, NULL, NULL, NULL, NULL, 0x1u, &a), EINVAL, "agg_range invalid flags");
        {
            char kbuf[] = "k";
            char dbuf[MDB_HASH_SIZE];
            memset(dbuf, 0, sizeof(dbuf));
            MDB_val k = { sizeof(kbuf)-1, kbuf };
            MDB_val d = { sizeof(dbuf), dbuf };
            uint64_t r = 0;
            expect_rc(mdb_agg_rank(txn, dbi_agg, &k, &d, MDB_AGG_WEIGHT_ENTRIES, 2u, &r, NULL), EINVAL, "rank invalid flags");
        }
        expect_rc(mdb_agg_info(txn, dbi_agg, NULL), EINVAL, "agg_info NULL out");
    }

    mdb_txn_abort(txn);
    mdb_env_close(env);
}

static void test_upgrade_policy(void)
{
    fprintf(stderr, "\ntest_upgrade_policy\n");

    MDB_env *env = open_env_dir(ENV_DIR, 8, (size_t)512u * 1024u * 1024u, 1);
    MDB_txn *txn = NULL;

    /* 1) named DB upgrade must be rejected */
    {
        MDB_dbi dbi;
        MDB_val k, v;
        char kbuf[256];
        unsigned char vbuf[MDB_HASH_SIZE];
        memset(vbuf, 'v', sizeof(vbuf));

        CHECK(mdb_txn_begin(env, NULL, 0, &txn), "upgrade named: begin");
        CHECK(mdb_dbi_open(txn, "upgrade_named", MDB_CREATE, &dbi), "upgrade named: create");

        v.mv_data = vbuf;
        v.mv_size = sizeof(vbuf);
        for (unsigned i = 0; i < 4000; ++i) {
            make_string_key(kbuf, sizeof(kbuf), i & 31u, i);
            k.mv_data = kbuf;
            k.mv_size = strlen(kbuf);
            CHECK(mdb_put(txn, dbi, &k, &v, 0), "upgrade named: put");
        }
        CHECK(mdb_txn_commit(txn), "upgrade named: commit");

        CHECK(mdb_txn_begin(env, NULL, 0, &txn), "upgrade named: reopen begin");
        int rc = mdb_dbi_open(txn, "upgrade_named", AGG_SCHEMA, &dbi);
        expect_rc(rc, MDB_INCOMPATIBLE, "upgrade named => incompatible");
        mdb_txn_abort(txn);
    }

    /* 2) main DB upgrade with depth>1 must be rejected */
    {
        MDB_dbi main_dbi;
        MDB_val k, v;
        char kbuf[256];
        unsigned char vbuf[MDB_HASH_SIZE];
        memset(vbuf, 'v', sizeof(vbuf));
        MDB_stat st;

        CHECK(mdb_txn_begin(env, NULL, 0, &txn), "upgrade main: begin");
        CHECK(mdb_dbi_open(txn, NULL, 0, &main_dbi), "upgrade main: open main");
        v.mv_data = vbuf;
        v.mv_size = sizeof(vbuf);
        for (unsigned i = 0; i < 12000; ++i) {
            snprintf(kbuf, sizeof(kbuf), "main/%08u", i);
            k.mv_data = kbuf;
            k.mv_size = strlen(kbuf);
            CHECK(mdb_put(txn, main_dbi, &k, &v, 0), "upgrade main: put");
        }
        CHECK(mdb_txn_commit(txn), "upgrade main: commit");

        CHECK(mdb_txn_begin(env, NULL, MDB_RDONLY, &txn), "upgrade main: ro");
        CHECK(mdb_dbi_open(txn, NULL, 0, &main_dbi), "upgrade main: ro open main");
        CHECK(mdb_stat(txn, main_dbi, &st), "upgrade main: stat");
        REQUIRE(st.ms_depth >= 2, "upgrade main: expected depth>=2");
        mdb_txn_abort(txn);

        CHECK(mdb_txn_begin(env, NULL, 0, &txn), "upgrade main: reopen begin");
        int rc = mdb_dbi_open(txn, NULL, AGG_SCHEMA, &main_dbi);
        expect_rc(rc, MDB_INCOMPATIBLE, "upgrade main depth>1 => incompatible");
        mdb_txn_abort(txn);
    }

    mdb_env_close(env);

    /* 3) main DB upgrade on empty (depth==1) is allowed */
    env = open_env_dir(ENV_DIR, 4, (size_t)128u * 1024u * 1024u, 1);
    {
        MDB_dbi main_dbi;
        unsigned f = 0;

        CHECK(mdb_txn_begin(env, NULL, 0, &txn), "upgrade main empty: begin");
        CHECK(mdb_dbi_open(txn, NULL, AGG_SCHEMA, &main_dbi), "upgrade main empty: enable agg");
        CHECK(mdb_txn_commit(txn), "upgrade main empty: commit");

        CHECK(mdb_txn_begin(env, NULL, MDB_RDONLY, &txn), "upgrade main empty: ro");
        CHECK(mdb_dbi_open(txn, NULL, 0, &main_dbi), "upgrade main empty: ro open main");
        CHECK(mdb_dbi_flags(txn, main_dbi, &f), "upgrade main empty: flags");
        REQUIRE((f & AGG_SCHEMA) == AGG_SCHEMA, "upgrade main empty: agg schema bits set");
        mdb_txn_abort(txn);
    }
    mdb_env_close(env);
}

static void test_plain_agg_stress(void)
{
    fprintf(stderr, "\ntest_plain_agg_stress\n");

    MDB_env *env = open_env_dir(ENV_DIR, 8, (size_t)1024u * 1024u * 1024u, 1);
    MDB_txn *txn = NULL;
    MDB_dbi dbi = 0;
    uint64_t rng = UINT64_C(0xC0FFEE1234567890);

    /* Bulk load enough records to create internal nodes. */
    {
        enum { N = 9000 };
        MDB_val k, v;
        char kbuf[512];

        CHECK(mdb_txn_begin(env, NULL, 0, &txn), "plain bulk: begin");
        CHECK(mdb_dbi_open(txn, "plain", MDB_CREATE | AGG_SCHEMA, &dbi), "plain bulk: open");

        for (unsigned i = 0; i < N; ++i) {
            make_string_key(kbuf, sizeof(kbuf), i & 31u, i);
            k.mv_data = kbuf;
            k.mv_size = strlen(kbuf);

            size_t vsz = choose_value_size(&rng);
            unsigned char *vbuf = (unsigned char *)malloc(vsz ? vsz : 1);
            if (!vbuf) fatal_errno("malloc plain bulk value");
            fill_value(vbuf, vsz, (uint64_t)i ^ UINT64_C(0xA5A5A5A5));
            v.mv_data = vbuf;
            v.mv_size = vsz;
            CHECK(mdb_put(txn, dbi, &k, &v, 0), "plain bulk: put");
            free(vbuf);
        }

        CHECK(mdb_txn_commit(txn), "plain bulk: commit");
    }

    /* Validate baseline. */
    {
        CHECK(mdb_txn_begin(env, NULL, MDB_RDONLY, &txn), "plain base: ro");
        CHECK(mdb_dbi_open(txn, "plain", 0, &dbi), "plain base: open");
        assert_depth_at_least(txn, dbi, 2, "plain depth >=2");
        validate_agg_db(txn, dbi, 0, &rng, "plain baseline");
        mdb_txn_abort(txn);
    }

    for (unsigned round = 0; round < 6; ++round) {
        uint64_t before_n = 0;
        struct kv_copy snaps[6];
        unsigned snap_n = 0;
        for (unsigned i = 0; i < 6; ++i) snaps[i].k = snaps[i].v = NULL;

        /* Take a small snapshot for rollback checks (entry-weight). */
        CHECK(mdb_txn_begin(env, NULL, MDB_RDONLY, &txn), "plain snap: ro");
        CHECK(mdb_dbi_open(txn, "plain", 0, &dbi), "plain snap: open");
        MDB_agg a_before;
        struct agg_oracle tot_before;
        CHECK(mdb_agg_totals(txn, dbi, &a_before), "plain snap: totals");
        agg_get_all(&a_before, &tot_before);
        before_n = tot_before.entries;

        if (before_n > 0) {
            uint64_t ranks[6];
            ranks[0] = 0;
            ranks[1] = before_n / 2;
            ranks[2] = before_n - 1;
            ranks[3] = splitmix64_next(&rng) % before_n;
            ranks[4] = splitmix64_next(&rng) % before_n;
            ranks[5] = splitmix64_next(&rng) % before_n;

            for (unsigned i = 0; i < 6; ++i) {
                MDB_val k, d;
                if (mdb_agg_select(txn, dbi, MDB_AGG_WEIGHT_ENTRIES, ranks[i], &k, &d, NULL) == MDB_SUCCESS) {
                    snaps[snap_n++] = kv_copy_make(ranks[i], &k, &d);
                }
            }
        }
        mdb_txn_abort(txn);

        /* Mutate. */
        CHECK(mdb_txn_begin(env, NULL, 0, &txn), "plain round: begin");
        CHECK(mdb_dbi_open(txn, "plain", 0, &dbi), "plain round: open");

        enum { OPS = 4200 };
        for (unsigned op = 0; op < OPS; ++op) {
            uint64_t r = splitmix64_next(&rng);
            MDB_agg a;
            struct agg_oracle tot;
            CHECK(mdb_agg_totals(txn, dbi, &a), "plain round: totals");
            agg_get_all(&a, &tot);

            if ((r % 100u) < 45u && tot.entries > 0) {
                /* delete a random record (plain: delete key) */
                uint64_t del_r = splitmix64_next(&rng) % tot.entries;
                MDB_val k, d;
                CHECK(mdb_agg_select(txn, dbi, MDB_AGG_WEIGHT_ENTRIES, del_r, &k, &d, NULL), "plain del: select");
                int drc = mdb_del(txn, dbi, &k, NULL);
                if (drc != MDB_SUCCESS && drc != MDB_NOTFOUND)
                    CHECK(drc, "plain del: mdb_del");

            } else {
                /* insert (or update) a pseudo-random key */
                MDB_val k, v;
                char kbuf[512];
                unsigned id = (unsigned)(splitmix64_next(&rng) % 40000u);
                make_string_key(kbuf, sizeof(kbuf), (unsigned)(r & 31u), id);
                k.mv_data = kbuf;
                k.mv_size = strlen(kbuf);

                size_t vsz = choose_value_size(&rng);
                unsigned char *vbuf = (unsigned char *)malloc(vsz ? vsz : 1);
                if (!vbuf) fatal_errno("malloc plain ins value");
                fill_value(vbuf, vsz, r ^ ((uint64_t)id << 1));
                v.mv_data = vbuf;
                v.mv_size = vsz;

                int prc = mdb_put(txn, dbi, &k, &v, MDB_NOOVERWRITE);
                if (prc == MDB_KEYEXIST) {
                    /* update existing */
                    CHECK(mdb_put(txn, dbi, &k, &v, 0), "plain update: put");
                } else if (prc != MDB_SUCCESS) {
                    CHECK(prc, "plain put");
                }
                free(vbuf);
            }
        }

        int do_abort = (splitmix64_next(&rng) & 1u) != 0u;
        if (do_abort) {
            mdb_txn_abort(txn);
        } else {
            CHECK(mdb_txn_commit(txn), "plain round: commit");
        }

        /* Post-round validation (RO). */
        CHECK(mdb_txn_begin(env, NULL, MDB_RDONLY, &txn), "plain post: ro");
        CHECK(mdb_dbi_open(txn, "plain", 0, &dbi), "plain post: open");

        MDB_agg a_after;
        struct agg_oracle tot_after;
        CHECK(mdb_agg_totals(txn, dbi, &a_after), "plain post: totals");
        agg_get_all(&a_after, &tot_after);
        uint64_t after_n = tot_after.entries;

        if (do_abort) {
            expect_u64_eq(after_n, before_n, "plain abort preserves entry count");
            for (unsigned i = 0; i < snap_n; ++i) {
                MDB_val k, d;
                CHECK(mdb_agg_select(txn, dbi, MDB_AGG_WEIGHT_ENTRIES, snaps[i].rank, &k, &d, NULL), "plain abort: reselect");
                MDB_val wk = { snaps[i].ks, snaps[i].k };
                MDB_val wd = { snaps[i].vs, snaps[i].v };
                expect_val_eq(&k, &wk, "plain abort preserves key");
                expect_val_eq(&d, &wd, "plain abort preserves data");
            }
        }

        if ((round % 2u) == 0u) {
            validate_agg_db(txn, dbi, 0, &rng, do_abort ? "plain validate after abort" : "plain validate after commit");
        }

        mdb_txn_abort(txn);
        for (unsigned i = 0; i < snap_n; ++i) kv_copy_free(&snaps[i]);
    }

    mdb_env_close(env);
}

static void test_dupsort_variable(void)
{
    fprintf(stderr, "\ntest_dupsort_variable\n");

    MDB_env *env = open_env_dir(ENV_DIR, 8, (size_t)1024u * 1024u * 1024u, 1);
    MDB_txn *txn = NULL;
    MDB_dbi dbi = 0;
    uint64_t rng = UINT64_C(0xD00DFEEDCAFEBABE);

    /* Bulk load: many keys, each with a random number of duplicates. */
    {
        enum { KEYS = 900, MAXD = 18 };
        MDB_val key, data;
        char kbuf[128];

        CHECK(mdb_txn_begin(env, NULL, 0, &txn), "dups bulk: begin");
        CHECK(mdb_dbi_open(txn, "dups", MDB_CREATE | MDB_DUPSORT | AGG_SCHEMA, &dbi), "dups bulk: open");

        for (unsigned i = 0; i < KEYS; ++i) {
            snprintf(kbuf, sizeof(kbuf), "dup/%08u", i);
            key.mv_data = kbuf;
            key.mv_size = strlen(kbuf);

            unsigned dcount = 1u + (unsigned)(splitmix64_next(&rng) % MAXD);
            for (unsigned d = 0; d < dcount; ++d) {
                size_t vsz = (size_t)MDB_HASH_SIZE + (size_t)(splitmix64_next(&rng) % 200u);
                unsigned char *vbuf = (unsigned char *)malloc(vsz ? vsz : 1);
                if (!vbuf) fatal_errno("malloc dups bulk value");
                fill_value(vbuf, vsz, ((uint64_t)i << 32) ^ (uint64_t)d ^ UINT64_C(0xABC00000));
                data.mv_data = vbuf;
                data.mv_size = vsz;
                CHECK(mdb_put(txn, dbi, &key, &data, 0), "dups bulk: put");
                free(vbuf);
            }
        }
        CHECK(mdb_txn_commit(txn), "dups bulk: commit");
    }

    /* Validate baseline. */
    {
        CHECK(mdb_txn_begin(env, NULL, MDB_RDONLY, &txn), "dups base: ro");
        CHECK(mdb_dbi_open(txn, "dups", 0, &dbi), "dups base: open");
        assert_depth_at_least(txn, dbi, 2, "dups depth >=2");
        validate_agg_db(txn, dbi, 1, &rng, "dups baseline");
        mdb_txn_abort(txn);
    }

    /* Mutation rounds. */
    for (unsigned round = 0; round < 6; ++round) {
        CHECK(mdb_txn_begin(env, NULL, 0, &txn), "dups round: begin");
        CHECK(mdb_dbi_open(txn, "dups", 0, &dbi), "dups round: open");

        enum { OPS = 3200 };
        for (unsigned op = 0; op < OPS; ++op) {
            uint64_t r = splitmix64_next(&rng);
            MDB_agg a;
            struct agg_oracle tot;
            CHECK(mdb_agg_totals(txn, dbi, &a), "dups round: totals");
            agg_get_all(&a, &tot);

            if ((r % 100u) < 45u) {
                /* add a duplicate (often with NODUPDATA to avoid identical values) */
                MDB_val key, data;
                char kbuf[128];
                unsigned kid = (unsigned)(splitmix64_next(&rng) % 1600u);
                snprintf(kbuf, sizeof(kbuf), "dup/%08u", kid);
                key.mv_data = kbuf;
                key.mv_size = strlen(kbuf);

                size_t vsz = (size_t)MDB_HASH_SIZE + (size_t)(splitmix64_next(&rng) % 240u);
                unsigned char *vbuf = (unsigned char *)malloc(vsz ? vsz : 1);
                if (!vbuf) fatal_errno("malloc dups add value");
                fill_value(vbuf, vsz, r ^ ((uint64_t)kid << 1) ^ (uint64_t)round);
                data.mv_data = vbuf;
                data.mv_size = vsz;

                unsigned pf = ((r & 1u) ? MDB_NODUPDATA : 0u);
                int prc = mdb_put(txn, dbi, &key, &data, pf);
                if (prc != MDB_SUCCESS && prc != MDB_KEYEXIST)
                    CHECK(prc, "dups add put");
                free(vbuf);

            } else if ((r % 100u) < 80u && tot.entries > 0) {
                /* delete a specific duplicate by selecting a random record */
                uint64_t del_r = splitmix64_next(&rng) % tot.entries;
                MDB_val k, d;
                CHECK(mdb_agg_select(txn, dbi, MDB_AGG_WEIGHT_ENTRIES, del_r, &k, &d, NULL), "dups del: select");
                int drc = mdb_del(txn, dbi, &k, &d);
                if (drc != MDB_SUCCESS && drc != MDB_NOTFOUND)
                    CHECK(drc, "dups del(key,data)");

            } else if (tot.entries > 0) {
                /* delete all duplicates for a key */
                uint64_t del_r = splitmix64_next(&rng) % tot.entries;
                MDB_val k, d;
                CHECK(mdb_agg_select(txn, dbi, MDB_AGG_WEIGHT_ENTRIES, del_r, &k, &d, NULL), "dups delall: select");
                int drc = mdb_del(txn, dbi, &k, NULL);
                if (drc != MDB_SUCCESS && drc != MDB_NOTFOUND)
                    CHECK(drc, "dups del(key,NULL)");
            }
        }

        CHECK(mdb_txn_commit(txn), "dups round: commit");

        CHECK(mdb_txn_begin(env, NULL, MDB_RDONLY, &txn), "dups post: ro");
        CHECK(mdb_dbi_open(txn, "dups", 0, &dbi), "dups post: open");
        validate_agg_db(txn, dbi, 1, &rng, "dups validate");
        mdb_txn_abort(txn);
    }

    mdb_env_close(env);
}

static void test_dupfixed_multiple(void)
{
    fprintf(stderr, "\ntest_dupfixed_multiple\n");

    MDB_env *env = open_env_dir(ENV_DIR, 8, (size_t)1024u * 1024u * 1024u, 1);
    MDB_txn *txn = NULL;
    MDB_dbi dbi = 0;
    uint64_t rng = UINT64_C(0x123456789ABCDEF0);

    enum { DF_KEYS = 520, DF_DUPS = 260, MUT_ROUNDS = 4, MUT_OPS = 1400 };
    const size_t ELSZ = (size_t)MDB_HASH_SIZE;

    /* Bulk insert using MDB_MULTIPLE. */
    {
        MDB_val key;
        char kbuf[128];

        CHECK(mdb_txn_begin(env, NULL, 0, &txn), "df bulk: begin");
        CHECK(mdb_dbi_open(txn, "dupfixed", MDB_CREATE | MDB_DUPSORT | MDB_DUPFIXED | AGG_SCHEMA, &dbi), "df bulk: open");

        MDB_cursor *cur = NULL;
        CHECK(mdb_cursor_open(txn, dbi, &cur), "df bulk: cursor_open");

        unsigned char *block = (unsigned char *)malloc((size_t)DF_DUPS * ELSZ);
        if (!block) fatal_errno("malloc df block");

        MDB_val dvals[2];
        dvals[0].mv_size = ELSZ;
        dvals[1].mv_data = NULL;

        for (unsigned i = 0; i < DF_KEYS; ++i) {
            snprintf(kbuf, sizeof(kbuf), "df/%08u", i);
            key.mv_data = kbuf;
            key.mv_size = strlen(kbuf);

            for (unsigned d = 0; d < DF_DUPS; ++d) {
                fill_fixed_elem(block + (size_t)d * ELSZ, ELSZ,
                                ((uint64_t)i << 32) ^ (uint64_t)d ^ UINT64_C(0xBEEF0000));
            }

            /* Robust MDB_MULTIPLE loop: caller provides remaining count in dvals[1].mv_size,
               callee updates it to the number of items actually consumed. */
            size_t remaining = DF_DUPS;
            size_t off = 0;
            while (remaining) {
                dvals[0].mv_data = block + off * ELSZ;
                dvals[1].mv_size = remaining;
                CHECK(mdb_cursor_put(cur, &key, dvals, MDB_MULTIPLE), "df bulk: put multiple");
                size_t consumed = dvals[1].mv_size;
                if (consumed == 0 || consumed > remaining) {
                    fprintf(stderr, "df bulk: MDB_MULTIPLE consumed=%zu remaining=%zu (unexpected)\n", consumed, remaining);
                    exit(EXIT_FAILURE);
                }
                off += consumed;
                remaining -= consumed;
            }
        }

        free(block);
        mdb_cursor_close(cur);
        CHECK(mdb_txn_commit(txn), "df bulk: commit");
    }

    /* Validate + exercise GET_MULTIPLE / NEXT_MULTIPLE. */
    {
        CHECK(mdb_txn_begin(env, NULL, MDB_RDONLY, &txn), "df ro: begin");
        CHECK(mdb_dbi_open(txn, "dupfixed", 0, &dbi), "df ro: open");
        assert_depth_at_least(txn, dbi, 2, "df depth >=2");
        validate_agg_db(txn, dbi, 1, &rng, "df baseline");

        MDB_cursor *cur = NULL;
        CHECK(mdb_cursor_open(txn, dbi, &cur), "df ro: cursor_open");
        MDB_val k, d;
        int rc = mdb_cursor_get(cur, &k, &d, MDB_FIRST);
        while (rc == MDB_SUCCESS) {
            mdb_size_t cc = 0;
            CHECK(mdb_cursor_count(cur, &cc), "df cursor_count");

            uint64_t seen = 0;
            MDB_val md = { 0, NULL };
            rc = mdb_cursor_get(cur, &k, &md, MDB_GET_MULTIPLE);
            while (rc == MDB_SUCCESS) {
                REQUIRE(md.mv_data != NULL, "df GET_MULTIPLE mv_data != NULL");
                REQUIRE(md.mv_size % ELSZ == 0, "df GET_MULTIPLE size multiple of ELSZ");
                seen += (uint64_t)(md.mv_size / ELSZ);
                rc = mdb_cursor_get(cur, &k, &md, MDB_NEXT_MULTIPLE);
            }
            REQUIRE(rc == MDB_NOTFOUND, "df NEXT_MULTIPLE ends with NOTFOUND");
            REQUIRE(seen == (uint64_t)cc, "df saw all dups via MULTIPLE ops");

            rc = mdb_cursor_get(cur, &k, &d, MDB_NEXT_NODUP);
        }
        REQUIRE(rc == MDB_NOTFOUND, "df reached end with NOTFOUND");
        mdb_cursor_close(cur);
        mdb_txn_abort(txn);
    }

    /* Mutations. */
    for (unsigned round = 0; round < MUT_ROUNDS; ++round) {
        CHECK(mdb_txn_begin(env, NULL, 0, &txn), "df mut: begin");
        CHECK(mdb_dbi_open(txn, "dupfixed", 0, &dbi), "df mut: open");

        MDB_val key;
        char kbuf[128];

        for (unsigned op = 0; op < MUT_OPS; ++op) {
            uint64_t r = splitmix64_next(&rng);
            unsigned act = (unsigned)(r % 100u);

            unsigned kid = (unsigned)(splitmix64_next(&rng) % (DF_KEYS + 250));
            snprintf(kbuf, sizeof(kbuf), "df/%08u", kid);
            key.mv_data = kbuf;
            key.mv_size = strlen(kbuf);

            if (act < 45) {
                /* Delete a specific fixed element (if present). */
                unsigned did = (unsigned)(splitmix64_next(&rng) % DF_DUPS);
                unsigned char elem[MDB_HASH_SIZE];
                fill_fixed_elem(elem, ELSZ, ((uint64_t)kid << 32) ^ (uint64_t)did ^ UINT64_C(0xBEEF0000));
                MDB_val dv = { ELSZ, elem };
                int rc = mdb_del(txn, dbi, &key, &dv);
                if (rc != MDB_SUCCESS && rc != MDB_NOTFOUND)
                    CHECK(rc, "df del(key,data)");

            } else if (act < 70) {
                /* Delete entire key (all duplicates). */
                int rc = mdb_del(txn, dbi, &key, NULL);
                if (rc != MDB_SUCCESS && rc != MDB_NOTFOUND)
                    CHECK(rc, "df del(key,NULL)");

            } else {
                /* Insert / replace key with DF_DUPS elements using MDB_MULTIPLE. */
                MDB_cursor *cur = NULL;
                CHECK(mdb_cursor_open(txn, dbi, &cur), "df mut: cursor_open");

                unsigned char *block = (unsigned char *)malloc((size_t)DF_DUPS * ELSZ);
                if (!block) fatal_errno("malloc df mut block");
                for (unsigned d = 0; d < DF_DUPS; ++d)
                    fill_fixed_elem(block + (size_t)d * ELSZ, ELSZ,
                                    ((uint64_t)kid << 32) ^ (uint64_t)d ^ UINT64_C(0xBEEF0000));

                MDB_val dvals[2];
                dvals[0].mv_size = ELSZ;
                dvals[1].mv_data = NULL;

                size_t remaining = DF_DUPS;
                size_t off = 0;
                while (remaining) {
                    dvals[0].mv_data = block + off * ELSZ;
                    dvals[1].mv_size = remaining;
                    int prc = mdb_cursor_put(cur, &key, dvals, MDB_MULTIPLE);
                    if (prc != MDB_SUCCESS) CHECK(prc, "df mut: put multiple");
                    size_t consumed = dvals[1].mv_size;
                    if (consumed == 0 || consumed > remaining) {
                        fprintf(stderr, "df mut: MDB_MULTIPLE consumed=%zu remaining=%zu (unexpected)\n", consumed, remaining);
                        exit(EXIT_FAILURE);
                    }
                    off += consumed;
                    remaining -= consumed;
                }

                free(block);
                mdb_cursor_close(cur);
            }
        }

        CHECK(mdb_txn_commit(txn), "df mut: commit");

        CHECK(mdb_txn_begin(env, NULL, MDB_RDONLY, &txn), "df post: ro");
        CHECK(mdb_dbi_open(txn, "dupfixed", 0, &dbi), "df post: open");
        validate_agg_db(txn, dbi, 1, &rng, "df validate");
        mdb_txn_abort(txn);
    }

    mdb_env_close(env);
}

int main(void)
{
    printf("test_empty_and_errors\n");
    test_empty_and_errors();
    printf("test_upgrade_policy\n");
    test_upgrade_policy();
    printf("test_plain_agg_stress\n");
    test_plain_agg_stress();
    printf("test_dupsort_variable\n");
    test_dupsort_variable();
    printf("test_dupfixed_multiple\n");
    test_dupfixed_multiple();
    fprintf(stdout, "\nALL OK\n");
    return 0;
}
