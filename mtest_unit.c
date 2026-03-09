/* mtest_agg_unit.c - deterministic unit tests for LMDB multi-aggregate extension
 *
 * This suite avoids randomization and instead applies targeted strategies that
 * force specific B+tree behaviors (splits, merges, rebalances, root collapse)
 * while verifying integrity and correctness of aggregate components:
 *   - MDB_AGG_ENTRIES  : logical entries (key/value pairs, includes duplicates)
 *   - MDB_AGG_KEYS     : distinct keys (differs from entries for DUPSORT DBIs)
 *   - MDB_AGG_HASHSUM  : wraparound sum of first MDB_HASH_SIZE bytes of each value
 *
 * After every modifying operation we invoke the aggregate debug checker when
 * compiled with MDB_DEBUG_AGG_INTEGRITY.
 *
 * Build (example):
 *   cc -O2 -std=c99 -I. mtest_unit.c mdb.c midl.c -o mtest_unit
 *   cc -g -std=c99 -DMDB_DEBUG_AGG_PRINT -DMDB_DEBUG_AGG_INTEGRITY -DMDB_DEBUG -I. mtest_unit.c mdb.c midl.c -o mtest_unit
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

#ifndef MDB_AGG_ENTRIES
#error "This test requires the multi-aggregate API (MDB_AGG_ENTRIES/MDB_agg/mdb_agg_*) in lmdb.h"
#endif

#define ENV_BASE "/tmp/testdb_agg_unit"

#define CHECK(rc, msg) do {                                                     \
    if ((rc) != MDB_SUCCESS) {                                                  \
        fprintf(stderr, "%s: %s (%d)\n", (msg), mdb_strerror(rc), (rc));      \
        exit(EXIT_FAILURE);                                                     \
    }                                                                           \
} while (0)

#define REQUIRE(cond, msg) do {                                                 \
    if (!(cond)) {                                                              \
        fprintf(stderr, "%s\n", (msg));                                       \
        exit(EXIT_FAILURE);                                                     \
    }                                                                           \
} while (0)

#ifdef MDB_DEBUG_AGG_INTEGRITY
#define DBG_CHECK(txn, dbi, msg) do { CHECK(mdb_dbg_check_agg_db((txn), (dbi)), (msg)); } while (0)
#else
#define DBG_CHECK(txn, dbi, msg) do { (void)(txn); (void)(dbi); (void)(msg); } while (0)
#endif

/* ------------------------------ hashsum algebra smoke-test ------------------------------ */

/* Reference implementation for hashsum wraparound arithmetic.
 * Semantics: treat the MDB_HASH_SIZE bytes as a little-endian unsigned integer.
 */
static void ref_hashsum_add(uint8_t *acc, const uint8_t *x)
{
    unsigned carry = 0;
    for (unsigned i = 0; i < MDB_HASH_SIZE; i++) {
        unsigned s = (unsigned)acc[i] + (unsigned)x[i] + carry;
        acc[i] = (uint8_t)s;
        carry = s >> 8;
    }
}

static void ref_hashsum_sub(uint8_t *acc, const uint8_t *x)
{
    unsigned borrow = 0;
    for (unsigned i = 0; i < MDB_HASH_SIZE; i++) {
        unsigned xi = (unsigned)x[i] + borrow;
        borrow = (unsigned)acc[i] < xi;
        acc[i] = (uint8_t)((unsigned)acc[i] - xi);
    }
}

static void ref_hashsum_diff(uint8_t *out, const uint8_t *a, const uint8_t *b)
{
    memcpy(out, a, MDB_HASH_SIZE);
    ref_hashsum_sub(out, b);
}

static void fill_hashsum_pattern(uint8_t *dst, uint64_t seed)
{
    /* Deterministic, non-cryptographic byte stream (LCG). */
    uint64_t x = seed ? seed : 0x9e3779b97f4a7c15ULL;
    for (unsigned i = 0; i < MDB_HASH_SIZE; i++) {
        x = x * 6364136223846793005ULL + 1442695040888963407ULL;
        dst[i] = (uint8_t)(x >> 56);
    }
}

static void test_hashsum_algebra(void)
{
    /* Explicit carry propagation across all bytes: (2^N - 1) + 1 == 0 */
    {
        uint8_t acc[MDB_HASH_SIZE];
        uint8_t one[MDB_HASH_SIZE];
        memset(acc, 0xFF, sizeof(acc));
        memset(one, 0, sizeof(one));
        one[0] = 1;

        uint8_t got[MDB_HASH_SIZE];
        uint8_t exp[MDB_HASH_SIZE];
        memcpy(got, acc, MDB_HASH_SIZE);
        memcpy(exp, acc, MDB_HASH_SIZE);
        mdb_hashsum_add(got, one);
        ref_hashsum_add(exp, one);
        REQUIRE(memcmp(got, exp, MDB_HASH_SIZE) == 0, "hashsum(add): full carry propagation mismatch");

        uint8_t z[MDB_HASH_SIZE];
        memset(z, 0, sizeof(z));
        REQUIRE(memcmp(got, z, MDB_HASH_SIZE) == 0, "hashsum(add): expected wrap to zero");
    }

    /* Explicit borrow propagation across all bytes: 0 - 1 == (2^N - 1) */
    {
        uint8_t acc[MDB_HASH_SIZE];
        uint8_t one[MDB_HASH_SIZE];
        memset(acc, 0, sizeof(acc));
        memset(one, 0, sizeof(one));
        one[0] = 1;

        uint8_t got[MDB_HASH_SIZE];
        uint8_t exp[MDB_HASH_SIZE];
        memcpy(got, acc, MDB_HASH_SIZE);
        memcpy(exp, acc, MDB_HASH_SIZE);
        mdb_hashsum_sub(got, one);
        ref_hashsum_sub(exp, one);
        REQUIRE(memcmp(got, exp, MDB_HASH_SIZE) == 0, "hashsum(sub): full borrow propagation mismatch");

        uint8_t allff[MDB_HASH_SIZE];
        memset(allff, 0xFF, sizeof(allff));
        REQUIRE(memcmp(got, allff, MDB_HASH_SIZE) == 0, "hashsum(sub): expected wrap to all-0xFF");
    }

    /* Boundary carry/borrow across an 8-byte limb edge (implementation uses 64-bit limbs). */
    if (MDB_HASH_SIZE >= 16) {
        uint8_t got[MDB_HASH_SIZE];
        uint8_t exp[MDB_HASH_SIZE];
        uint8_t x[MDB_HASH_SIZE];

        memset(got, 0, sizeof(got));
        memset(exp, 0, sizeof(exp));
        memset(x, 0, sizeof(x));

        /* set low limb to 0xFF..FF so adding 1 carries into byte 8 */
        for (unsigned i = 0; i < 8; i++) {
            got[i] = 0xFF;
            exp[i] = 0xFF;
        }
        x[0] = 1;

        mdb_hashsum_add(got, x);
        ref_hashsum_add(exp, x);
        REQUIRE(memcmp(got, exp, MDB_HASH_SIZE) == 0, "hashsum(add): limb-edge carry mismatch");
        REQUIRE(got[8] == 1, "hashsum(add): expected carry into byte 8");

        /* Now borrow back across the same edge. */
        mdb_hashsum_sub(got, x);
        ref_hashsum_sub(exp, x);
        REQUIRE(memcmp(got, exp, MDB_HASH_SIZE) == 0, "hashsum(sub): limb-edge borrow mismatch");
        for (unsigned i = 0; i < 8; i++)
            REQUIRE(got[i] == 0xFF, "hashsum(sub): expected low limb restored to 0xFF");
        REQUIRE(got[8] == 0, "hashsum(sub): expected byte 8 restored to 0");
    }

    /* Fuzz-ish deterministic cross-checks vs reference implementation. */
    for (unsigned t = 0; t < 400; t++) {
        uint8_t a[MDB_HASH_SIZE];
        uint8_t b[MDB_HASH_SIZE];
        uint8_t got[MDB_HASH_SIZE];
        uint8_t exp[MDB_HASH_SIZE];
        uint8_t dgot[MDB_HASH_SIZE];
        uint8_t dexp[MDB_HASH_SIZE];

        fill_hashsum_pattern(a, 0xA55AA55AULL ^ (uint64_t)t * 0x9E3779B97F4A7C15ULL);
        fill_hashsum_pattern(b, 0xC001C0DEULL ^ (uint64_t)t * 0xD1B54A32D192ED03ULL);

        /* add */
        memcpy(got, a, MDB_HASH_SIZE);
        memcpy(exp, a, MDB_HASH_SIZE);
        mdb_hashsum_add(got, b);
        ref_hashsum_add(exp, b);
        REQUIRE(memcmp(got, exp, MDB_HASH_SIZE) == 0, "hashsum(add): mismatch vs reference");

        /* sub */
        memcpy(got, a, MDB_HASH_SIZE);
        memcpy(exp, a, MDB_HASH_SIZE);
        mdb_hashsum_sub(got, b);
        ref_hashsum_sub(exp, b);
        REQUIRE(memcmp(got, exp, MDB_HASH_SIZE) == 0, "hashsum(sub): mismatch vs reference");

        /* diff */
        mdb_hashsum_diff(dgot, a, b);
        ref_hashsum_diff(dexp, a, b);
        REQUIRE(memcmp(dgot, dexp, MDB_HASH_SIZE) == 0, "hashsum(diff): mismatch vs reference");

        /* Roundtrip: (a + b) - b == a */
        memcpy(got, a, MDB_HASH_SIZE);
        mdb_hashsum_add(got, b);
        mdb_hashsum_sub(got, b);
        REQUIRE(memcmp(got, a, MDB_HASH_SIZE) == 0, "hashsum: add/sub roundtrip failed");

        /* Consistency: diff(a,b) + b == a */
        memcpy(got, dgot, MDB_HASH_SIZE);
        mdb_hashsum_add(got, b);
        REQUIRE(memcmp(got, a, MDB_HASH_SIZE) == 0, "hashsum: diff/add consistency failed");
    }
}

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

static void create_clean_dir(const char *dir)
{
        /* Keep it simple: ensure the directory exists and remove prior LMDB files. */
        ensure_dir(ENV_BASE);
        ensure_dir(dir);
        unlink_env_files(dir);
}

static void make_key_string(MDB_val *k, const char *s)
{
        k->mv_data = (void *)s;
        k->mv_size = strlen(s) + 1u; /* include NUL to match existing test patterns */
}

static void make_val_with_hash(uint8_t *buf, size_t cap, const char *tag, unsigned id, MDB_val *out);

static void seed_main_keys(MDB_txn *txn, MDB_dbi dbi, const char *tag, unsigned nkeys)
{
        uint8_t smallbuf[256];
        for (unsigned i = 0; i < nkeys; i++) {
                char kbuf[32];
                MDB_val k, d;
                snprintf(kbuf, sizeof(kbuf), "seed/%s/%08u", tag ? tag : "x", i);
                k.mv_data = kbuf;
                k.mv_size = strlen(kbuf) + 1u;
                make_val_with_hash(smallbuf, sizeof(smallbuf), tag ? tag : "seed", i, &d);
                CHECK(mdb_put(txn, dbi, &k, &d, 0), "seed_main_keys: put");
        }
}


static MDB_env *open_env_dir(const char *dir, int maxdbs, size_t mapsize, int fresh)
{
    MDB_env *env = NULL;

    ensure_dir(ENV_BASE);
    ensure_dir(dir);

    if (fresh)
        unlink_env_files(dir);

    CHECK(mdb_env_create(&env), "mdb_env_create");
    CHECK(mdb_env_set_maxdbs(env, maxdbs), "mdb_env_set_maxdbs");
    CHECK(mdb_env_set_mapsize(env, mapsize), "mdb_env_set_mapsize");

    /* Fast test defaults. Optionally enable internal aggregate checks. */
    {
        unsigned int eflags = MDB_NOLOCK | MDB_NOSYNC | MDB_NOMETASYNC;
#ifdef MDB_DEBUG_AGG_INTEGRITY
        eflags |= MDB_AGG_CHECK;
#endif
        CHECK(mdb_env_open(env, dir, eflags, 0664), "mdb_env_open");
    }
    return env;
}



static MDB_env *open_env_dir_flags(const char *dir, int maxdbs, size_t mapsize, int fresh, unsigned extra_eflags)
{
    MDB_env *env = NULL;

    ensure_dir(ENV_BASE);
    ensure_dir(dir);

    if (fresh)
        unlink_env_files(dir);

    CHECK(mdb_env_create(&env), "mdb_env_create");
    CHECK(mdb_env_set_maxdbs(env, maxdbs), "mdb_env_set_maxdbs");
    CHECK(mdb_env_set_mapsize(env, mapsize), "mdb_env_set_mapsize");

    {
        unsigned int eflags = MDB_NOLOCK | MDB_NOSYNC | MDB_NOMETASYNC | extra_eflags;
#ifdef MDB_DEBUG_AGG_INTEGRITY
        eflags |= MDB_AGG_CHECK;
#endif
        CHECK(mdb_env_open(env, dir, eflags, 0664), "mdb_env_open");
    }
    return env;
}

static void make_test_dir(char *out, size_t cap, const char *name)
{
    snprintf(out, cap, "%s/%s", ENV_BASE, name);
}

/* ------------------------------ deterministic KV generators ------------------------------ */

static void make_key(char *buf, size_t cap, const char *pfx, unsigned id)
{
    snprintf(buf, cap, "%s%08u", pfx, id);
}


static void make_key_padded(char *buf, size_t cap, const char *pfx, unsigned id, size_t padlen, char padch)
{
    /* Build a key that is lexicographically ordered by (pfx,id) but with a large
     * suffix to pressure branch-node key sizes (mdb_update_key split triggers).
     */
    int n = snprintf(buf, cap, "%s%08u/", pfx, id);
    REQUIRE(n > 0, "make_key_padded: snprintf failed");
    REQUIRE((size_t)n < cap, "make_key_padded: cap too small");
    REQUIRE((size_t)n + padlen + 1u <= cap, "make_key_padded: cap too small for pad");

    memset(buf + n, padch, padlen);
    buf[n + (int)padlen] = '\0';
}

static void fill_hash_prefix(uint8_t *dst, uint64_t tag)
{
    /* Deterministic pseudo-random bytes derived from tag. */
    uint64_t x = tag ^ 0x9E3779B97F4A7C15ULL;
    for (unsigned i = 0; i < MDB_HASH_SIZE; i++) {
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        dst[i] = (uint8_t)(x & 0xFFu) ^ (uint8_t)i;
    }
}

static void make_val_with_hash(uint8_t *buf, size_t cap, const char *tag, unsigned id, MDB_val *out)
{
    REQUIRE(cap >= (size_t)MDB_HASH_SIZE + 64u, "buffer too small for value");

    fill_hash_prefix(buf, ((uint64_t)id << 32) ^ 0xA55AA55Au);
    int n = snprintf((char *)buf + MDB_HASH_SIZE, cap - (size_t)MDB_HASH_SIZE,
                     "%s:%08u:%c%c%c%c%c%c%c%c",
                     tag, id,
                     (char)('A' + (id % 26u)), (char)('B' + (id % 26u)), (char)('C' + (id % 26u)),
                     (char)('D' + (id % 26u)), (char)('E' + (id % 26u)), (char)('F' + (id % 26u)),
                     (char)('G' + (id % 26u)), (char)('H' + (id % 26u)));
    REQUIRE(n > 0, "snprintf failed");

    out->mv_data = buf;
    out->mv_size = (size_t)MDB_HASH_SIZE + (size_t)n;
}



static void make_val_with_hash_salted(uint8_t *buf, size_t cap, const char *tag, unsigned id, uint64_t salt, MDB_val *out)
{
    REQUIRE(cap >= (size_t)MDB_HASH_SIZE + 64u, "buffer too small for value");

    /* fixed-width suffix so sizes remain stable across salts */
    fill_hash_prefix(buf, (((uint64_t)id << 32) ^ 0xA55AA55Au) ^ salt);
    int n = snprintf((char *)buf + MDB_HASH_SIZE, cap - (size_t)MDB_HASH_SIZE,
                     "%s:%08u:%016" PRIx64,
                     tag, id, (uint64_t)salt);
    REQUIRE(n > 0, "snprintf failed");

    out->mv_data = buf;
    out->mv_size = (size_t)MDB_HASH_SIZE + (size_t)n;
}

static void fill_prefix_bytes(uint8_t *dst, size_t n, uint64_t tag)
{
    /* Deterministic bytes for the non-hash prefix, distinct from fill_hash_prefix(). */
    uint64_t x = tag ^ 0xD1B54A32D192ED03ULL;
    for (size_t i = 0; i < n; i++) {
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        dst[i] = (uint8_t)(x & 0xFFu) ^ (uint8_t)(i * 13u);
    }
}

static void make_val_with_hash_offset_salted(uint8_t *buf, size_t cap, size_t off,
                                             const char *tag, unsigned id, uint64_t salt,
                                             MDB_val *out)
{
    REQUIRE(cap >= off + (size_t)MDB_HASH_SIZE + 64u, "buffer too small for offset value");

    /* Prefix controls ordering / user payload; hash starts at offset. Prefix is salt-independent. */
    fill_prefix_bytes(buf, off, ((uint64_t)id << 1) ^ 0xBADC0FFEE0DDF00DULL);

    /* Hash bytes at offset. */
    fill_hash_prefix(buf + off, (((uint64_t)id << 32) ^ 0xA55AA55Au) ^ salt);

    /* Fixed-width suffix so sizes remain stable across salts (also DUPFIXED-friendly). */
    int n = snprintf((char *)buf + off + MDB_HASH_SIZE,
                     cap - (off + (size_t)MDB_HASH_SIZE),
                     "%s:%08u:%016" PRIx64,
                     tag, id, (uint64_t)salt);
    REQUIRE(n > 0, "snprintf failed");

    out->mv_data = buf;
    out->mv_size = off + (size_t)MDB_HASH_SIZE + (size_t)n;
}

static void expect_hash_for_id_salt(uint8_t out[MDB_HASH_SIZE], unsigned id, uint64_t salt)
{
    fill_hash_prefix(out, (((uint64_t)id << 32) ^ 0xA55AA55Au) ^ salt);
}

static void expect_prefix_first_hashbytes(uint8_t out[MDB_HASH_SIZE], unsigned id)
{
    /* For our offset tests, off >= MDB_HASH_SIZE, so the first MDB_HASH_SIZE bytes are prefix-only. */
    fill_prefix_bytes(out, (size_t)MDB_HASH_SIZE, ((uint64_t)id << 1) ^ 0xBADC0FFEE0DDF00DULL);
}

static void make_big_val_with_hash(uint8_t *buf, size_t cap, unsigned id, size_t extra, MDB_val *out)
{
    REQUIRE(cap >= (size_t)MDB_HASH_SIZE + extra, "buffer too small for big value");
    fill_hash_prefix(buf, ((uint64_t)id << 32) ^ 0xC001C0DEu);
    for (size_t i = 0; i < extra; i++) {
        /* Deterministic body bytes; not used by aggregator except for storage pressure. */
        buf[MDB_HASH_SIZE + i] = (uint8_t)((id * 131u) ^ (unsigned)i);
    }
    out->mv_data = buf;
    out->mv_size = (size_t)MDB_HASH_SIZE + extra;
}


static void fill_hash_prefix_monotonic32(uint8_t *dst, uint32_t id)
{
    /* Monotonic lexicographic prefix for DUPSORT ordering: big-endian id repeated. */
    const uint8_t b0 = (uint8_t)((id >> 24) & 0xFFu);
    const uint8_t b1 = (uint8_t)((id >> 16) & 0xFFu);
    const uint8_t b2 = (uint8_t)((id >>  8) & 0xFFu);
    const uint8_t b3 = (uint8_t)((id >>  0) & 0xFFu);
    for (unsigned i = 0; i < MDB_HASH_SIZE; i += 4) {
        dst[i + 0] = b0;
        dst[i + 1] = b1;
        dst[i + 2] = b2;
        dst[i + 3] = b3;
    }
}

static void make_big_val_with_monotonic_hash(uint8_t *buf, size_t cap, unsigned id, size_t extra, MDB_val *out)
{
    REQUIRE(cap >= (size_t)MDB_HASH_SIZE + extra, "buffer too small for big value");
    fill_hash_prefix_monotonic32(buf, (uint32_t)id);
    for (size_t i = 0; i < extra; i++) {
        buf[MDB_HASH_SIZE + i] = (uint8_t)((id * 131u) ^ (unsigned)i);
    }
    out->mv_data = buf;
    out->mv_size = (size_t)MDB_HASH_SIZE + extra;
}

#define AGG_SCHEMA (MDB_AGG_ENTRIES|MDB_AGG_KEYS|MDB_AGG_HASHSUM)

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

typedef struct {
    size_t *idx;      /* indices in RecVec of first record for each distinct key */
    size_t len;
    size_t cap;
} KeyIdxVec;

static void recvec_init(RecVec *vv)
{
    vv->v = NULL;
    vv->len = 0;
    vv->cap = 0;
}

static void recvec_free(RecVec *vv)
{
    if (!vv) return;
    for (size_t i = 0; i < vv->len; i++) {
        free(vv->v[i].k);
        free(vv->v[i].d);
    }
    free(vv->v);
    vv->v = NULL;
    vv->len = 0;
    vv->cap = 0;
}

static void keyidx_free(KeyIdxVec *kv)
{
    if (!kv) return;
    free(kv->idx);
    kv->idx = NULL;
    kv->len = 0;
    kv->cap = 0;
}

static void recvec_push_copy(RecVec *vv, const MDB_val *key, const MDB_val *data)
{
    if (vv->len == vv->cap) {
        size_t ncap = vv->cap ? vv->cap * 2u : 256u;
        Rec *nv = (Rec *)realloc(vv->v, ncap * sizeof(Rec));
        REQUIRE(nv != NULL, "realloc failed");
        vv->v = nv;
        vv->cap = ncap;
    }

    Rec *r = &vv->v[vv->len++];
    r->klen = key->mv_size;
    r->dlen = data->mv_size;

    r->k = NULL;
    r->d = NULL;

    if (r->klen) {
        r->k = (unsigned char *)malloc(r->klen);
        REQUIRE(r->k != NULL, "malloc key failed");
        memcpy(r->k, key->mv_data, r->klen);
    }

    if (r->dlen) {
        r->d = (unsigned char *)malloc(r->dlen);
        REQUIRE(r->d != NULL, "malloc data failed");
        memcpy(r->d, data->mv_data, r->dlen);
    }
}

static int key_equal_rec(const Rec *a, const Rec *b)
{
    if (a->klen != b->klen) return 0;
    if (a->klen == 0) return 1;
    return memcmp(a->k, b->k, a->klen) == 0;
}

static void oracle_build(MDB_txn *txn, MDB_dbi dbi, RecVec *out)
{
    MDB_cursor *cur = NULL;
    MDB_val k, d;
    int rc;

    recvec_init(out);
    CHECK(mdb_cursor_open(txn, dbi, &cur), "mdb_cursor_open");

    rc = mdb_cursor_get(cur, &k, &d, MDB_FIRST);
    while (rc == MDB_SUCCESS) {
        recvec_push_copy(out, &k, &d);
        rc = mdb_cursor_get(cur, &k, &d, MDB_NEXT);
    }
    if (rc != MDB_NOTFOUND)
        CHECK(rc, "oracle cursor walk");

    mdb_cursor_close(cur);
}

static void oracle_build_keyidx(const RecVec *vv, KeyIdxVec *kv)
{
    kv->idx = NULL;
    kv->len = 0;
    kv->cap = 0;

    for (size_t i = 0; i < vv->len; i++) {
        if (i == 0 || !key_equal_rec(&vv->v[i], &vv->v[i - 1])) {
            if (kv->len == kv->cap) {
                size_t ncap = kv->cap ? kv->cap * 2u : 64u;
                size_t *ni = (size_t *)realloc(kv->idx, ncap * sizeof(size_t));
                REQUIRE(ni != NULL, "realloc keyidx failed");
                kv->idx = ni;
                kv->cap = ncap;
            }
            kv->idx[kv->len++] = i;
        }
    }
}

static uint64_t oracle_dup_index(const RecVec *vv, size_t rec_index)
{
    if (rec_index >= vv->len) return 0;
    uint64_t di = 0;
    for (size_t i = rec_index; i > 0; i--) {
        if (!key_equal_rec(&vv->v[i], &vv->v[i - 1]))
            break;
        di++;
    }
    return di;
}

static int bytes_cmp_lex(const void *a, size_t alen, const void *b, size_t blen)
{
    size_t n = alen < blen ? alen : blen;
    int c = 0;
    if (n)
        c = memcmp(a, b, n);
    if (c)
        return c;
    if (alen < blen)
        return -1;
    if (alen > blen)
        return 1;
    return 0;
}

static int rec_key_eq_val(const Rec *r, const MDB_val *k)
{
    if (r->klen != k->mv_size)
        return 0;
    if (r->klen == 0)
        return 1;
    return memcmp(r->k, k->mv_data, r->klen) == 0;
}

static void find_key_run_or_die(const RecVec *vv, const MDB_val *key, size_t *start, size_t *end)
{
    size_t s = (size_t)-1;
    size_t e = (size_t)-1;
    for (size_t i = 0; i < vv->len; i++) {
        if (rec_key_eq_val(&vv->v[i], key)) {
            if (s == (size_t)-1)
                s = i;
            e = i + 1;
        } else if (s != (size_t)-1) {
            break;
        }
    }
    if (s == (size_t)-1) {
        fprintf(stderr, "find_key_run: key not found in oracle\n");
        exit(EXIT_FAILURE);
    }
    *start = s;
    *end = e;
}

static size_t lower_bound_dups(const RecVec *vv, size_t start, size_t end, const MDB_val *qd)
{
    /* Return first index in [start,end) with data >= qd (lex order). */
    for (size_t i = start; i < end; i++) {
        const Rec *r = &vv->v[i];
        int c = bytes_cmp_lex(r->d, r->dlen, qd->mv_data, qd->mv_size);
        if (c >= 0)
            return i;
    }
    return end;
}

static void oracle_hashsum_total(const RecVec *vv, uint8_t acc[MDB_HASH_SIZE])
{
    memset(acc, 0, MDB_HASH_SIZE);
    for (size_t i = 0; i < vv->len; i++) {
        REQUIRE(vv->v[i].dlen >= MDB_HASH_SIZE, "value too small for hashsum oracle");
        mdb_hashsum_add(acc, (const uint8_t *)vv->v[i].d);
    }
}

static int rec_key_cmp(MDB_txn *txn, MDB_dbi dbi, const Rec *r, const MDB_val *b)
{
    MDB_val rk;
    rk.mv_size = r->klen;
    rk.mv_data = r->k;
    return mdb_cmp(txn, dbi, &rk, b);
}

static void oracle_range_agg(MDB_txn *txn, MDB_dbi dbi, const RecVec *vv,
                            const MDB_val *low, const MDB_val *high, unsigned flags,
                            MDB_agg *out)
{
    const int lincl = (flags & MDB_RANGE_LOWER_INCL) != 0;
    const int hincl = (flags & MDB_RANGE_UPPER_INCL) != 0;

    memset(out, 0, sizeof(*out));
    out->mv_flags = AGG_SCHEMA; /* used only for comparisons in tests */

    uint8_t hacc[MDB_HASH_SIZE];
    memset(hacc, 0, MDB_HASH_SIZE);

    uint64_t entries = 0, keys = 0;
    int have_last = 0;
    const Rec *last = NULL;

    for (size_t i = 0; i < vv->len; i++) {
        const Rec *r = &vv->v[i];

        if (low) {
            int c = rec_key_cmp(txn, dbi, r, low);
            if (c < 0 || (!lincl && c == 0))
                continue;
        }
        if (high) {
            int c = rec_key_cmp(txn, dbi, r, high);
            if (c > 0 || (!hincl && c == 0))
                continue;
        }

        /* include this record */
        entries++;
        REQUIRE(r->dlen >= MDB_HASH_SIZE, "value too small for range hashsum");
        mdb_hashsum_add(hacc, (const uint8_t *)r->d);

        if (!have_last || !key_equal_rec(r, last)) {
            keys++;
            have_last = 1;
            last = r;
        }
    }

    out->mv_agg_entries = entries;
    out->mv_agg_keys = keys;
    memcpy(out->mv_agg_hashes, hacc, MDB_HASH_SIZE);
}


/* --- record-order (key,data) oracle helpers for mdb_agg_prefix/mdb_agg_range --- */

static int rec_pair_cmp(MDB_txn *txn, MDB_dbi dbi, const Rec *r,
                        const MDB_val *k, const MDB_val *d)
{
    MDB_val rk = { r->klen, r->k };
    int c = mdb_cmp(txn, dbi, &rk, (MDB_val *)k);
    if (c)
        return c;
    if (!d)
        return 0;
    /* Plain DBIs have no data comparator; compare by data only for MDB_DUPSORT. */
    unsigned int dbi_flags = 0;
    if (mdb_dbi_flags(txn, dbi, &dbi_flags) != MDB_SUCCESS || (dbi_flags & MDB_DUPSORT) == 0)
        return 0;
    MDB_val rd = { r->dlen, r->d };
    return mdb_dcmp(txn, dbi, &rd, (MDB_val *)d);
}

static int oracle_find_first_duprun(const RecVec *vv, size_t *start, size_t *end)
{
    if (!vv || vv->len < 2)
        return 0;
    for (size_t i = 1; i < vv->len; i++) {
        if (key_equal_rec(&vv->v[i], &vv->v[i - 1])) {
            size_t s = i - 1;
            size_t e = i + 1;
            while (s > 0 && key_equal_rec(&vv->v[s], &vv->v[s - 1]))
                s--;
            while (e < vv->len && key_equal_rec(&vv->v[e], &vv->v[s]))
                e++;
            *start = s;
            *end = e;
            return 1;
        }
    }
    return 0;
}

static void oracle_prefix_agg(MDB_txn *txn, MDB_dbi dbi, const RecVec *vv,
                              const MDB_val *key, const MDB_val *data, unsigned flags,
                              MDB_agg *out)
{
    unsigned dbi_flags = 0;
    CHECK(mdb_dbi_flags(txn, dbi, &dbi_flags), "mdb_dbi_flags (oracle_prefix)");
    const int use_data = (data != NULL) && ((dbi_flags & MDB_DUPSORT) != 0);
    const int incl = (flags & MDB_AGG_PREFIX_INCL) != 0;

    memset(out, 0, sizeof(*out));
    out->mv_flags = AGG_SCHEMA; /* used only for comparisons in tests */

    uint8_t hacc[MDB_HASH_SIZE];
    memset(hacc, 0, MDB_HASH_SIZE);

    uint64_t entries = 0, keys = 0;
    int have_last = 0;
    const Rec *last = NULL;

    for (size_t i = 0; i < vv->len; i++) {
        const Rec *r = &vv->v[i];

        int c;
        if (use_data) {
            c = rec_pair_cmp(txn, dbi, r, key, data);
        } else {
            c = rec_key_cmp(txn, dbi, r, key);
        }

        if (c < 0 || (incl && c == 0)) {
            entries++;
            REQUIRE(r->dlen >= MDB_HASH_SIZE, "value too small for prefix hashsum");
            mdb_hashsum_add(hacc, (const uint8_t *)r->d);

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

static void oracle_range_agg_pair(MDB_txn *txn, MDB_dbi dbi, const RecVec *vv,
                             const MDB_val *low_key, const MDB_val *low_data,
                             const MDB_val *high_key, const MDB_val *high_data,
                             unsigned flags, MDB_agg *out)
{
    unsigned dbi_flags = 0;
    CHECK(mdb_dbi_flags(txn, dbi, &dbi_flags), "mdb_dbi_flags (oracle_range)");
    const int is_dupsort = (dbi_flags & MDB_DUPSORT) != 0;

    /* If not DUPSORT, or no data bounds are provided, fall back to key-range semantics. */
    if (!is_dupsort || (!low_data && !high_data)) {
        oracle_range_agg(txn, dbi, vv, low_key, high_key, flags, out);
        return;
    }

    const int lincl = (flags & MDB_RANGE_LOWER_INCL) != 0;
    const int hincl = (flags & MDB_RANGE_UPPER_INCL) != 0;

    memset(out, 0, sizeof(*out));
    out->mv_flags = AGG_SCHEMA; /* used only for comparisons in tests */

    uint8_t hacc[MDB_HASH_SIZE];
    memset(hacc, 0, MDB_HASH_SIZE);

    uint64_t entries = 0, keys = 0;
    int have_last = 0;
    const Rec *last = NULL;

    for (size_t i = 0; i < vv->len; i++) {
        const Rec *r = &vv->v[i];

        if (low_key) {
            int c = low_data ? rec_pair_cmp(txn, dbi, r, low_key, low_data)
                             : rec_key_cmp(txn, dbi, r, low_key);
            if (c < 0 || (!lincl && c == 0))
                continue;
        }
        if (high_key) {
            int c = high_data ? rec_pair_cmp(txn, dbi, r, high_key, high_data)
                              : rec_key_cmp(txn, dbi, r, high_key);
            if (c > 0 || (!hincl && c == 0))
                continue;
        }

        entries++;
        REQUIRE(r->dlen >= MDB_HASH_SIZE, "value too small for range hashsum");
        mdb_hashsum_add(hacc, (const uint8_t *)r->d);

        if (!have_last || !key_equal_rec(r, last)) {
            keys++;
            have_last = 1;
            last = r;
        }
    }

    out->mv_agg_entries = entries;
    out->mv_agg_keys = keys;
    memcpy(out->mv_agg_hashes, hacc, MDB_HASH_SIZE);
}



/* -------------------- Step 1/2/3 API oracles & helpers -------------------- */

static void require_mdbvals_eq_rec(const MDB_val *k, const MDB_val *d, const Rec *r, const char *where)
{
    REQUIRE(k != NULL, "require_mdbvals_eq_rec: NULL key");
    REQUIRE(d != NULL, "require_mdbvals_eq_rec: NULL data");
    REQUIRE(k->mv_size == r->klen, "mdbval key size mismatch");
    REQUIRE(d->mv_size == r->dlen, "mdbval data size mismatch");
    if (r->klen)
        REQUIRE(memcmp(k->mv_data, r->k, r->klen) == 0, where);
    if (r->dlen)
        REQUIRE(memcmp(d->mv_data, r->d, r->dlen) == 0, where);
}

/* Oracle aggregate over an absolute rank interval [abs_begin, abs_end). */
static void oracle_rank_range_agg(const RecVec *vv, size_t abs_begin, size_t abs_end, MDB_agg *out)
{
    memset(out, 0, sizeof(*out));
    out->mv_flags = AGG_SCHEMA; /* used only for comparisons in tests */

    if (!vv) return;
    if (abs_end > vv->len) abs_end = vv->len;
    if (abs_begin > abs_end) abs_begin = abs_end;

    uint8_t hacc[MDB_HASH_SIZE];
    memset(hacc, 0, MDB_HASH_SIZE);

    uint64_t entries = 0, keys = 0;
    int have_last = 0;
    const Rec *last = NULL;

    for (size_t i = abs_begin; i < abs_end; i++) {
        const Rec *r = &vv->v[i];
        entries++;
        REQUIRE(r->dlen >= MDB_HASH_SIZE, "value too small for rank-range hashsum");
        mdb_hashsum_add(hacc, (const uint8_t *)r->d);

        if (!have_last || !key_equal_rec(r, last)) {
            keys++;
            have_last = 1;
            last = r;
        }
    }

    out->mv_agg_entries = entries;
    out->mv_agg_keys = keys;
    memcpy(out->mv_agg_hashes, hacc, MDB_HASH_SIZE);
}

/* Oracle lower bound for key-only query: first index whose key is >= key (if incl) or > key (if !incl). */
static size_t oracle_lower_bound_key_only(MDB_txn *txn, MDB_dbi dbi, const RecVec *vv, const MDB_val *key, int incl)
{
    if (!key) return 0;
    for (size_t i = 0; i < vv->len; i++) {
        int c = rec_key_cmp(txn, dbi, &vv->v[i], key);
        if (c > 0 || (incl && c == 0))
            return i;
    }
    return vv->len;
}

/* Oracle upper bound for key-only window endpoint:
 * - if upper_incl: first index whose key is > key
 * - if !upper_incl: first index whose key is >= key
 */
static size_t oracle_upper_bound_key_only(MDB_txn *txn, MDB_dbi dbi, const RecVec *vv, const MDB_val *key, int upper_incl)
{
    if (!key) return vv->len;
    for (size_t i = 0; i < vv->len; i++) {
        int c = rec_key_cmp(txn, dbi, &vv->v[i], key);
        if (c > 0 || (!upper_incl && c == 0))
            return i;
    }
    return vv->len;
}

static void oracle_window_bounds_key_only(MDB_txn *txn, MDB_dbi dbi, const RecVec *vv,
                                         const MDB_val *low_key, const MDB_val *high_key,
                                         unsigned range_flags,
                                         size_t *abs_lo, size_t *abs_hi)
{
    const int lower_incl = (range_flags & MDB_RANGE_LOWER_INCL) != 0;
    const int upper_incl = (range_flags & MDB_RANGE_UPPER_INCL) != 0;

    size_t lo = oracle_lower_bound_key_only(txn, dbi, vv, low_key, lower_incl);
    size_t hi = oracle_upper_bound_key_only(txn, dbi, vv, high_key, upper_incl);

    if (hi < lo) hi = lo;

    if (abs_lo) *abs_lo = lo;
    if (abs_hi) *abs_hi = hi;
}

/* Oracle for window-relative rank2(key-only): clamp lower-bound into [abs_lo, abs_hi] and subtract abs_lo. */
static uint64_t oracle_window_rel_rank_key_only(MDB_txn *txn, MDB_dbi dbi, const RecVec *vv,
                                               size_t abs_lo, size_t abs_hi,
                                               const MDB_val *key)
{
    size_t abs = oracle_lower_bound_key_only(txn, dbi, vv, key, 1 /* incl */);
    if (abs < abs_lo) abs = abs_lo;
    if (abs > abs_hi) abs = abs_hi;
    return (uint64_t)(abs - abs_lo);
}

/* -------------------------------------------------------------------------- */

/* ------------------------------ verification against oracle ------------------------------ */

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
        fprintf(stderr, "%s: flags mismatch a=0x%x b=0x%x schema=0x%x\n",
                msg, a->mv_flags, b->mv_flags, schema);
        exit(EXIT_FAILURE);
    }
    if ((schema & MDB_AGG_ENTRIES) && a->mv_agg_entries != b->mv_agg_entries) {
        fprintf(stderr, "%s: entries mismatch a=%" PRIu64 " b=%" PRIu64 "\n",
                msg, a->mv_agg_entries, b->mv_agg_entries);
        exit(EXIT_FAILURE);
    }
    if ((schema & MDB_AGG_KEYS) && a->mv_agg_keys != b->mv_agg_keys) {
        fprintf(stderr, "%s: keys mismatch a=%" PRIu64 " b=%" PRIu64 "\n",
                msg, a->mv_agg_keys, b->mv_agg_keys);
        exit(EXIT_FAILURE);
    }
    if ((schema & MDB_AGG_HASHSUM) && memcmp(a->mv_agg_hashes, b->mv_agg_hashes, MDB_HASH_SIZE) != 0) {
        fprintf(stderr, "%s: hashsum mismatch\n", msg);
        exit(EXIT_FAILURE);
    }
}


/* ------------------------------ randomized oracle cross-checks ------------------------------ */

/* Deterministic FNV-1a 64-bit hash, used to seed the RNG from test labels. */
static uint64_t fnv1a64(const char *s)
{
    uint64_t h = 1469598103934665603ULL;
    while (*s) {
        h ^= (uint8_t)*s++;
        h *= 1099511628211ULL;
    }
    return h ? h : 1ULL;
}

/* xorshift64* - simple, fast deterministic RNG (not cryptographic). */
static uint64_t rng_next_u64(uint64_t *st)
{
    uint64_t x = *st;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *st = x;
    return x * 2685821657736338717ULL;
}

static size_t rng_index(uint64_t *st, size_t n)
{
    if (n == 0) return 0;
    return (size_t)(rng_next_u64(st) % (uint64_t)n);
}

static int rng_bit(uint64_t *st)
{
    return (int)(rng_next_u64(st) & 1u);
}

static void agg_check_basic_invariants(unsigned schema, const MDB_agg *a, const char *msg)
{
    if ((schema & MDB_AGG_ENTRIES) && (schema & MDB_AGG_KEYS)) {
        if (a->mv_agg_keys > a->mv_agg_entries) {
            fprintf(stderr, "%s: invariant violated (keys > entries)\n", msg);
            exit(EXIT_FAILURE);
        }
        if ((a->mv_agg_entries == 0) != (a->mv_agg_keys == 0)) {
            fprintf(stderr, "%s: invariant violated (entries==0 xor keys==0)\n", msg);
            exit(EXIT_FAILURE);
        }
    }
    if ((schema & MDB_AGG_ENTRIES) && (schema & MDB_AGG_HASHSUM) && a->mv_agg_entries == 0) {
        /* hashsum must be zero when no entries are included */
        uint8_t z[MDB_HASH_SIZE];
        memset(z, 0, MDB_HASH_SIZE);
        if (memcmp(a->mv_agg_hashes, z, MDB_HASH_SIZE) != 0) {
            fprintf(stderr, "%s: invariant violated (hashsum nonzero when entries==0)\n", msg);
            exit(EXIT_FAILURE);
        }
    }
}

/* Randomized checks (still deterministic via fixed seeds) to cover odd boundary alignments. */
static void verify_against_oracle_randomized(MDB_txn *txn, MDB_dbi dbi, const char *label,
                                            const RecVec *vv, unsigned schema, unsigned is_dupsort)
{
    uint64_t st = fnv1a64(label) ^ (uint64_t)vv->len * 0x9E3779B97F4A7C15ULL;
    if (!st) st = 1;

    size_t trials = 120;
    if (vv->len < 40) trials = 40;
    if (vv->len > 5000) trials = 80;
#ifdef MDB_DEBUG_AGG_INTEGRITY
    if (trials > 80) trials = 80;
#endif

    /* scratch buffers for "non-exact" data bounds */
    uint8_t dtmp1[2048];
    uint8_t dtmp2[2048];

    for (size_t t = 0; t < trials; t++) {
        const int do_prefix = rng_bit(&st);

        if (do_prefix) {
            MDB_agg got, exp;
            MDB_val k = {0, NULL};
            MDB_val d = {0, NULL};
            MDB_val *dp = NULL;

            /* Pick a key boundary. If empty, use a known-missing key. */
            char kbuf[32];
            if (vv->len == 0 || (rng_next_u64(&st) % 5u) == 0) {
                /* missing-ish key */
                snprintf(kbuf, sizeof(kbuf), "Z%08u", (unsigned)(rng_next_u64(&st) & 0xffffffffu));
                k.mv_data = kbuf;
                k.mv_size = strlen(kbuf);
            } else {
                const Rec *r = &vv->v[rng_index(&st, vv->len)];
                k.mv_data = r->k;
                k.mv_size = r->klen;

                if (is_dupsort && (rng_next_u64(&st) % 3u) != 0) {
                    /* exercise record-order prefix (sometimes non-exact bounds) */
                    if (r->dlen <= sizeof(dtmp1) && (rng_next_u64(&st) & 1u)) {
                        memcpy(dtmp1, r->d, r->dlen);
                        dtmp1[r->dlen - 1] ^= 0x01u; /* very likely non-exact */
                        d.mv_data = dtmp1;
                        d.mv_size = r->dlen;
                    } else {
                        d.mv_data = r->d;
                        d.mv_size = r->dlen;
                    }
                    dp = &d;
                }
            }

            unsigned pflags = rng_bit(&st) ? MDB_AGG_PREFIX_INCL : 0;
            CHECK(mdb_agg_prefix(txn, dbi, &k, dp, pflags, &got), "rand: mdb_agg_prefix");
            oracle_prefix_agg(txn, dbi, vv, &k, dp, pflags, &exp);
            exp.mv_flags = schema;
            agg_require_equal(schema, &got, &exp, "rand prefix == oracle");
            agg_check_basic_invariants(schema, &got, "rand prefix invariants");
        } else {
            MDB_agg got, exp;

            MDB_val lk = {0, NULL}, hk = {0, NULL};
            MDB_val ld = {0, NULL}, hd = {0, NULL};
            MDB_val *lkp = NULL, *hkp = NULL;
            MDB_val *ldp = NULL, *hdp = NULL;

            /* Choose bounds: sometimes open-ended, sometimes swapped (empty ranges). */
            if (vv->len > 0 && (rng_next_u64(&st) % 6u) != 0) {
                const Rec *rl = &vv->v[rng_index(&st, vv->len)];
                lk.mv_data = rl->k; lk.mv_size = rl->klen;
                lkp = &lk;
                if (is_dupsort && (rng_next_u64(&st) & 1u)) {
                    ld.mv_data = rl->d; ld.mv_size = rl->dlen;
                    ldp = &ld;
                }
            } else if ((rng_next_u64(&st) % 10u) == 0 && vv->len > 0) {
                /* exercise "data with NULL key" (current API ignores data when key is NULL) */
                const Rec *r = &vv->v[rng_index(&st, vv->len)];
                if (r->dlen <= sizeof(dtmp1)) {
                    memcpy(dtmp1, r->d, r->dlen);
                    dtmp1[r->dlen - 1] ^= 0x02u;
                    ld.mv_data = dtmp1; ld.mv_size = r->dlen;
                    ldp = &ld;
                }
                lkp = NULL;
            }

            if (vv->len > 0 && (rng_next_u64(&st) % 6u) != 0) {
                const Rec *rh = &vv->v[rng_index(&st, vv->len)];
                hk.mv_data = rh->k; hk.mv_size = rh->klen;
                hkp = &hk;
                if (is_dupsort && (rng_next_u64(&st) & 1u)) {
                    if (rh->dlen <= sizeof(dtmp2) && (rng_next_u64(&st) & 1u)) {
                        memcpy(dtmp2, rh->d, rh->dlen);
                        dtmp2[rh->dlen - 1] ^= 0x04u;
                        hd.mv_data = dtmp2; hd.mv_size = rh->dlen;
                    } else {
                        hd.mv_data = rh->d; hd.mv_size = rh->dlen;
                    }
                    hdp = &hd;
                }
            } else if ((rng_next_u64(&st) % 10u) == 0 && vv->len > 0) {
                /* "data with NULL key" on the high side */
                const Rec *r = &vv->v[rng_index(&st, vv->len)];
                if (r->dlen <= sizeof(dtmp2)) {
                    memcpy(dtmp2, r->d, r->dlen);
                    dtmp2[r->dlen - 1] ^= 0x08u;
                    hd.mv_data = dtmp2; hd.mv_size = r->dlen;
                    hdp = &hd;
                }
                hkp = NULL;
            }

            unsigned rflags = 0;
            if (rng_bit(&st)) rflags |= MDB_RANGE_LOWER_INCL;
            if (rng_bit(&st)) rflags |= MDB_RANGE_UPPER_INCL;

            CHECK(mdb_agg_range(txn, dbi, lkp, ldp, hkp, hdp, rflags, &got), "rand: mdb_agg_range");
            oracle_range_agg_pair(txn, dbi, vv, lkp, ldp, hkp, hdp, rflags, &exp);
            exp.mv_flags = schema;
            agg_require_equal(schema, &got, &exp, "rand range == oracle");
            agg_check_basic_invariants(schema, &got, "rand range invariants");
        }
    }
}


static void verify_against_oracle(MDB_txn *txn, MDB_dbi dbi, const char *label)
{
    unsigned dbi_flags = 0;
    CHECK(mdb_dbi_flags(txn, dbi, &dbi_flags), "mdb_dbi_flags");
    unsigned is_dupsort = 0 != (dbi_flags & MDB_DUPSORT);

    /* Build oracle views. */
    RecVec vv;
    KeyIdxVec kv;
    oracle_build(txn, dbi, &vv);
    oracle_build_keyidx(&vv, &kv);

    /* Expected totals from oracle. */
    uint64_t exp_entries = (uint64_t)vv.len;
    uint64_t exp_keys = (uint64_t)kv.len;
    uint8_t exp_hash[MDB_HASH_SIZE];
    oracle_hashsum_total(&vv, exp_hash);

    unsigned schema = 0;
    CHECK(mdb_agg_info(txn, dbi, &schema), "mdb_agg_info");

    /* Totals */
    {
        MDB_agg got;
        CHECK(mdb_agg_totals(txn, dbi, &got), "mdb_agg_totals");
        agg_expect_flags(label, got.mv_flags, schema);

        if (schema & MDB_AGG_ENTRIES) {
            if (got.mv_agg_entries != exp_entries) {
                fprintf(stderr, "%s: totals entries mismatch got=%" PRIu64 " exp=%" PRIu64 "\n",
                        label, got.mv_agg_entries, exp_entries);
                exit(EXIT_FAILURE);
            }
        }
        if (schema & MDB_AGG_KEYS) {
            if (got.mv_agg_keys != exp_keys) {
                fprintf(stderr, "%s: totals keys mismatch got=%" PRIu64 " exp=%" PRIu64 "\n",
                        label, got.mv_agg_keys, exp_keys);
                exit(EXIT_FAILURE);
            }
        }
        if (schema & MDB_AGG_HASHSUM) {
            if (memcmp(got.mv_agg_hashes, exp_hash, MDB_HASH_SIZE) != 0) {
                fprintf(stderr, "%s: totals hashsum mismatch\n", label);
                exit(EXIT_FAILURE);
            }
        }
    }

    /* Range checks: a few deterministic ranges. */
    {
        MDB_agg got, exp;

        /* Open range should equal totals. */
        CHECK(mdb_agg_range(txn, dbi, NULL, NULL, NULL, NULL, 0, &got), "mdb_agg_range all");
        oracle_range_agg(txn, dbi, &vv, NULL, NULL, 0, &exp);
        agg_expect_flags(label, got.mv_flags, schema);
        if ((schema & MDB_AGG_ENTRIES) && got.mv_agg_entries != exp.mv_agg_entries)
            REQUIRE(0, "range(all) entries mismatch");
        if ((schema & MDB_AGG_KEYS) && got.mv_agg_keys != exp.mv_agg_keys)
            REQUIRE(0, "range(all) keys mismatch");
        if ((schema & MDB_AGG_HASHSUM) && memcmp(got.mv_agg_hashes, exp.mv_agg_hashes, MDB_HASH_SIZE) != 0)
            REQUIRE(0, "range(all) hashsum mismatch");

        if (vv.len > 0) {
            MDB_val low = { vv.v[0].klen, vv.v[0].k };
            MDB_val high = { vv.v[vv.len-1].klen, vv.v[vv.len-1].k };

            /* [first,last] */
            CHECK(mdb_agg_range(txn, dbi, &low, NULL, &high, NULL, MDB_RANGE_LOWER_INCL|MDB_RANGE_UPPER_INCL, &got),
                  "mdb_agg_range [first,last]");
            oracle_range_agg(txn, dbi, &vv, &low, &high, MDB_RANGE_LOWER_INCL|MDB_RANGE_UPPER_INCL, &exp);
            if ((schema & MDB_AGG_ENTRIES) && got.mv_agg_entries != exp.mv_agg_entries)
                REQUIRE(0, "range([first,last]) entries mismatch");
            if ((schema & MDB_AGG_KEYS) && got.mv_agg_keys != exp.mv_agg_keys)
                REQUIRE(0, "range([first,last]) keys mismatch");
            if ((schema & MDB_AGG_HASHSUM) && memcmp(got.mv_agg_hashes, exp.mv_agg_hashes, MDB_HASH_SIZE) != 0)
                REQUIRE(0, "range([first,last]) hashsum mismatch");

            /* (first,last) */
            CHECK(mdb_agg_range(txn, dbi, &low, NULL, &high, NULL, 0, &got), "mdb_agg_range (first,last)");
            oracle_range_agg(txn, dbi, &vv, &low, &high, 0, &exp);
            if ((schema & MDB_AGG_ENTRIES) && got.mv_agg_entries != exp.mv_agg_entries)
                REQUIRE(0, "range((first,last)) entries mismatch");
            if ((schema & MDB_AGG_KEYS) && got.mv_agg_keys != exp.mv_agg_keys)
                REQUIRE(0, "range((first,last)) keys mismatch");
            if ((schema & MDB_AGG_HASHSUM) && memcmp(got.mv_agg_hashes, exp.mv_agg_hashes, MDB_HASH_SIZE) != 0)
                REQUIRE(0, "range((first,last)) hashsum mismatch");
        }
    }

    
    /* Record-order prefix/range checks: mdb_agg_prefix()/mdb_agg_range() */
    {
        MDB_agg got, exp;

        if (vv.len > 0) {
            /* Use an existing record as a deterministic (key,data) boundary. */
            const size_t mid = vv.len / 2u;
            const Rec *mr = &vv.v[mid];
            MDB_val mk = { mr->klen, mr->k };
            MDB_val md = { mr->dlen, mr->d };

            CHECK(mdb_agg_prefix(txn, dbi, &mk, &md, 0, &got), "mdb_agg_prefix excl");
            oracle_prefix_agg(txn, dbi, &vv, &mk, &md, 0, &exp);
            agg_expect_flags(label, got.mv_flags, schema);
            if ((schema & MDB_AGG_ENTRIES) && got.mv_agg_entries != exp.mv_agg_entries)
                REQUIRE(0, "prefix(excl) entries mismatch");
            if ((schema & MDB_AGG_KEYS) && got.mv_agg_keys != exp.mv_agg_keys)
                REQUIRE(0, "prefix(excl) keys mismatch");
            if ((schema & MDB_AGG_HASHSUM) && memcmp(got.mv_agg_hashes, exp.mv_agg_hashes, MDB_HASH_SIZE) != 0)
                REQUIRE(0, "prefix(excl) hashsum mismatch");

            CHECK(mdb_agg_prefix(txn, dbi, &mk, &md, MDB_AGG_PREFIX_INCL, &got), "mdb_agg_prefix incl");
            oracle_prefix_agg(txn, dbi, &vv, &mk, &md, MDB_AGG_PREFIX_INCL, &exp);
            agg_expect_flags(label, got.mv_flags, schema);
            if ((schema & MDB_AGG_ENTRIES) && got.mv_agg_entries != exp.mv_agg_entries)
                REQUIRE(0, "prefix(incl) entries mismatch");
            if ((schema & MDB_AGG_KEYS) && got.mv_agg_keys != exp.mv_agg_keys)
                REQUIRE(0, "prefix(incl) keys mismatch");
            if ((schema & MDB_AGG_HASHSUM) && memcmp(got.mv_agg_hashes, exp.mv_agg_hashes, MDB_HASH_SIZE) != 0)
                REQUIRE(0, "prefix(incl) hashsum mismatch");

            /* Fallback behavior: prefix(key,NULL) == prefix(key) */
            {
                MDB_agg p2, p1;
                CHECK(mdb_agg_prefix(txn, dbi, &mk, NULL, 0, &p2), "mdb_agg_prefix fallback excl");
                CHECK(mdb_agg_prefix(txn, dbi, &mk, NULL, 0, &p1), "mdb_agg_prefix fallback excl");
                agg_require_equal(schema, &p2, &p1, "prefix fallback(excl) == prefix");

                CHECK(mdb_agg_prefix(txn, dbi, &mk, NULL, MDB_AGG_PREFIX_INCL, &p2), "mdb_agg_prefix fallback incl");
                CHECK(mdb_agg_prefix(txn, dbi, &mk, NULL, MDB_AGG_PREFIX_INCL, &p1), "mdb_agg_prefix fallback incl");
                agg_require_equal(schema, &p2, &p1, "prefix fallback(incl) == prefix");
            }
        }

        if (vv.len > 2) {
            const size_t lo = vv.len / 4u;
            const size_t hi = (vv.len * 3u) / 4u;
            const Rec *lr = &vv.v[lo];
            const Rec *hr = &vv.v[hi];
            MDB_val lk = { lr->klen, lr->k };
            MDB_val ld = { lr->dlen, lr->d };
            MDB_val hk = { hr->klen, hr->k };
            MDB_val hd = { hr->dlen, hr->d };

            unsigned rf = MDB_RANGE_LOWER_INCL | MDB_RANGE_UPPER_INCL;
            CHECK(mdb_agg_range(txn, dbi, &lk, &ld, &hk, &hd, rf, &got), "mdb_agg_range [lo,hi]");
            oracle_range_agg_pair(txn, dbi, &vv, &lk, &ld, &hk, &hd, rf, &exp);
            agg_expect_flags(label, got.mv_flags, schema);
            if ((schema & MDB_AGG_ENTRIES) && got.mv_agg_entries != exp.mv_agg_entries)
                REQUIRE(0, "range([lo,hi]) entries mismatch");
            if ((schema & MDB_AGG_KEYS) && got.mv_agg_keys != exp.mv_agg_keys)
                REQUIRE(0, "range([lo,hi]) keys mismatch");
            if ((schema & MDB_AGG_HASHSUM) && memcmp(got.mv_agg_hashes, exp.mv_agg_hashes, MDB_HASH_SIZE) != 0)
                REQUIRE(0, "range([lo,hi]) hashsum mismatch");

            CHECK(mdb_agg_range(txn, dbi, &lk, &ld, &hk, &hd, 0, &got), "mdb_agg_range (lo,hi)");
            oracle_range_agg_pair(txn, dbi, &vv, &lk, &ld, &hk, &hd, 0, &exp);
            agg_expect_flags(label, got.mv_flags, schema);
            if ((schema & MDB_AGG_ENTRIES) && got.mv_agg_entries != exp.mv_agg_entries)
                REQUIRE(0, "range((lo,hi)) entries mismatch");
            if ((schema & MDB_AGG_KEYS) && got.mv_agg_keys != exp.mv_agg_keys)
                REQUIRE(0, "range((lo,hi)) keys mismatch");
            if ((schema & MDB_AGG_HASHSUM) && memcmp(got.mv_agg_hashes, exp.mv_agg_hashes, MDB_HASH_SIZE) != 0)
                REQUIRE(0, "range((lo,hi)) hashsum mismatch");

            /* Fallback behavior: range(key,NULL,key,NULL) == range(key,key) */
            {
                MDB_agg r2, r1;
                CHECK(mdb_agg_range(txn, dbi, &lk, NULL, &hk, NULL, rf, &r2), "mdb_agg_range fallback");
                CHECK(mdb_agg_range(txn, dbi, &lk, NULL, &hk, NULL, rf, &r1), "mdb_agg_range fallback");
                agg_require_equal(schema, &r2, &r1, "range fallback == range");
            }
        }

        /* If we can find a dup-run, exercise bounds that fall strictly inside it. */
        if (is_dupsort) {
            size_t s = 0, e = 0;
            if (oracle_find_first_duprun(&vv, &s, &e) && (e - s) >= 4u) {
                const Rec *ra = &vv.v[s + 1u];
                const Rec *rb = &vv.v[s + 2u];
                const Rec *rc = &vv.v[s + 3u];

                MDB_val k = { ra->klen, ra->k };
                MDB_val da = { ra->dlen, ra->d };
                MDB_val db = { rb->dlen, rb->d };
                MDB_val dc = { rc->dlen, rc->d };

                unsigned rf = MDB_RANGE_LOWER_INCL | MDB_RANGE_UPPER_INCL;
                CHECK(mdb_agg_range(txn, dbi, &k, &da, &k, &dc, rf, &got), "mdb_agg_range within dupset [a,c]");
                oracle_range_agg_pair(txn, dbi, &vv, &k, &da, &k, &dc, rf, &exp);
                agg_expect_flags(label, got.mv_flags, schema);
                if ((schema & MDB_AGG_ENTRIES) && got.mv_agg_entries != exp.mv_agg_entries)
                    REQUIRE(0, "range(within dupset) entries mismatch");
                if ((schema & MDB_AGG_KEYS) && got.mv_agg_keys != exp.mv_agg_keys)
                    REQUIRE(0, "range(within dupset) keys mismatch");
                if ((schema & MDB_AGG_HASHSUM) && memcmp(got.mv_agg_hashes, exp.mv_agg_hashes, MDB_HASH_SIZE) != 0)
                    REQUIRE(0, "range(within dupset) hashsum mismatch");

                /* Mixed bounds: start inside dupset, end at key boundary (include full key). */
                CHECK(mdb_agg_range(txn, dbi, &k, &db, &k, NULL,
                                    MDB_RANGE_LOWER_INCL | MDB_RANGE_UPPER_INCL, &got),
                      "mdb_agg_range mixed (low data, high key-only)");
                oracle_range_agg_pair(txn, dbi, &vv, &k, &db, &k, NULL,
                                 MDB_RANGE_LOWER_INCL | MDB_RANGE_UPPER_INCL, &exp);
                agg_expect_flags(label, got.mv_flags, schema);
                if ((schema & MDB_AGG_ENTRIES) && got.mv_agg_entries != exp.mv_agg_entries)
                    REQUIRE(0, "range(mixed) entries mismatch");
                if ((schema & MDB_AGG_KEYS) && got.mv_agg_keys != exp.mv_agg_keys)
                    REQUIRE(0, "range(mixed) keys mismatch");
                if ((schema & MDB_AGG_HASHSUM) && memcmp(got.mv_agg_hashes, exp.mv_agg_hashes, MDB_HASH_SIZE) != 0)
                    REQUIRE(0, "range(mixed) hashsum mismatch");
            }
        }
    }

/* Rank/select checks - entry weight */
    if (schema & MDB_AGG_ENTRIES) {
        uint64_t ranks[6];
        size_t nr = 0;
        ranks[nr++] = 0;
        if (vv.len > 1) ranks[nr++] = 1;
        if (vv.len > 2) ranks[nr++] = (uint64_t)(vv.len / 2);
        if (vv.len > 0) ranks[nr++] = (uint64_t)(vv.len - 1);

        for (size_t i = 0; i < nr; i++) {
            uint64_t r = ranks[i];
            if (vv.len == 0) {
                MDB_val k = {0, NULL}, d = {0, NULL};
                uint64_t di = UINT64_MAX;
                int rc = mdb_agg_select(txn, dbi, MDB_AGG_WEIGHT_ENTRIES, 0, &k, &d, &di);
                expect_rc(rc, MDB_NOTFOUND, "select(entries) on empty");
                break;
            }
            const Rec *or = &vv.v[r];

            MDB_val k = {0, NULL}, d = {0, NULL};
            uint64_t di = UINT64_MAX;
            CHECK(mdb_agg_select(txn, dbi, MDB_AGG_WEIGHT_ENTRIES, r, &k, &d, &di), "mdb_agg_select(entries)");
            if (k.mv_size != or->klen || memcmp(k.mv_data, or->k, or->klen) != 0 ||
                d.mv_size != or->dlen || memcmp(d.mv_data, or->d, or->dlen) != 0) {
                fprintf(stderr, "%s: select(entries,%" PRIu64 ") mismatch\n", label, r);
                exit(EXIT_FAILURE);
            }

            /* exact rank of the selected record */
            MDB_val qk = { or->klen, or->k };
            MDB_val qd = { .mv_size = is_dupsort ? or->dlen : 0, 
                           .mv_data = is_dupsort ? or->d : NULL };
            uint64_t rr = UINT64_MAX;
            uint64_t rdi = UINT64_MAX;
            CHECK(mdb_agg_rank(txn, dbi, &qk, &qd, MDB_AGG_WEIGHT_ENTRIES, MDB_AGG_RANK_EXACT, &rr, &rdi),
                  "mdb_agg_rank(entries exact)");
            if (rr != r) {
                fprintf(stderr, "%s: rank(entries exact) expected=%" PRIu64 " got=%" PRIu64 "\n", label, r, rr);
                exit(EXIT_FAILURE);
            }
            if (dbi_flags & MDB_DUPSORT) {
                uint64_t exp_di = oracle_dup_index(&vv, (size_t)r);
                if (rdi != exp_di) {
                    fprintf(stderr, "%s: dup_index mismatch for r=%" PRIu64 " exp=%" PRIu64 " got=%" PRIu64 "\n",
                            label, r, exp_di, rdi);
                    exit(EXIT_FAILURE);
                }
            } else {
                if (rdi != 0) {
                    fprintf(stderr, "%s: dup_index expected 0 for non-dupsort\n", label);
                    exit(EXIT_FAILURE);
                }
            }
        }

        /* set_range(min): smallest valid key should locate first record */
        {
            uint8_t minb = 0;
            MDB_val qk = { 1, &minb };
            uint8_t mindbuf[MDB_HASH_SIZE];
            MDB_val qd = { 0, NULL };
            if ((dbi_flags & MDB_DUPSORT) && (dbi_flags & MDB_DUPFIXED) && vv.len > 0) {
                /* DUPFIXED requires a correctly-sized data value even for set-range. */
                memset(mindbuf, 0, MDB_HASH_SIZE);
                qd.mv_size = vv.v[0].dlen; /* should equal MDB_HASH_SIZE */
                qd.mv_data = mindbuf;
            }
            uint64_t rr = UINT64_MAX;
            uint64_t rdi = UINT64_MAX;
            int rc = mdb_agg_rank(txn, dbi, &qk, &qd, MDB_AGG_WEIGHT_ENTRIES, MDB_AGG_RANK_SET_RANGE, &rr, &rdi);
            if (vv.len == 0) {
                expect_rc(rc, MDB_NOTFOUND, "set_range(entries,min) on empty");
            } else {
                CHECK(rc, "mdb_agg_rank(entries set_range(min))");
                if (rr != 0) {
                    fprintf(stderr, "%s: set_range(entries,min) expected 0 got=%" PRIu64 "\n", label, rr);
                    exit(EXIT_FAILURE);
                }
                const Rec *or0 = &vv.v[0];
                if (qk.mv_size != or0->klen || memcmp(qk.mv_data, or0->k, or0->klen) != 0)
                    REQUIRE(0, "set_range(entries,min) did not update key");
                if (qd.mv_size != or0->dlen || memcmp(qd.mv_data, or0->d, or0->dlen) != 0)
                    REQUIRE(0, "set_range(entries,min) did not update data");
            }
        }
    }

    /* Rank/select checks - key weight */
    if (schema & MDB_AGG_KEYS) {
        uint64_t ranks[5];
        size_t nr = 0;
        ranks[nr++] = 0;
        if (kv.len > 1) ranks[nr++] = 1;
        if (kv.len > 2) ranks[nr++] = (uint64_t)(kv.len / 2);
        if (kv.len > 0) ranks[nr++] = (uint64_t)(kv.len - 1);

        for (size_t i = 0; i < nr; i++) {
            uint64_t kr = ranks[i];
            if (kv.len == 0) {
                MDB_val k = {0, NULL}, d = {0, NULL};
                uint64_t di = UINT64_MAX;
                int rc = mdb_agg_select(txn, dbi, MDB_AGG_WEIGHT_KEYS, 0, &k, &d, &di);
                expect_rc(rc, MDB_NOTFOUND, "select(keys) on empty");
                break;
            }

            size_t ri = kv.idx[kr];
            const Rec *or = &vv.v[ri];

            MDB_val k = {0, NULL}, d = {0, NULL};
            uint64_t di = UINT64_MAX;
            CHECK(mdb_agg_select(txn, dbi, MDB_AGG_WEIGHT_KEYS, kr, &k, &d, &di), "mdb_agg_select(keys)");
            if (di != 0) {
                fprintf(stderr, "%s: select(keys) dup_index expected 0 got=%" PRIu64 "\n", label, di);
                exit(EXIT_FAILURE);
            }
            if (k.mv_size != or->klen || memcmp(k.mv_data, or->k, or->klen) != 0) {
                fprintf(stderr, "%s: select(keys,%" PRIu64 ") key mismatch\n", label, kr);
                exit(EXIT_FAILURE);
            }
            /* For DUPSORT, key-weight returns first dup; for plain DB it matches entries. */
            if (d.mv_size != or->dlen || memcmp(d.mv_data, or->d, or->dlen) != 0) {
                fprintf(stderr, "%s: select(keys,%" PRIu64 ") data mismatch\n", label, kr);
                exit(EXIT_FAILURE);
            }

            /* exact key-rank */
            MDB_val qk = { or->klen, or->k };
            MDB_val qd = { 0, NULL };
            uint64_t rr = UINT64_MAX;
            uint64_t rdi = UINT64_MAX;
            CHECK(mdb_agg_rank(txn, dbi, &qk, &qd, MDB_AGG_WEIGHT_KEYS, MDB_AGG_RANK_EXACT, &rr, &rdi),
                  "mdb_agg_rank(keys exact)");
            if (rr != kr) {
                fprintf(stderr, "%s: rank(keys exact) expected=%" PRIu64 " got=%" PRIu64 "\n", label, kr, rr);
                exit(EXIT_FAILURE);
            }
            if (rdi != 0) {
                fprintf(stderr, "%s: rank(keys exact) dup_index expected 0 got=%" PRIu64 "\n", label, rdi);
                exit(EXIT_FAILURE);
            }
        }

        /* set_range(min) for key-weight */
        {
            uint8_t minb = 0;
            MDB_val qk = { 1, &minb };
            MDB_val qd = { 0, NULL };
            uint64_t rr = UINT64_MAX;
            uint64_t rdi = UINT64_MAX;
            int rc = mdb_agg_rank(txn, dbi, &qk, &qd, MDB_AGG_WEIGHT_KEYS, MDB_AGG_RANK_SET_RANGE, &rr, &rdi);
            if (kv.len == 0) {
                expect_rc(rc, MDB_NOTFOUND, "set_range(keys,min) on empty");
            } else {
                CHECK(rc, "mdb_agg_rank(keys set_range(min))");
                if (rr != 0) {
                    fprintf(stderr, "%s: set_range(keys,min) expected 0 got=%" PRIu64 "\n", label, rr);
                    exit(EXIT_FAILURE);
                }
                const Rec *or0 = &vv.v[kv.idx[0]];
                if (qk.mv_size != or0->klen || memcmp(qk.mv_data, or0->k, or0->klen) != 0)
                    REQUIRE(0, "set_range(keys,min) did not update key");
                if (qd.mv_size != or0->dlen || memcmp(qd.mv_data, or0->d, or0->dlen) != 0)
                    REQUIRE(0, "set_range(keys,min) did not update data");
            }
        }
    }

    /* Randomized oracle cross-checks (deterministic by label seed). */
    verify_against_oracle_randomized(txn, dbi, label, &vv, schema, is_dupsort);

    recvec_free(&vv);
    keyidx_free(&kv);
}

/* ------------------------------ test cases ------------------------------ */

static void test_plain_rightsplit(void)
{
    char dir[256];
    make_test_dir(dir, sizeof(dir), "t01_plain_right");
    MDB_env *env = open_env_dir(dir, 8, 64u * 1024u * 1024u, 1);

    MDB_txn *txn = NULL;
    MDB_dbi dbi;

    CHECK(mdb_txn_begin(env, NULL, 0, &txn), "txn_begin");
    CHECK(mdb_dbi_open(txn, "plain", MDB_CREATE | AGG_SCHEMA, &dbi), "dbi_open plain");

    const unsigned N = 700;
    char kbuf[64];
    uint8_t vbuf[MDB_HASH_SIZE + 128];
    for (unsigned i = 0; i < N; i++) {
        make_key(kbuf, sizeof(kbuf), "K", i);
        MDB_val k = { strlen(kbuf), kbuf };
        MDB_val d;
        make_val_with_hash(vbuf, sizeof(vbuf), "V", i, &d);
        CHECK(mdb_put(txn, dbi, &k, &d, 0), "plain right: put");
        DBG_CHECK(txn, dbi, "plain right: dbg agg");
    }

    verify_against_oracle(txn, dbi, "plain right");
    CHECK(mdb_txn_commit(txn), "plain right: commit");

    /* verify in a read txn after reopen */
    CHECK(mdb_txn_begin(env, NULL, MDB_RDONLY, &txn), "plain right: ro begin");
    verify_against_oracle(txn, dbi, "plain right (ro)");
    mdb_txn_abort(txn);

    mdb_dbi_close(env, dbi);
    mdb_env_close(env);
}

static void test_plain_leftsplit(void)
{
    char dir[256];
    make_test_dir(dir, sizeof(dir), "t02_plain_left");
    MDB_env *env = open_env_dir(dir, 8, 64u * 1024u * 1024u, 1);

    MDB_txn *txn = NULL;
    MDB_dbi dbi;

    CHECK(mdb_txn_begin(env, NULL, 0, &txn), "txn_begin");
    CHECK(mdb_dbi_open(txn, "plain", MDB_CREATE | AGG_SCHEMA, &dbi), "dbi_open plain");

    const unsigned N = 700;
    char kbuf[64];
    uint8_t vbuf[MDB_HASH_SIZE + 128];
    for (unsigned i = N; i-- > 0; ) {
        make_key(kbuf, sizeof(kbuf), "K", i);
        MDB_val k = { strlen(kbuf), kbuf };
        MDB_val d;
        make_val_with_hash(vbuf, sizeof(vbuf), "V", i, &d);
        CHECK(mdb_put(txn, dbi, &k, &d, 0), "plain left: put");
        DBG_CHECK(txn, dbi, "plain left: dbg agg");
    }

    verify_against_oracle(txn, dbi, "plain left");
    CHECK(mdb_txn_commit(txn), "plain left: commit");

    CHECK(mdb_txn_begin(env, NULL, MDB_RDONLY, &txn), "plain left: ro begin");
    verify_against_oracle(txn, dbi, "plain left (ro)");
    mdb_txn_abort(txn);

    mdb_dbi_close(env, dbi);
    mdb_env_close(env);
}

static void test_plain_middlefill(void)
{
    char dir[256];
    make_test_dir(dir, sizeof(dir), "t03_plain_middle");
    MDB_env *env = open_env_dir(dir, 8, 64u * 1024u * 1024u, 1);

    MDB_txn *txn = NULL;
    MDB_dbi dbi;

    CHECK(mdb_txn_begin(env, NULL, 0, &txn), "txn_begin");
    CHECK(mdb_dbi_open(txn, "plain", MDB_CREATE | AGG_SCHEMA, &dbi), "dbi_open plain");

    const unsigned N = 900;
    const unsigned mid = N / 2;
    char kbuf[64];
    uint8_t vbuf[MDB_HASH_SIZE + 128];

    for (unsigned step = 0; step < N; step++) {
        unsigned id;
        if (step == 0) {
            id = mid;
        } else {
            unsigned off = (step + 1) / 2;
            id = (step & 1) ? (mid - off) : (mid + off);
        }
        make_key(kbuf, sizeof(kbuf), "K", id);
        MDB_val k = { strlen(kbuf), kbuf };
        MDB_val d;
        make_val_with_hash(vbuf, sizeof(vbuf), "V", id, &d);
        CHECK(mdb_put(txn, dbi, &k, &d, 0), "plain middle: put");
        DBG_CHECK(txn, dbi, "plain middle: dbg agg");
    }

    verify_against_oracle(txn, dbi, "plain middle");
    CHECK(mdb_txn_commit(txn), "plain middle: commit");

    CHECK(mdb_txn_begin(env, NULL, MDB_RDONLY, &txn), "plain middle: ro begin");
    verify_against_oracle(txn, dbi, "plain middle (ro)");
    mdb_txn_abort(txn);

    mdb_dbi_close(env, dbi);
    mdb_env_close(env);
}

static void test_plain_delete_middle(void)
{
    char dir[256];
    make_test_dir(dir, sizeof(dir), "t04_plain_delmid");
    MDB_env *env = open_env_dir(dir, 8, 96u * 1024u * 1024u, 1);

    MDB_txn *txn = NULL;
    MDB_dbi dbi;

    CHECK(mdb_txn_begin(env, NULL, 0, &txn), "txn_begin");
    CHECK(mdb_dbi_open(txn, "plain", MDB_CREATE | AGG_SCHEMA, &dbi), "dbi_open plain");

    const unsigned N = 900;
    char kbuf[64];
    uint8_t vbuf[MDB_HASH_SIZE + 128];

    for (unsigned i = 0; i < N; i++) {
        make_key(kbuf, sizeof(kbuf), "K", i);
        MDB_val k = { strlen(kbuf), kbuf };
        MDB_val d;
        make_val_with_hash(vbuf, sizeof(vbuf), "V", i, &d);
        CHECK(mdb_put(txn, dbi, &k, &d, 0), "plain delmid: put");
        DBG_CHECK(txn, dbi, "plain delmid: dbg agg");
    }

    /* delete middle third */
    unsigned lo = N / 3;
    unsigned hi = (2 * N) / 3;
    for (unsigned i = lo; i < hi; i++) {
        make_key(kbuf, sizeof(kbuf), "K", i);
        MDB_val k = { strlen(kbuf), kbuf };
        int rc = mdb_del(txn, dbi, &k, NULL);
        CHECK(rc, "plain delmid: del");
        DBG_CHECK(txn, dbi, "plain delmid: dbg agg");
    }

    verify_against_oracle(txn, dbi, "plain delmid");
    CHECK(mdb_txn_commit(txn), "plain delmid: commit");

    CHECK(mdb_txn_begin(env, NULL, MDB_RDONLY, &txn), "plain delmid: ro begin");
    verify_against_oracle(txn, dbi, "plain delmid (ro)");
    mdb_txn_abort(txn);

    mdb_dbi_close(env, dbi);
    mdb_env_close(env);
}

static void test_plain_root_collapse(void)
{
    char dir[256];
    make_test_dir(dir, sizeof(dir), "t05_plain_collapse");
    MDB_env *env = open_env_dir(dir, 8, 128u * 1024u * 1024u, 1);

    MDB_txn *txn = NULL;
    MDB_dbi dbi;

    CHECK(mdb_txn_begin(env, NULL, 0, &txn), "txn_begin");
    CHECK(mdb_dbi_open(txn, "plain", MDB_CREATE | AGG_SCHEMA, &dbi), "dbi_open plain");

    const unsigned N = 1400;
    char kbuf[64];
    uint8_t vbuf[MDB_HASH_SIZE + 128];

    for (unsigned i = 0; i < N; i++) {
        make_key(kbuf, sizeof(kbuf), "K", i);
        MDB_val k = { strlen(kbuf), kbuf };
        MDB_val d;
        make_val_with_hash(vbuf, sizeof(vbuf), "V", i, &d);
        CHECK(mdb_put(txn, dbi, &k, &d, 0), "plain collapse: put");
        DBG_CHECK(txn, dbi, "plain collapse: dbg agg");
    }

    /* delete almost everything to force collapse */
    for (unsigned i = 2; i < N; i++) {
        make_key(kbuf, sizeof(kbuf), "K", i);
        MDB_val k = { strlen(kbuf), kbuf };
        CHECK(mdb_del(txn, dbi, &k, NULL), "plain collapse: del");
        DBG_CHECK(txn, dbi, "plain collapse: dbg agg");
    }

    verify_against_oracle(txn, dbi, "plain collapse");
    CHECK(mdb_txn_commit(txn), "plain collapse: commit");

    CHECK(mdb_txn_begin(env, NULL, MDB_RDONLY, &txn), "plain collapse: ro begin");
    verify_against_oracle(txn, dbi, "plain collapse (ro)");
    mdb_txn_abort(txn);

    mdb_dbi_close(env, dbi);
    mdb_env_close(env);
}

static void test_plain_overwrite(void)
{
    char dir[256];
    make_test_dir(dir, sizeof(dir), "t06_plain_overwrite");
    MDB_env *env = open_env_dir(dir, 8, 64u * 1024u * 1024u, 1);

    MDB_txn *txn = NULL;
    MDB_dbi dbi;

    CHECK(mdb_txn_begin(env, NULL, 0, &txn), "txn_begin");
    CHECK(mdb_dbi_open(txn, "plain", MDB_CREATE | AGG_SCHEMA, &dbi), "dbi_open plain");

    const unsigned N = 200;
    char kbuf[64];
    uint8_t vbuf1[MDB_HASH_SIZE + 128];
    uint8_t vbuf2[MDB_HASH_SIZE + 128];

    for (unsigned i = 0; i < N; i++) {
        make_key(kbuf, sizeof(kbuf), "K", i);
        MDB_val k = { strlen(kbuf), kbuf };
        MDB_val d;
        make_val_with_hash(vbuf1, sizeof(vbuf1), "V", i, &d);
        CHECK(mdb_put(txn, dbi, &k, &d, 0), "plain overwrite: put");
        DBG_CHECK(txn, dbi, "plain overwrite: dbg agg");
    }

    /* Overwrite values for a subset: should keep entries/keys, change hashsum. */
    for (unsigned i = 0; i < N; i += 3) {
        make_key(kbuf, sizeof(kbuf), "K", i);
        MDB_val k = { strlen(kbuf), kbuf };
        MDB_val d;
        make_val_with_hash(vbuf2, sizeof(vbuf2), "W", i ^ 0x55u, &d);
        CHECK(mdb_put(txn, dbi, &k, &d, 0), "plain overwrite: overwrite");
        DBG_CHECK(txn, dbi, "plain overwrite: dbg agg");
    }

    verify_against_oracle(txn, dbi, "plain overwrite");
    CHECK(mdb_txn_commit(txn), "plain overwrite: commit");

    CHECK(mdb_txn_begin(env, NULL, MDB_RDONLY, &txn), "plain overwrite: ro begin");
    verify_against_oracle(txn, dbi, "plain overwrite (ro)");
    mdb_txn_abort(txn);

    mdb_dbi_close(env, dbi);
    mdb_env_close(env);
}

static void test_plain_overflow_values_hashsum(void)
{
    char dir[256];
    make_test_dir(dir, sizeof(dir), "t14_plain_overflow");
    MDB_env *env = open_env_dir(dir, 8, 256u * 1024u * 1024u, 1);

    MDB_txn *txn = NULL;
    MDB_dbi dbi;

    CHECK(mdb_txn_begin(env, NULL, 0, &txn), "overflow: begin");
    CHECK(mdb_dbi_open(txn, "plain", MDB_CREATE | AGG_SCHEMA, &dbi), "overflow: dbi_open");

    const unsigned N = 180;
    const size_t EXTRA = 8192u; /* strong overflow on 4k/8k pages */
    char kbuf[64];
    uint8_t vbuf[MDB_HASH_SIZE + 8192u];

    for (unsigned i = 0; i < N; i++) {
        make_key(kbuf, sizeof(kbuf), "K", i);
        MDB_val k = { strlen(kbuf), kbuf };
        MDB_val d;
        make_big_val_with_hash(vbuf, sizeof(vbuf), i, EXTRA, &d);
        CHECK(mdb_put(txn, dbi, &k, &d, 0), "overflow: put");
        DBG_CHECK(txn, dbi, "overflow: dbg agg");
    }

    /* Overwrite some values (hashsum should update correctly). */
    for (unsigned i = 0; i < N; i += 7) {
        make_key(kbuf, sizeof(kbuf), "K", i);
        MDB_val k = { strlen(kbuf), kbuf };
        MDB_val d;
        make_big_val_with_hash(vbuf, sizeof(vbuf), i ^ 0xA5A5u, EXTRA, &d);
        CHECK(mdb_put(txn, dbi, &k, &d, 0), "overflow: overwrite");
        DBG_CHECK(txn, dbi, "overflow: dbg agg");
    }

    /* Delete a middle band (encourages merges with overflow nodes present). */
    unsigned lo = N / 3u;
    unsigned hi = (2u * N) / 3u;
    for (unsigned i = lo; i < hi; i++) {
        make_key(kbuf, sizeof(kbuf), "K", i);
        MDB_val k = { strlen(kbuf), kbuf };
        CHECK(mdb_del(txn, dbi, &k, NULL), "overflow: del");
        DBG_CHECK(txn, dbi, "overflow: dbg agg");
    }

    verify_against_oracle(txn, dbi, "plain overflow");
    CHECK(mdb_txn_commit(txn), "overflow: commit");

    CHECK(mdb_txn_begin(env, NULL, MDB_RDONLY, &txn), "overflow: ro begin");
    verify_against_oracle(txn, dbi, "plain overflow (ro)");
    mdb_txn_abort(txn);

    mdb_dbi_close(env, dbi);
    mdb_env_close(env);
}



/* ------------------------------ structural stress additions (ver39) ------------------------------ */

static void test_plain_deep_split_merge_oscillation(void)
{
    char dir[256];
    make_test_dir(dir, sizeof(dir), "t07_plain_deep_osc");
    MDB_env *env = open_env_dir(dir, 16, 512u * 1024u * 1024u, 1);

    MDB_txn *txn = NULL;
    MDB_dbi dbi;

    CHECK(mdb_txn_begin(env, NULL, 0, &txn), "plain deep osc: begin");
    CHECK(mdb_dbi_open(txn, "plain", MDB_CREATE | AGG_SCHEMA, &dbi), "plain deep osc: open");

#ifdef MDB_DEBUG_AGG_INTEGRITY
    const unsigned N = 1600;
    const unsigned CYCLES = 2;
#else
    const unsigned N = 6500;
    const unsigned CYCLES = 3;
#endif
    const size_t EXTRA_BASE = 760u; /* non-overflow but reduces fanout */

    char kbuf[64];
    uint8_t vbuf[MDB_HASH_SIZE + 1024u];

    /* Build a deeper tree quickly (large values => fewer nodes per page). */
    for (unsigned i = 0; i < N; i++) {
        make_key(kbuf, sizeof(kbuf), "K", i);
        MDB_val k = { strlen(kbuf), kbuf };
        MDB_val d;
        make_big_val_with_hash(vbuf, sizeof(vbuf), i, EXTRA_BASE, &d);
        CHECK(mdb_put(txn, dbi, &k, &d, 0), "plain deep osc: put build");
        if ((i & 63u) == 0u)
            DBG_CHECK(txn, dbi, "plain deep osc: dbg build");
    }
    DBG_CHECK(txn, dbi, "plain deep osc: dbg build end");

    /* Repeatedly delete patterns that force rebalancing/merges, then reinsert into holes
     * to trigger new splits. This exercises split/merge interplay across levels.
     */
    for (unsigned cyc = 0; cyc < CYCLES; cyc++) {
        /* Distributed delete: remove alternating keys. */
        for (unsigned i = 1u + (cyc & 1u); i < N; i += 2u) {
            make_key(kbuf, sizeof(kbuf), "K", i);
            MDB_val k = { strlen(kbuf), kbuf };
            (void)mdb_del(txn, dbi, &k, NULL);
            if ((i & 127u) == 0u)
                DBG_CHECK(txn, dbi, "plain deep osc: dbg del alt");
        }

        /* Left-edge delete: encourages mdb_update_key / leftmost merges. */
        unsigned left_hi = N / 10u;
        for (unsigned i = 0; i < left_hi; i += 3u) {
            make_key(kbuf, sizeof(kbuf), "K", i);
            MDB_val k = { strlen(kbuf), kbuf };
            (void)mdb_del(txn, dbi, &k, NULL);
            if ((i & 127u) == 0u)
                DBG_CHECK(txn, dbi, "plain deep osc: dbg del left");
        }

        /* Middle sparse delete to provoke branch cascades. */
        unsigned lo = N / 3u;
        unsigned hi = (2u * N) / 3u;
        for (unsigned i = lo + (cyc % 5u); i < hi; i += 5u) {
            make_key(kbuf, sizeof(kbuf), "K", i);
            MDB_val k = { strlen(kbuf), kbuf };
            (void)mdb_del(txn, dbi, &k, NULL);
            if ((i & 255u) == 0u)
                DBG_CHECK(txn, dbi, "plain deep osc: dbg del mid");
        }

        DBG_CHECK(txn, dbi, "plain deep osc: dbg after deletes");

        /* Reinsert into holes with a slightly different payload size (split points shift). */
        const size_t extra = EXTRA_BASE + (size_t)((cyc & 1u) ? 96u : 0u);

        /* Reinstate alternating holes using NOOVERWRITE to avoid overwriting survivors. */
        for (unsigned i = 1u + (cyc & 1u); i < N; i += 2u) {
            make_key(kbuf, sizeof(kbuf), "K", i);
            MDB_val k = { strlen(kbuf), kbuf };
            MDB_val d;
            make_big_val_with_hash(vbuf, sizeof(vbuf), i ^ (cyc * 0x9e37u), extra, &d);
            int rc = mdb_put(txn, dbi, &k, &d, MDB_NOOVERWRITE);
            if (rc != MDB_SUCCESS && rc != MDB_KEYEXIST)
                CHECK(rc, "plain deep osc: reinsert alt");
            if ((i & 127u) == 0u)
                DBG_CHECK(txn, dbi, "plain deep osc: dbg rein alt");
        }

        for (unsigned i = 0; i < left_hi; i += 3u) {
            make_key(kbuf, sizeof(kbuf), "K", i);
            MDB_val k = { strlen(kbuf), kbuf };
            MDB_val d;
            make_big_val_with_hash(vbuf, sizeof(vbuf), i ^ (cyc * 0x7f4au), extra, &d);
            int rc = mdb_put(txn, dbi, &k, &d, MDB_NOOVERWRITE);
            if (rc != MDB_SUCCESS && rc != MDB_KEYEXIST)
                CHECK(rc, "plain deep osc: reinsert left");
        }

        for (unsigned i = lo + (cyc % 5u); i < hi; i += 5u) {
            make_key(kbuf, sizeof(kbuf), "K", i);
            MDB_val k = { strlen(kbuf), kbuf };
            MDB_val d;
            make_big_val_with_hash(vbuf, sizeof(vbuf), i ^ (cyc * 0x1234u), extra, &d);
            int rc = mdb_put(txn, dbi, &k, &d, MDB_NOOVERWRITE);
            if (rc != MDB_SUCCESS && rc != MDB_KEYEXIST)
                CHECK(rc, "plain deep osc: reinsert mid");
        }

        DBG_CHECK(txn, dbi, "plain deep osc: dbg after reinserts");
    }

    verify_against_oracle(txn, dbi, "plain deep osc");
    CHECK(mdb_txn_commit(txn), "plain deep osc: commit");

    CHECK(mdb_txn_begin(env, NULL, MDB_RDONLY, &txn), "plain deep osc: ro begin");
    verify_against_oracle(txn, dbi, "plain deep osc (ro)");
    mdb_txn_abort(txn);

    mdb_dbi_close(env, dbi);
    mdb_env_close(env);
}

static void test_plain_merge_update_key_growth_split(void)
{
    /* Target: exercise mdb_page_merge() path that deletes node index 0 in a branch page,
     * triggering mdb_update_key() with a *larger* replacement key (high chance of parent split).
     *
     * Strategy: create three ordered key ranges A < B < C. C keys are very long, A/B short.
     * Delete the entire B band to force merges in the interior. When the B subtree disappears,
     * the first separator key in some parent levels can jump from short (B...) to long (C...),
     * stressing update_key + split interactions during delete/rebalance.
     */
    char dir[256];
    make_test_dir(dir, sizeof(dir), "t08_plain_merge_updatekey");
    MDB_env *env = open_env_dir(dir, 16, 768u * 1024u * 1024u, 1);

    MDB_txn *txn = NULL;
    MDB_dbi dbi;

    CHECK(mdb_txn_begin(env, NULL, 0, &txn), "plain merge updatekey: begin");
    CHECK(mdb_dbi_open(txn, "plain", MDB_CREATE | AGG_SCHEMA, &dbi), "plain merge updatekey: open");

#ifdef MDB_DEBUG_AGG_INTEGRITY
    const unsigned NA = 700, NB = 700, NC = 700;
#else
    const unsigned NA = 2200, NB = 2200, NC = 2200;
#endif
    const size_t PADLEN = 380u;
    const size_t EXTRA = 720u;

    char kbuf[1024];
    uint8_t vbuf[MDB_HASH_SIZE + 1024u];

    /* Insert A short keys */
    for (unsigned i = 0; i < NA; i++) {
        make_key(kbuf, sizeof(kbuf), "A", i);
        MDB_val k = { strlen(kbuf), kbuf };
        MDB_val d;
        make_big_val_with_hash(vbuf, sizeof(vbuf), i, EXTRA, &d);
        CHECK(mdb_put(txn, dbi, &k, &d, 0), "plain merge updatekey: put A");
        if ((i & 127u) == 0u) DBG_CHECK(txn, dbi, "plain merge updatekey: dbg A");
    }

    /* Insert B short keys (the band to delete) */
    for (unsigned i = 0; i < NB; i++) {
        make_key(kbuf, sizeof(kbuf), "B", i);
        MDB_val k = { strlen(kbuf), kbuf };
        MDB_val d;
        make_big_val_with_hash(vbuf, sizeof(vbuf), 0x100000u ^ i, EXTRA, &d);
        CHECK(mdb_put(txn, dbi, &k, &d, 0), "plain merge updatekey: put B");
        if ((i & 127u) == 0u) DBG_CHECK(txn, dbi, "plain merge updatekey: dbg B");
    }

    /* Insert C long keys (pressure parent separator sizes) */
    for (unsigned i = 0; i < NC; i++) {
        make_key_padded(kbuf, sizeof(kbuf), "C", i, PADLEN, (char)('a' + (i % 23u)));
        MDB_val k = { strlen(kbuf), kbuf };
        MDB_val d;
        make_big_val_with_hash(vbuf, sizeof(vbuf), 0x200000u ^ i, EXTRA, &d);
        CHECK(mdb_put(txn, dbi, &k, &d, 0), "plain merge updatekey: put C");
        if ((i & 63u) == 0u) DBG_CHECK(txn, dbi, "plain merge updatekey: dbg C");
    }

    DBG_CHECK(txn, dbi, "plain merge updatekey: dbg build end");

    /* Delete entire B band (forces interior merges and key-updates). */
    for (unsigned i = 0; i < NB; i++) {
        make_key(kbuf, sizeof(kbuf), "B", i);
        MDB_val k = { strlen(kbuf), kbuf };
        (void)mdb_del(txn, dbi, &k, NULL);
        if ((i & 127u) == 0u) DBG_CHECK(txn, dbi, "plain merge updatekey: dbg del B");
    }

    /* Also delete most of A (leave a small anchor range) to encourage cascading rebalances. */
    for (unsigned i = 20; i < NA; i++) {
        make_key(kbuf, sizeof(kbuf), "A", i);
        MDB_val k = { strlen(kbuf), kbuf };
        (void)mdb_del(txn, dbi, &k, NULL);
        if ((i & 255u) == 0u) DBG_CHECK(txn, dbi, "plain merge updatekey: dbg del A");
    }

    DBG_CHECK(txn, dbi, "plain merge updatekey: dbg after deletes");

    verify_against_oracle(txn, dbi, "plain merge updatekey");
    CHECK(mdb_txn_commit(txn), "plain merge updatekey: commit");

    CHECK(mdb_txn_begin(env, NULL, MDB_RDONLY, &txn), "plain merge updatekey: ro begin");
    verify_against_oracle(txn, dbi, "plain merge updatekey (ro)");
    mdb_txn_abort(txn);

    mdb_dbi_close(env, dbi);
    mdb_env_close(env);
}

static void test_dupsort_main_merge_update_key_growth_split(void)
{
    /* Same idea as test_plain_merge_update_key_growth_split, but for a DUPSORT DBI.
     * We insert multiple duplicates per key to ensure entries!=keys semantics are active.
     */
    char dir[256];
    make_test_dir(dir, sizeof(dir), "t09_dups_merge_updatekey");
    MDB_env *env = open_env_dir(dir, 16, 768u * 1024u * 1024u, 1);

    MDB_txn *txn = NULL;
    MDB_dbi dbi;

    CHECK(mdb_txn_begin(env, NULL, 0, &txn), "dups merge updatekey: begin");
    CHECK(mdb_dbi_open(txn, "dups", MDB_CREATE | MDB_DUPSORT | AGG_SCHEMA, &dbi), "dups merge updatekey: open");

#ifdef MDB_DEBUG_AGG_INTEGRITY
    const unsigned NA = 500, NB = 500, NC = 500;
#else
    const unsigned NA = 1600, NB = 1600, NC = 1600;
#endif
    const unsigned NDUPS = 3;
    const size_t PADLEN = 360u;
    /* For MDB_DUPSORT, each duplicate data item must be small enough to be stored
     * inside the dupset (it can't be an overflow item). Keep value size well below
     * MDB_MAXKEYSIZE.
     */
#define MDB_MAXKEYSIZE 511
    const size_t EXTRA = (MDB_MAXKEYSIZE > (MDB_HASH_SIZE + 80u)
                          ? (size_t)MDB_MAXKEYSIZE - (size_t)MDB_HASH_SIZE - 80u
                          : 256u);

    char kbuf[1024];
    uint8_t vbuf[MDB_HASH_SIZE + 1024u];

    /* A */
    for (unsigned i = 0; i < NA; i++) {
        make_key(kbuf, sizeof(kbuf), "A", i);
        MDB_val k = { strlen(kbuf), kbuf };
        for (unsigned j = 0; j < NDUPS; j++) {
            MDB_val d;
            make_big_val_with_hash(vbuf, sizeof(vbuf), (i * 10u) + j, EXTRA, &d);
            CHECK(mdb_put(txn, dbi, &k, &d, 0), "dups merge updatekey: put A");
        }
        if ((i & 127u) == 0u) DBG_CHECK(txn, dbi, "dups merge updatekey: dbg A");
    }

    /* B band */
    for (unsigned i = 0; i < NB; i++) {
        make_key(kbuf, sizeof(kbuf), "B", i);
        MDB_val k = { strlen(kbuf), kbuf };
        for (unsigned j = 0; j < NDUPS; j++) {
            MDB_val d;
            make_big_val_with_hash(vbuf, sizeof(vbuf), 0x100000u + (i * 10u) + j, EXTRA, &d);
            CHECK(mdb_put(txn, dbi, &k, &d, 0), "dups merge updatekey: put B");
        }
        if ((i & 127u) == 0u) DBG_CHECK(txn, dbi, "dups merge updatekey: dbg B");
    }

    /* C long keys */
    for (unsigned i = 0; i < NC; i++) {
        make_key_padded(kbuf, sizeof(kbuf), "C", i, PADLEN, (char)('k' + (i % 19u)));
        MDB_val k = { strlen(kbuf), kbuf };
        for (unsigned j = 0; j < NDUPS; j++) {
            MDB_val d;
            make_big_val_with_hash(vbuf, sizeof(vbuf), 0x200000u + (i * 10u) + j, EXTRA, &d);
            CHECK(mdb_put(txn, dbi, &k, &d, 0), "dups merge updatekey: put C");
        }
        if ((i & 63u) == 0u) DBG_CHECK(txn, dbi, "dups merge updatekey: dbg C");
    }

    DBG_CHECK(txn, dbi, "dups merge updatekey: dbg build end");

    /* Delete entire B band (all dups). */
    for (unsigned i = 0; i < NB; i++) {
        make_key(kbuf, sizeof(kbuf), "B", i);
        MDB_val k = { strlen(kbuf), kbuf };
        (void)mdb_del(txn, dbi, &k, NULL);
        if ((i & 127u) == 0u) DBG_CHECK(txn, dbi, "dups merge updatekey: dbg del B");
    }

    /* Delete most of A to increase rebalance pressure near the front. */
    for (unsigned i = 10; i < NA; i++) {
        make_key(kbuf, sizeof(kbuf), "A", i);
        MDB_val k = { strlen(kbuf), kbuf };
        (void)mdb_del(txn, dbi, &k, NULL);
        if ((i & 255u) == 0u) DBG_CHECK(txn, dbi, "dups merge updatekey: dbg del A");
    }

    DBG_CHECK(txn, dbi, "dups merge updatekey: dbg after deletes");

    verify_against_oracle(txn, dbi, "dups merge updatekey");
    CHECK(mdb_txn_commit(txn), "dups merge updatekey: commit");

    CHECK(mdb_txn_begin(env, NULL, MDB_RDONLY, &txn), "dups merge updatekey: ro begin");
    verify_against_oracle(txn, dbi, "dups merge updatekey (ro)");
    mdb_txn_abort(txn);

    mdb_dbi_close(env, dbi);
    mdb_env_close(env);
}


static void test_regress_del_bigdata_same_txn_after_freelist_loaded(void)
{
    /* Regression for: freeing overflow pages before del0 reads them for hashsum/agg.
     *
     * Repro strategy:
     *  1) Create a small freelist record by inserting and then deleting a small overflow value,
     *     committing the delete so freeDB has at least one record.
     *  2) In a new write txn, insert a *large* overflow value that requires more pages than
     *     the freelist can satisfy, forcing allocation past the current file end.
     *     This also forces mdb_page_alloc() to read freeDB and thus create env->me_pghead.
     *  3) Delete that large overflow value in the same txn. In the buggy code, _mdb_cursor_del()
     *     frees the overflow pages before mdb_cursor_del0() reads them for hashsum deltas,
     *     causing mdb_page_get() to fall back to mmap past EOF -> SIGBUS.
     */
    char dir[256];
    make_test_dir(dir, sizeof(dir), "t19_regress_del_bigdata_freelist");
    MDB_env *env = open_env_dir(dir, 8, 192u * 1024u * 1024u, 1);

    MDB_txn *txn = NULL;
    MDB_dbi dbi;

    MDB_val k0, k1, d;
    make_key_string(&k0, "OV0");
    make_key_string(&k1, "OV1");

    /* Txn1: create DB and insert a modest overflow value. */
    {
        uint8_t small_ov[MDB_HASH_SIZE + 8192u];
        CHECK(mdb_txn_begin(env, NULL, 0, &txn), "regress ov: begin1");
        CHECK(mdb_dbi_open(txn, "plain", MDB_CREATE | AGG_SCHEMA, &dbi), "regress ov: dbi_open");
        make_big_val_with_hash(small_ov, sizeof(small_ov), 1u, 8192u, &d);
        CHECK(mdb_put(txn, dbi, &k0, &d, 0), "regress ov: put small overflow");
        CHECK(mdb_txn_commit(txn), "regress ov: commit1");
    }

    /* Txn2: delete it and commit so freeDB has a record and page_alloc will build me_pghead. */
    CHECK(mdb_txn_begin(env, NULL, 0, &txn), "regress ov: begin2");
    CHECK(mdb_del(txn, dbi, &k0, NULL), "regress ov: del small overflow");
    CHECK(mdb_txn_commit(txn), "regress ov: commit2");

    /* Txn3: insert a larger overflow value (forces new allocation past EOF), then delete it
     * in the same txn (triggers the buggy free-before-read behavior). We abort at the end so
     * the file is not extended to cover the newly allocated pages.
     */
    const size_t big_extra = 4096u * 48u; /* ~192 KiB body => many overflow pages */
    uint8_t *big = (uint8_t *)malloc((size_t)MDB_HASH_SIZE + big_extra);
    REQUIRE(big != NULL, "malloc failed for big overflow buffer");

    CHECK(mdb_txn_begin(env, NULL, 0, &txn), "regress ov: begin3");
    make_big_val_with_hash(big, (size_t)MDB_HASH_SIZE + big_extra, 2u, big_extra, &d);
    CHECK(mdb_put(txn, dbi, &k1, &d, 0), "regress ov: put big overflow");
    DBG_CHECK(txn, dbi, "regress ov: dbg after put big overflow");

    CHECK(mdb_del(txn, dbi, &k1, NULL), "regress ov: del big overflow (same txn)");
    DBG_CHECK(txn, dbi, "regress ov: dbg after del big overflow");

    mdb_txn_abort(txn);
    free(big);

    mdb_dbi_close(env, dbi);
    mdb_env_close(env);
}


static void test_dupsort_divergence(void)
{
    char dir[256];
    make_test_dir(dir, sizeof(dir), "t07_dupsort_diverge");
    MDB_env *env = open_env_dir(dir, 8, 64u * 1024u * 1024u, 1);

    MDB_txn *txn = NULL;
    MDB_dbi dbi;
    char kbuf[64];
    uint8_t vbuf[MDB_HASH_SIZE + 128];

    CHECK(mdb_txn_begin(env, NULL, 0, &txn), "txn_begin");
    CHECK(mdb_dbi_open(txn, "dups", MDB_CREATE | MDB_DUPSORT | AGG_SCHEMA, &dbi), "dbi_open dups");

    /* K00000001 has 3 dups, K00000002 has 1 dup, K00000003 has 2 dups. */
    struct { unsigned key; unsigned dups; } spec[] = {{1,3},{2,1},{3,2}};
    for (size_t si = 0; si < sizeof(spec)/sizeof(spec[0]); si++) {
        make_key(kbuf, sizeof(kbuf), "K", spec[si].key);
        MDB_val k = { strlen(kbuf), kbuf };
        for (unsigned j = 0; j < spec[si].dups; j++) {
            MDB_val d;
            make_val_with_hash(vbuf, sizeof(vbuf), "D", (spec[si].key<<8) ^ j, &d);
            CHECK(mdb_put(txn, dbi, &k, &d, 0), "dups diverge: put");
            DBG_CHECK(txn, dbi, "dups diverge: dbg agg");
        }
    }

    verify_against_oracle(txn, dbi, "dups diverge (after inserts)");

    /* Delete one duplicate from key 1, then delete all remaining from key 1 */
    {
        make_key(kbuf, sizeof(kbuf), "K", 1);
        MDB_val k = { strlen(kbuf), kbuf };
        MDB_val d;
        make_val_with_hash(vbuf, sizeof(vbuf), "D", (1u<<8) ^ 1u, &d);
        CHECK(mdb_del(txn, dbi, &k, &d), "dups diverge: del one dup");
        DBG_CHECK(txn, dbi, "dups diverge: dbg agg");

        /* delete the remaining dups for key 1 */
        make_val_with_hash(vbuf, sizeof(vbuf), "D", (1u<<8) ^ 0u, &d);
        CHECK(mdb_del(txn, dbi, &k, &d), "dups diverge: del dup 0");
        DBG_CHECK(txn, dbi, "dups diverge: dbg agg");
        make_val_with_hash(vbuf, sizeof(vbuf), "D", (1u<<8) ^ 2u, &d);
        CHECK(mdb_del(txn, dbi, &k, &d), "dups diverge: del dup 2");
        DBG_CHECK(txn, dbi, "dups diverge: dbg agg");
    }

    verify_against_oracle(txn, dbi, "dups diverge (after deletes)");
    CHECK(mdb_txn_commit(txn), "dups diverge: commit");

    CHECK(mdb_txn_begin(env, NULL, MDB_RDONLY, &txn), "dups diverge: ro begin");
    verify_against_oracle(txn, dbi, "dups diverge (ro)");
    mdb_txn_abort(txn);

    mdb_dbi_close(env, dbi);
    mdb_env_close(env);
}

static void test_dupsort_round_robin(void)
{
    char dir[256];
    make_test_dir(dir, sizeof(dir), "t08_dupsort_roundrobin");
    MDB_env *env = open_env_dir(dir, 8, 96u * 1024u * 1024u, 1);

    MDB_txn *txn = NULL;
    MDB_dbi dbi;

    CHECK(mdb_txn_begin(env, NULL, 0, &txn), "txn_begin");
    CHECK(mdb_dbi_open(txn, "dups", MDB_CREATE | MDB_DUPSORT | AGG_SCHEMA, &dbi), "dbi_open dups");

    const unsigned KEYS = 60;
    const unsigned DUPS = 5;
    char kbuf[64];
    uint8_t vbuf[MDB_HASH_SIZE + 128];

    /* Phase 1: unique keys with one dup */
    for (unsigned i = 0; i < KEYS; i++) {
        make_key(kbuf, sizeof(kbuf), "K", i);
        MDB_val k = { strlen(kbuf), kbuf };
        MDB_val d;
        make_val_with_hash(vbuf, sizeof(vbuf), "D", (i<<8) ^ 0u, &d);
        CHECK(mdb_put(txn, dbi, &k, &d, 0), "dups rr: put base");
        DBG_CHECK(txn, dbi, "dups rr: dbg agg");
    }

    /* Phase 2: round-robin additional duplicates */
    for (unsigned j = 1; j < DUPS; j++) {
        for (unsigned i = 0; i < KEYS; i++) {
            make_key(kbuf, sizeof(kbuf), "K", i);
            MDB_val k = { strlen(kbuf), kbuf };
            MDB_val d;
            make_val_with_hash(vbuf, sizeof(vbuf), "D", (i<<8) ^ j, &d);
            CHECK(mdb_put(txn, dbi, &k, &d, 0), "dups rr: put dup");
            DBG_CHECK(txn, dbi, "dups rr: dbg agg");
        }
    }

    verify_against_oracle(txn, dbi, "dups rr");
    CHECK(mdb_txn_commit(txn), "dups rr: commit");

    CHECK(mdb_txn_begin(env, NULL, MDB_RDONLY, &txn), "dups rr: ro begin");
    verify_against_oracle(txn, dbi, "dups rr (ro)");
    mdb_txn_abort(txn);

    mdb_dbi_close(env, dbi);
    mdb_env_close(env);
}

/* ------------------------------ deep DUPSORT dup-subDB (F_SUBDATA) coverage ------------------------------ */

#ifdef MDB_DEBUG_AGG_INTEGRITY
#define SUBDB_KEYS 300u
#define SUBDB_DUPS 600u
#else
#define SUBDB_KEYS 1500u
#define SUBDB_DUPS 3000u
#endif

static void build_dupsort_large_dupset(MDB_txn *txn, MDB_dbi dbi, unsigned keys, unsigned dups,
                                      unsigned mid_key)
{
    char kbuf[64];
    uint8_t vbuf[MDB_HASH_SIZE + 128];

    /* Phase 1: many keys with one dup each (helps grow main tree). */
    for (unsigned i = 0; i < keys; i++) {
        make_key(kbuf, sizeof(kbuf), "K", i);
        MDB_val k = { strlen(kbuf), kbuf };
        MDB_val d;
        make_val_with_hash(vbuf, sizeof(vbuf), "B", (i<<8) ^ 0u, &d);
        CHECK(mdb_put(txn, dbi, &k, &d, 0), "dups subdb: put base");
        DBG_CHECK(txn, dbi, "dups subdb: dbg agg");
    }

    /* Phase 2: one hot key with a very large dupset (forces conversion to subDB). */
    make_key(kbuf, sizeof(kbuf), "K", mid_key);
    MDB_val k = { strlen(kbuf), kbuf };
    for (unsigned j = 1; j < dups; j++) {
        MDB_val d;
        make_val_with_hash(vbuf, sizeof(vbuf), "D", ((mid_key<<16) ^ j), &d);
        CHECK(mdb_put(txn, dbi, &k, &d, 0), "dups subdb: put dup");
        DBG_CHECK(txn, dbi, "dups subdb: dbg agg");
    }
}

static void test_dupsort_large_dups_subdb_build(void)
{
    char dir[256];
    make_test_dir(dir, sizeof(dir), "t09_dups_subdb_build");
    MDB_env *env = open_env_dir(dir, 16, 256u * 1024u * 1024u, 1);

    MDB_txn *txn = NULL;
    MDB_dbi dbi;

    CHECK(mdb_txn_begin(env, NULL, 0, &txn), "subdb build: begin");
    CHECK(mdb_dbi_open(txn, "dups", MDB_CREATE | MDB_DUPSORT | AGG_SCHEMA, &dbi), "subdb build: dbi_open");

    const unsigned mid = SUBDB_KEYS / 2u;
    build_dupsort_large_dupset(txn, dbi, SUBDB_KEYS, SUBDB_DUPS, mid);

    verify_against_oracle(txn, dbi, "dups subdb build");

    /* Targeted checks inside the large dupset: first/mid/last dup ranks.
     * This helps exercise aggregate traversal inside a deep dup tree.
     */
    {
        char kbuf[64];
        make_key(kbuf, sizeof(kbuf), "K", mid);
        MDB_val km = { strlen(kbuf), kbuf };

        RecVec vv;
        KeyIdxVec kv;
        oracle_build(txn, dbi, &vv);
        oracle_build_keyidx(&vv, &kv);
        size_t s, e;
        find_key_run_or_die(&vv, &km, &s, &e);
        REQUIRE(e > s + 10, "subdb build: dupset too small (did not grow?)");

        /* Verify key-local range aggregation: should count all duplicates under the key and exactly 1 key. */
        {
            unsigned schema = 0;
            CHECK(mdb_agg_info(txn, dbi, &schema), "subdb build: mdb_agg_info");
            MDB_agg got, exp;
            CHECK(mdb_agg_range(txn, dbi, &km, NULL, &km,
                                NULL, MDB_RANGE_LOWER_INCL | MDB_RANGE_UPPER_INCL,
                                &got),
                  "subdb build: mdb_agg_range key-only");
            oracle_range_agg(txn, dbi, &vv, &km, &km,
                             MDB_RANGE_LOWER_INCL | MDB_RANGE_UPPER_INCL,
                             &exp);
            agg_expect_flags("subdb build(range)", got.mv_flags, schema);
            if ((schema & MDB_AGG_ENTRIES) && got.mv_agg_entries != exp.mv_agg_entries)
                REQUIRE(0, "subdb build: key-only range entries mismatch");
            if ((schema & MDB_AGG_KEYS) && got.mv_agg_keys != exp.mv_agg_keys)
                REQUIRE(0, "subdb build: key-only range keys mismatch");
            if ((schema & MDB_AGG_HASHSUM) && memcmp(got.mv_agg_hashes, exp.mv_agg_hashes, MDB_HASH_SIZE) != 0)
                REQUIRE(0, "subdb build: key-only range hashsum mismatch");
        }

        size_t picks[3] = { s, s + (e - s) / 2u, e - 1 };
        for (unsigned pi = 0; pi < 3; pi++) {
            size_t idx = picks[pi];
            const Rec *or = &vv.v[idx];

            MDB_val k = {0, NULL}, d = {0, NULL};
            uint64_t di = UINT64_MAX;
            CHECK(mdb_agg_select(txn, dbi, MDB_AGG_WEIGHT_ENTRIES, (uint64_t)idx, &k, &d, &di),
                  "subdb build: select(entries)");
            REQUIRE(k.mv_size == or->klen && memcmp(k.mv_data, or->k, or->klen) == 0, "subdb build: select key");
            REQUIRE(d.mv_size == or->dlen && memcmp(d.mv_data, or->d, or->dlen) == 0, "subdb build: select data");

            MDB_val qk = { or->klen, or->k };
            MDB_val qd = { or->dlen, or->d };
            uint64_t rr = UINT64_MAX, rdi = UINT64_MAX;
            CHECK(mdb_agg_rank(txn, dbi, &qk, &qd, MDB_AGG_WEIGHT_ENTRIES, MDB_AGG_RANK_EXACT, &rr, &rdi),
                  "subdb build: rank(entries exact)");
            REQUIRE(rr == (uint64_t)idx, "subdb build: rank mismatch");
            REQUIRE(rdi == oracle_dup_index(&vv, idx), "subdb build: dup_index mismatch");
        }

        recvec_free(&vv);
        keyidx_free(&kv);
    }

    CHECK(mdb_txn_commit(txn), "subdb build: commit");

    CHECK(mdb_txn_begin(env, NULL, MDB_RDONLY, &txn), "subdb build: ro begin");
    verify_against_oracle(txn, dbi, "dups subdb build (ro)");
    mdb_txn_abort(txn);

    mdb_dbi_close(env, dbi);
    mdb_env_close(env);
}

static void test_dupsort_subdb_merge_and_collapse(void)
{
    char dir[256];
    make_test_dir(dir, sizeof(dir), "t10_dups_subdb_merge");
    MDB_env *env = open_env_dir(dir, 16, 256u * 1024u * 1024u, 1);

    MDB_txn *txn = NULL;
    MDB_dbi dbi;

    CHECK(mdb_txn_begin(env, NULL, 0, &txn), "subdb merge: begin");
    CHECK(mdb_dbi_open(txn, "dups", MDB_CREATE | MDB_DUPSORT | AGG_SCHEMA, &dbi), "subdb merge: dbi_open");

    const unsigned mid = SUBDB_KEYS / 2u;
    build_dupsort_large_dupset(txn, dbi, SUBDB_KEYS, SUBDB_DUPS, mid);
    verify_against_oracle(txn, dbi, "dups subdb merge (after build)");

    /* Delete a middle band of duplicates to force dup-tree merges/rebalances. */
    {
        char kbuf[64];
        make_key(kbuf, sizeof(kbuf), "K", mid);
        MDB_val k = { strlen(kbuf), kbuf };
        uint8_t vbuf[MDB_HASH_SIZE + 128];

        unsigned lo = SUBDB_DUPS / 3u;
        unsigned hi = (2u * SUBDB_DUPS) / 3u;
        for (unsigned j = lo; j < hi; j++) {
            MDB_val d;
            make_val_with_hash(vbuf, sizeof(vbuf), "D", ((mid<<16) ^ j), &d);
            int rc = mdb_del(txn, dbi, &k, &d);
            CHECK(rc, "subdb merge: del middle dup");
            DBG_CHECK(txn, dbi, "subdb merge: dbg agg");
        }
    }

    verify_against_oracle(txn, dbi, "dups subdb merge (after band delete)");

    /* Delete all remaining duplicates except j==0 (forces further merges/root collapse). */
    {
        char kbuf[64];
        make_key(kbuf, sizeof(kbuf), "K", mid);
        MDB_val k = { strlen(kbuf), kbuf };
        uint8_t vbuf[MDB_HASH_SIZE + 128];

        unsigned lo = SUBDB_DUPS / 3u;
        unsigned hi = (2u * SUBDB_DUPS) / 3u;
        for (unsigned j = 1; j < SUBDB_DUPS; j++) {
            if (j >= lo && j < hi)
                continue; /* already deleted */
            MDB_val d;
            make_val_with_hash(vbuf, sizeof(vbuf), "D", ((mid<<16) ^ j), &d);
            int rc = mdb_del(txn, dbi, &k, &d);
            CHECK(rc, "subdb merge: del remaining dup");
            DBG_CHECK(txn, dbi, "subdb merge: dbg agg");
        }
    }

    verify_against_oracle(txn, dbi, "dups subdb merge (after shrink-to-1)");

    /* Delete last remaining duplicate j==0, removing the key from the main tree. */
    {
        char kbuf[64];
        make_key(kbuf, sizeof(kbuf), "K", mid);
        MDB_val k = { strlen(kbuf), kbuf };
        uint8_t vbuf[MDB_HASH_SIZE + 128];
        MDB_val d;
        make_val_with_hash(vbuf, sizeof(vbuf), "B", (mid<<8) ^ 0u, &d); /* base dup data */
        CHECK(mdb_del(txn, dbi, &k, &d), "subdb merge: del last dup");
        DBG_CHECK(txn, dbi, "subdb merge: dbg agg");
    }

    verify_against_oracle(txn, dbi, "dups subdb merge (after remove key)");
    CHECK(mdb_txn_commit(txn), "subdb merge: commit");

    CHECK(mdb_txn_begin(env, NULL, MDB_RDONLY, &txn), "subdb merge: ro begin");
    verify_against_oracle(txn, dbi, "dups subdb merge (ro)");
    mdb_txn_abort(txn);

    mdb_dbi_close(env, dbi);
    mdb_env_close(env);
}

static void test_dupsort_set_range_within_dups(void)
{
    char dir[256];
    make_test_dir(dir, sizeof(dir), "t11_dups_setrange_dups");
    MDB_env *env = open_env_dir(dir, 16, 256u * 1024u * 1024u, 1);

    MDB_txn *txn = NULL;
    MDB_dbi dbi;

    CHECK(mdb_txn_begin(env, NULL, 0, &txn), "setrange dups: begin");
    CHECK(mdb_dbi_open(txn, "dups", MDB_CREATE | MDB_DUPSORT | AGG_SCHEMA, &dbi), "setrange dups: dbi_open");

    const unsigned mid = SUBDB_KEYS / 2u;
    build_dupsort_large_dupset(txn, dbi, SUBDB_KEYS, SUBDB_DUPS, mid);

    /* Build oracle and pick a SET_RANGE query whose insertion point lands inside the dupset of Kmid. */
    RecVec vv;
    KeyIdxVec kv;
    oracle_build(txn, dbi, &vv);
    oracle_build_keyidx(&vv, &kv);

    char kbuf[64];
    make_key(kbuf, sizeof(kbuf), "K", mid);
    MDB_val km = { strlen(kbuf), kbuf };

    size_t s, e;
    find_key_run_or_die(&vv, &km, &s, &e);
    REQUIRE(e > s + 10, "setrange dups: dupset too small");

    uint8_t qbuf[MDB_HASH_SIZE + 128];
    MDB_val qd;
    size_t expect_idx = (size_t)-1;
    unsigned picked = 0;

    /* Deterministic candidates. We try to find one whose ordering falls inside [s,e). */
    const unsigned cand_list[] = { SUBDB_DUPS + 7u, SUBDB_DUPS + 123u, SUBDB_DUPS + 999u, SUBDB_DUPS + 4096u };
    for (size_t ci = 0; ci < sizeof(cand_list)/sizeof(cand_list[0]); ci++) {
        make_val_with_hash(qbuf, sizeof(qbuf), "Q", ((mid<<16) ^ cand_list[ci]), &qd);
        size_t lb = lower_bound_dups(&vv, s, e, &qd);
        if (lb > s && lb < e) {
            expect_idx = lb;
            picked = 1;
            break;
        }
    }
    if (!picked) {
        /* Fallback: use an existing mid-duplicate (SET_RANGE should land exactly there). */
        size_t mididx = s + (e - s) / 2u;
        const Rec *mr = &vv.v[mididx];
        qd.mv_size = mr->dlen;
        qd.mv_data = mr->d;
        expect_idx = mididx;
    }

    MDB_val qk = km;
    uint64_t rr = UINT64_MAX, rdi = UINT64_MAX;
    CHECK(mdb_agg_rank(txn, dbi, &qk, &qd, MDB_AGG_WEIGHT_ENTRIES, MDB_AGG_RANK_SET_RANGE, &rr, &rdi),
          "setrange dups: mdb_agg_rank(entries set_range)");

    REQUIRE(rr == (uint64_t)expect_idx, "setrange dups: expected rank mismatch");
    const Rec *or = &vv.v[expect_idx];
    REQUIRE(qk.mv_size == or->klen && memcmp(qk.mv_data, or->k, or->klen) == 0, "setrange dups: key update");
    REQUIRE(qd.mv_size == or->dlen && memcmp(qd.mv_data, or->d, or->dlen) == 0, "setrange dups: data update");
    REQUIRE(rdi == oracle_dup_index(&vv, expect_idx), "setrange dups: dup_index mismatch");

    recvec_free(&vv);
    keyidx_free(&kv);

    verify_against_oracle(txn, dbi, "setrange dups");
    CHECK(mdb_txn_commit(txn), "setrange dups: commit");

    CHECK(mdb_txn_begin(env, NULL, MDB_RDONLY, &txn), "setrange dups: ro begin");
    verify_against_oracle(txn, dbi, "setrange dups (ro)");
    mdb_txn_abort(txn);

    mdb_dbi_close(env, dbi);
    mdb_env_close(env);
}


static void prefix_vs_range_check_pair(MDB_txn *txn, MDB_dbi dbi, const char *label,
                                      const MDB_val *qk, const MDB_val *qd);

static void test_dupsort_prefix_range_partial(void)
{
    char dir[256];
    make_test_dir(dir, sizeof(dir), "t12_dups_prefix_range");
    MDB_env *env = open_env_dir(dir, 16, 96u * 1024u * 1024u, 1);

    MDB_txn *txn = NULL;
    MDB_dbi dbi;

    CHECK(mdb_txn_begin(env, NULL, 0, &txn), "prefix/range: begin");
    CHECK(mdb_dbi_open(txn, "dups", MDB_CREATE | MDB_DUPSORT | AGG_SCHEMA, &dbi), "prefix/range: dbi_open");

    char kbuf[64];
    uint8_t vbuf[MDB_HASH_SIZE + 128];

    /* Build a small but structured DB:
     * - K0 has 2 dups
     * - K1 has 7 dups (enough to test partial dup-ranges)
     * - K2 has 1 value
     */
    for (unsigned i = 0; i < 3; i++) {
        make_key(kbuf, sizeof(kbuf), "K", i);
        MDB_val k = { strlen(kbuf), kbuf };
        unsigned ndups = (i == 0) ? 2u : (i == 1) ? 7u : 1u;
        for (unsigned j = 0; j < ndups; j++) {
            MDB_val d;
            make_val_with_hash_salted(vbuf, sizeof(vbuf), "D", (i * 100u) + j, (uint64_t)j, &d);
            CHECK(mdb_put(txn, dbi, &k, &d, 0), "prefix/range: put");
        }
    }

    /* Oracle view in record order. */
    RecVec vv;
    KeyIdxVec kv;
    oracle_build(txn, dbi, &vv);
    oracle_build_keyidx(&vv, &kv);

    /* Locate the dup-run for K1. */
    make_key(kbuf, sizeof(kbuf), "K", 1u);
    MDB_val k1 = { strlen(kbuf), kbuf };
    size_t s, e;
    find_key_run_or_die(&vv, &k1, &s, &e);
    REQUIRE((e - s) >= 7u, "prefix/range: expected K1 dup-run length");

    const Rec *r2 = &vv.v[s + 2u];
    const Rec *r4 = &vv.v[s + 4u];
    const Rec *r6 = &vv.v[s + 6u];

    MDB_val k = { r2->klen, r2->k };
    MDB_val d2 = { r2->dlen, r2->d };
    MDB_val d4 = { r4->dlen, r4->d };
    MDB_val d6 = { r6->dlen, r6->d };

    /* prefix/range equivalence at a boundary inside a dupset */
    prefix_vs_range_check_pair(txn, dbi, "dups K1 mid-dup", &k, &d4);

    /* prefix correctness via oracle (exclusive & inclusive) */
    {
        MDB_agg got, exp;

        CHECK(mdb_agg_prefix(txn, dbi, &k, &d4, 0, &got), "prefix/range: mdb_agg_prefix excl");
        oracle_prefix_agg(txn, dbi, &vv, &k, &d4, 0, &exp);
        agg_require_equal(AGG_SCHEMA, &got, &exp, "prefix(excl) oracle");

        CHECK(mdb_agg_prefix(txn, dbi, &k, &d4, MDB_AGG_PREFIX_INCL, &got), "prefix/range: mdb_agg_prefix incl");
        oracle_prefix_agg(txn, dbi, &vv, &k, &d4, MDB_AGG_PREFIX_INCL, &exp);
        agg_require_equal(AGG_SCHEMA, &got, &exp, "prefix(incl) oracle");
    }

    /* range strictly inside the dupset */
    {
        MDB_agg got, exp;
        unsigned rf = MDB_RANGE_LOWER_INCL | MDB_RANGE_UPPER_INCL;

        CHECK(mdb_agg_range(txn, dbi, &k, &d2, &k, &d6, rf, &got), "prefix/range: range within dupset");
        oracle_range_agg_pair(txn, dbi, &vv, &k, &d2, &k, &d6, rf, &exp);
        agg_require_equal(AGG_SCHEMA, &got, &exp, "range(within dupset) oracle");

        /* Keys touched by a dup-internal range must be 1. */
        REQUIRE(got.mv_agg_keys == 1, "prefix/range: KEYS should be 1 for within-dupset range");
    }

    /* Mixed bounds: low inside dupset, high at key boundary (include all of K1). */
    {
        MDB_agg got, exp;
        unsigned rf = MDB_RANGE_LOWER_INCL | MDB_RANGE_UPPER_INCL;

        CHECK(mdb_agg_range(txn, dbi, &k, &d4, &k, NULL, rf, &got), "prefix/range: range mixed bounds");
        oracle_range_agg_pair(txn, dbi, &vv, &k, &d4, &k, NULL, rf, &exp);
        agg_require_equal(AGG_SCHEMA, &got, &exp, "range(mixed) oracle");
        REQUIRE(got.mv_agg_keys == 1, "prefix/range: KEYS should be 1 for mixed bounds within same key");
    }

    /* Cross-key record-range: from a mid-dup of K1 to the single value of K2. */
    {
        make_key(kbuf, sizeof(kbuf), "K", 2u);
        MDB_val k2 = { strlen(kbuf), kbuf };
        MDB_val d;
        /* Use the exact existing value of K2 (the only record in its dup-run). */
        size_t s2, e2;
        find_key_run_or_die(&vv, &k2, &s2, &e2);
        REQUIRE(e2 == s2 + 1u, "prefix/range: expected K2 single value");
        const Rec *r = &vv.v[s2];
        d.mv_size = r->dlen;
        d.mv_data = r->d;

        MDB_agg got, exp;
        unsigned rf = MDB_RANGE_LOWER_INCL | MDB_RANGE_UPPER_INCL;

        CHECK(mdb_agg_range(txn, dbi, &k, &d4, &k2, &d, rf, &got), "prefix/range: range cross-key");
        oracle_range_agg_pair(txn, dbi, &vv, &k, &d4, &k2, &d, rf, &exp);
        agg_require_equal(AGG_SCHEMA, &got, &exp, "range(cross-key) oracle");
        REQUIRE(got.mv_agg_keys == 2, "prefix/range: KEYS should be 2 for K1..K2 record-range");
    }

    recvec_free(&vv);
    keyidx_free(&kv);

    CHECK(mdb_txn_commit(txn), "prefix/range: commit");
    mdb_dbi_close(env, dbi);
    mdb_env_close(env);
}


static void make_ordered_dup_val_suffix(uint8_t *buf, size_t cap, uint8_t hfill, const char *suffix, MDB_val *out)
{
    size_t slen = strlen(suffix);
    REQUIRE(cap >= (size_t)MDB_HASH_SIZE + slen, "buffer too small for ordered dup value");
    memset(buf, hfill, MDB_HASH_SIZE);
    memcpy(buf + MDB_HASH_SIZE, suffix, slen);
    out->mv_data = buf;
    out->mv_size = (size_t)MDB_HASH_SIZE + slen;
}

static void test_dupsort_prefix_range_cornercases(void)
{
    char dir[256];
    make_test_dir(dir, sizeof(dir), "t12c_dups_prefix_range_micro");
    MDB_env *env = open_env_dir(dir, 16, 64u * 1024u * 1024u, 1);

    MDB_txn *txn = NULL;
    MDB_dbi dbi;

    CHECK(mdb_txn_begin(env, NULL, 0, &txn), "prefix micro: begin");
    CHECK(mdb_dbi_open(txn, "dups", MDB_CREATE | MDB_DUPSORT | AGG_SCHEMA, &dbi), "prefix micro: dbi_open");

    char kbuf[64];
    uint8_t vbuf[MDB_HASH_SIZE + 32];

    /* Small deterministic DB:
     *   K0: single value
     *   K1: 7 duplicates with identical hash-prefix (ordering driven by suffix)
     *   K2: single value with a different hash-prefix
     */
    {
        /* K0 */
        make_key(kbuf, sizeof(kbuf), "K", 0u);
        MDB_val k0 = { strlen(kbuf), kbuf };
        MDB_val d0;
        make_ordered_dup_val_suffix(vbuf, sizeof(vbuf), 0x01, "a000", &d0);
        CHECK(mdb_put(txn, dbi, &k0, &d0, 0), "prefix micro: put K0");
        DBG_CHECK(txn, dbi, "prefix micro: dbg K0");
    }
    {
        /* K1 dupset */
        make_key(kbuf, sizeof(kbuf), "K", 1u);
        MDB_val k1 = { strlen(kbuf), kbuf };
        char sbuf[16];
        for (unsigned j = 0; j < 7u; j++) {
            snprintf(sbuf, sizeof(sbuf), "d%03u", j); /* d000..d006 */
            MDB_val dj;
            make_ordered_dup_val_suffix(vbuf, sizeof(vbuf), 0x11, sbuf, &dj);
            CHECK(mdb_put(txn, dbi, &k1, &dj, 0), "prefix micro: put K1 dup");
            DBG_CHECK(txn, dbi, "prefix micro: dbg K1 dup");
        }
    }
    {
        /* K2 */
        make_key(kbuf, sizeof(kbuf), "K", 2u);
        MDB_val k2 = { strlen(kbuf), kbuf };
        MDB_val d2;
        make_ordered_dup_val_suffix(vbuf, sizeof(vbuf), 0x22, "z000", &d2);
        CHECK(mdb_put(txn, dbi, &k2, &d2, 0), "prefix micro: put K2");
        DBG_CHECK(txn, dbi, "prefix micro: dbg K2");
    }

    /* Oracle view in record order. */
    RecVec vv;
    KeyIdxVec kv;
    oracle_build(txn, dbi, &vv);
    oracle_build_keyidx(&vv, &kv);

    /* Prepare reusable keys. */
    char k1buf[64], k2buf[64], k1abuf[64];
    make_key(k1buf, sizeof(k1buf), "K", 1u);
    MDB_val k1 = { strlen(k1buf), k1buf };
    make_key(k2buf, sizeof(k2buf), "K", 2u);
    MDB_val k2 = { strlen(k2buf), k2buf };
    snprintf(k1abuf, sizeof(k1abuf), "K%08uA", 1u); /* between K1 and K2 */
    MDB_val k1a = { strlen(k1abuf), k1abuf };

    /* Prepare query datas. */
    uint8_t dbuf_between_lo[MDB_HASH_SIZE + 32];
    uint8_t dbuf_between_hi[MDB_HASH_SIZE + 32];
    uint8_t dbuf_before_first[MDB_HASH_SIZE + 32];
    uint8_t dbuf_after_last[MDB_HASH_SIZE + 32];
    uint8_t dbuf_exact_d003[MDB_HASH_SIZE + 32];
    uint8_t dbuf_k2_low[MDB_HASH_SIZE + 32];
    uint8_t dbuf_k2_high[MDB_HASH_SIZE + 32];

    MDB_val d_between_lo, d_between_hi, d_before_first, d_after_last, d_exact_d003, d_k2_low, d_k2_high;
    make_ordered_dup_val_suffix(dbuf_between_lo, sizeof(dbuf_between_lo), 0x11, "d0025", &d_between_lo); /* between d002 and d003 */
    make_ordered_dup_val_suffix(dbuf_between_hi, sizeof(dbuf_between_hi), 0x11, "d0055", &d_between_hi); /* between d005 and d006 */
    make_ordered_dup_val_suffix(dbuf_before_first, sizeof(dbuf_before_first), 0x11, "d-01", &d_before_first); /* before d000 */
    make_ordered_dup_val_suffix(dbuf_after_last, sizeof(dbuf_after_last), 0x11, "d999", &d_after_last); /* after d006 */
    make_ordered_dup_val_suffix(dbuf_exact_d003, sizeof(dbuf_exact_d003), 0x11, "d003", &d_exact_d003); /* existing */
    make_ordered_dup_val_suffix(dbuf_k2_low, sizeof(dbuf_k2_low), 0x22, "y999", &d_k2_low);  /* before z000 */
    make_ordered_dup_val_suffix(dbuf_k2_high, sizeof(dbuf_k2_high), 0x22, "zzzz", &d_k2_high); /* after z000 */

    /* (1) prefix: non-existing data inside dupset => inclusive/exclusive must match. */
    {
        MDB_agg got_ex, got_in, exp;

        CHECK(mdb_agg_prefix(txn, dbi, &k1, &d_between_lo, 0, &got_ex), "prefix micro: prefix excl (between)");
        CHECK(mdb_agg_prefix(txn, dbi, &k1, &d_between_lo, MDB_AGG_PREFIX_INCL, &got_in), "prefix micro: prefix incl (between)");
        oracle_prefix_agg(txn, dbi, &vv, &k1, &d_between_lo, 0, &exp);
        agg_require_equal(AGG_SCHEMA, &got_ex, &exp, "prefix(between,excl) oracle");
        agg_require_equal(AGG_SCHEMA, &got_in, &exp, "prefix(between,incl) oracle");
        agg_require_equal(AGG_SCHEMA, &got_ex, &got_in, "prefix(between) incl==excl");
    }

    /* (2) prefix: data before first dup => equals key-only prefix(K1, excl). */
    {
        MDB_agg got, exp, keyonly;

        CHECK(mdb_agg_prefix(txn, dbi, &k1, &d_before_first, 0, &got), "prefix micro: prefix before-first");
        oracle_prefix_agg(txn, dbi, &vv, &k1, &d_before_first, 0, &exp);
        agg_require_equal(AGG_SCHEMA, &got, &exp, "prefix(before-first) oracle");

        CHECK(mdb_agg_prefix(txn, dbi, &k1, NULL, 0, &keyonly), "prefix micro: key-only prefix(K1)");
        agg_require_equal(AGG_SCHEMA, &got, &keyonly, "prefix(before-first) == prefix(K1)");
    }

    /* (3) prefix: data after last dup => equals key-only prefix(K2, excl) (all records < K2). */
    {
        MDB_agg got, exp, keyonly;

        CHECK(mdb_agg_prefix(txn, dbi, &k1, &d_after_last, 0, &got), "prefix micro: prefix after-last");
        oracle_prefix_agg(txn, dbi, &vv, &k1, &d_after_last, 0, &exp);
        agg_require_equal(AGG_SCHEMA, &got, &exp, "prefix(after-last) oracle");

        CHECK(mdb_agg_prefix(txn, dbi, &k2, NULL, 0, &keyonly), "prefix micro: key-only prefix(K2)");
        agg_require_equal(AGG_SCHEMA, &got, &keyonly, "prefix(after-last) == prefix(K2)");
    }

    /* (4) range: non-existing bounds inside dupset.
     * low=(K1,d0025) high=(K1,d0055) should include d003..d005 (3 records), KEYS==1.
     * inclusive/exclusive do not matter for non-existing endpoints.
     */
    {
        MDB_agg got_in, got_ex, exp;
        unsigned rf_in = MDB_RANGE_LOWER_INCL | MDB_RANGE_UPPER_INCL;
        unsigned rf_ex = 0; /* both bounds exclusive */

        CHECK(mdb_agg_range(txn, dbi, &k1, &d_between_lo, &k1, &d_between_hi, rf_in, &got_in),
              "prefix micro: range in (between..between)");
        CHECK(mdb_agg_range(txn, dbi, &k1, &d_between_lo, &k1, &d_between_hi, rf_ex, &got_ex),
              "prefix micro: range ex (between..between)");

        oracle_range_agg_pair(txn, dbi, &vv, &k1, &d_between_lo, &k1, &d_between_hi, rf_in, &exp);
        agg_require_equal(AGG_SCHEMA, &got_in, &exp, "range(between..between,in) oracle");
        agg_require_equal(AGG_SCHEMA, &got_ex, &exp, "range(between..between,ex) oracle");
        agg_require_equal(AGG_SCHEMA, &got_in, &got_ex, "range(between..between) incl==excl");

        REQUIRE(got_in.mv_agg_entries == 3, "range(between..between): expected 3 entries (d003..d005)");
        REQUIRE(got_in.mv_agg_keys == 1, "range(between..between): expected KEYS==1");
    }

    /* (5) range: empty range when bounds are before first dup. */
    {
        MDB_agg got, exp;
        unsigned rf = MDB_RANGE_LOWER_INCL | MDB_RANGE_UPPER_INCL;

        CHECK(mdb_agg_range(txn, dbi, &k1, &d_before_first, &k1, &d_before_first, rf, &got),
              "prefix micro: range empty (before..before)");
        oracle_range_agg_pair(txn, dbi, &vv, &k1, &d_before_first, &k1, &d_before_first, rf, &exp);
        agg_require_equal(AGG_SCHEMA, &got, &exp, "range(empty before) oracle");

        REQUIRE(got.mv_agg_entries == 0, "range(empty before): expected 0 entries");
        REQUIRE(got.mv_agg_keys == 0, "range(empty before): expected 0 keys");
        uint8_t z[MDB_HASH_SIZE]; memset(z, 0, MDB_HASH_SIZE);
        REQUIRE(memcmp(got.mv_agg_hashes, z, MDB_HASH_SIZE) == 0, "range(empty before): expected zero hashsum");
    }

    /* (6) range: exact bound with exclusivity -> empty.
     * [d003, d003) in record-order should be empty.
     */
    {
        MDB_agg got, exp;
        unsigned rf = MDB_RANGE_LOWER_INCL; /* upper exclusive */

        CHECK(mdb_agg_range(txn, dbi, &k1, &d_exact_d003, &k1, &d_exact_d003, rf, &got),
              "prefix micro: range empty (exact incl..exact excl)");
        oracle_range_agg_pair(txn, dbi, &vv, &k1, &d_exact_d003, &k1, &d_exact_d003, rf, &exp);
        agg_require_equal(AGG_SCHEMA, &got, &exp, "range(empty exact) oracle");

        REQUIRE(got.mv_agg_entries == 0, "range(empty exact): expected 0 entries");
        REQUIRE(got.mv_agg_keys == 0, "range(empty exact): expected 0 keys");
    }

    /* (7) range: mixed bounds across keys (low has data, high is key-only). */
    {
        MDB_agg got, exp;
        unsigned rf = MDB_RANGE_LOWER_INCL | MDB_RANGE_UPPER_INCL;

        CHECK(mdb_agg_range(txn, dbi, &k1, &d_between_hi, &k2, NULL, rf, &got),
              "prefix micro: range mixed (K1,d0055 .. K2)");
        oracle_range_agg_pair(txn, dbi, &vv, &k1, &d_between_hi, &k2, NULL, rf, &exp);
        agg_require_equal(AGG_SCHEMA, &got, &exp, "range(mixed) oracle");

        REQUIRE(got.mv_agg_keys == 2, "range(mixed): expected KEYS==2 (K1 tail + K2)");
    }

    /* (8) prefix: missing key must ignore data and match key-only prefix(). */
    {
        MDB_agg got, exp;
        CHECK(mdb_agg_prefix(txn, dbi, &k1a, &d_between_lo, 0, &got), "prefix micro: prefix missing key");
        CHECK(mdb_agg_prefix(txn, dbi, &k1a, NULL, 0, &exp), "prefix micro: prefix missing key");
        agg_require_equal(AGG_SCHEMA, &got, &exp, "prefix(missing key) == prefix(missing key)");
    }

    /* (9) prefix: single-value key compares against data bound. */
    {
        MDB_agg got_lo, got_hi, exp_lo, exp_hi, keyonly_ex, keyonly_in;

        CHECK(mdb_agg_prefix(txn, dbi, &k2, &d_k2_low, 0, &got_lo), "prefix micro: prefix K2 low");
        CHECK(mdb_agg_prefix(txn, dbi, &k2, &d_k2_high, 0, &got_hi), "prefix micro: prefix K2 high");

        oracle_prefix_agg(txn, dbi, &vv, &k2, &d_k2_low, 0, &exp_lo);
        oracle_prefix_agg(txn, dbi, &vv, &k2, &d_k2_high, 0, &exp_hi);

        agg_require_equal(AGG_SCHEMA, &got_lo, &exp_lo, "prefix(K2,low) oracle");
        agg_require_equal(AGG_SCHEMA, &got_hi, &exp_hi, "prefix(K2,high) oracle");

        CHECK(mdb_agg_prefix(txn, dbi, &k2, NULL, 0, &keyonly_ex), "prefix micro: prefix(K2,excl)");
        CHECK(mdb_agg_prefix(txn, dbi, &k2, NULL, MDB_AGG_PREFIX_INCL, &keyonly_in), "prefix micro: prefix(K2,incl)");

        agg_require_equal(AGG_SCHEMA, &got_lo, &keyonly_ex, "prefix(K2,low) == prefix(K2,excl)");
        agg_require_equal(AGG_SCHEMA, &got_hi, &keyonly_in, "prefix(K2,high) == prefix(K2,incl)");
    }

    recvec_free(&vv);
    keyidx_free(&kv);

    CHECK(mdb_txn_commit(txn), "prefix micro: commit");
    mdb_dbi_close(env, dbi);
    mdb_env_close(env);
}


static void test_dupsort_main_tree_delete_middle(void)
{
    char dir[256];
    make_test_dir(dir, sizeof(dir), "t12_dups_main_delmid");
    MDB_env *env = open_env_dir(dir, 16, 256u * 1024u * 1024u, 1);

    MDB_txn *txn = NULL;
    MDB_dbi dbi;

    CHECK(mdb_txn_begin(env, NULL, 0, &txn), "dups main delmid: begin");
    CHECK(mdb_dbi_open(txn, "dups", MDB_CREATE | MDB_DUPSORT | AGG_SCHEMA, &dbi), "dups main delmid: dbi_open");

    const unsigned KEYS = (SUBDB_KEYS < 800u) ? 800u : SUBDB_KEYS;
    const unsigned DUPS = 2;
    char kbuf[64];
    uint8_t vbuf[MDB_HASH_SIZE + 128];

    for (unsigned i = 0; i < KEYS; i++) {
        make_key(kbuf, sizeof(kbuf), "K", i);
        MDB_val k = { strlen(kbuf), kbuf };
        for (unsigned j = 0; j < DUPS; j++) {
            MDB_val d;
            make_val_with_hash(vbuf, sizeof(vbuf), "D", ((i<<8) ^ j), &d);
            CHECK(mdb_put(txn, dbi, &k, &d, 0), "dups main delmid: put");
            DBG_CHECK(txn, dbi, "dups main delmid: dbg agg");
        }
    }

    /* Delete middle third of keys entirely (data=NULL), forcing merges/rebalances in main tree. */
    unsigned lo = KEYS / 3u;
    unsigned hi = (2u * KEYS) / 3u;
    for (unsigned i = lo; i < hi; i++) {
        make_key(kbuf, sizeof(kbuf), "K", i);
        MDB_val k = { strlen(kbuf), kbuf };
        CHECK(mdb_del(txn, dbi, &k, NULL), "dups main delmid: del key");
        DBG_CHECK(txn, dbi, "dups main delmid: dbg agg");
    }

    verify_against_oracle(txn, dbi, "dups main delmid");
    CHECK(mdb_txn_commit(txn), "dups main delmid: commit");

    CHECK(mdb_txn_begin(env, NULL, MDB_RDONLY, &txn), "dups main delmid: ro begin");
    verify_against_oracle(txn, dbi, "dups main delmid (ro)");
    mdb_txn_abort(txn);

    mdb_dbi_close(env, dbi);
    mdb_env_close(env);
}

#define DF_ELEM_SIZE MDB_HASH_SIZE
#define DF_DUPS 48

static void test_dupfixed_multiple(void)
{
    char dir[256];
    make_test_dir(dir, sizeof(dir), "t13_dupfixed_multiple");
    MDB_env *env = open_env_dir(dir, 8, 128u * 1024u * 1024u, 1);

    MDB_txn *txn = NULL;
    MDB_dbi dbi;

    CHECK(mdb_txn_begin(env, NULL, 0, &txn), "txn_begin");
    CHECK(mdb_dbi_open(txn, "df", MDB_CREATE | MDB_DUPSORT | MDB_DUPFIXED | AGG_SCHEMA, &dbi),
          "dbi_open dupfixed");

    MDB_cursor *cur = NULL;
    CHECK(mdb_cursor_open(txn, dbi, &cur), "cursor_open dupfixed");

    const unsigned KEYS = 25;
    char kbuf[64];

    unsigned char elems[DF_DUPS * DF_ELEM_SIZE];
    MDB_val dvals[2];
    dvals[0].mv_size = DF_ELEM_SIZE;

    for (unsigned i = 0; i < KEYS; i++) {
        make_key(kbuf, sizeof(kbuf), "K", i);
        MDB_val key = { strlen(kbuf), kbuf };

        /* Fill deterministic elements */
        for (unsigned j = 0; j < DF_DUPS; j++) {
            fill_hash_prefix((uint8_t *)&elems[j * DF_ELEM_SIZE], ((uint64_t)i<<16) ^ j);
        }

        size_t remaining = DF_DUPS;
        unsigned char *p = elems;
        while (remaining) {
            dvals[0].mv_data = p;
            dvals[1].mv_size = remaining;
            int rc = mdb_cursor_put(cur, &key, dvals, MDB_MULTIPLE);
            CHECK(rc, "df multiple: cursor_put");
            DBG_CHECK(txn, dbi, "df multiple: dbg agg");

            size_t wrote = dvals[1].mv_size;
            if (wrote == 0 || wrote > remaining) {
                fprintf(stderr, "df multiple: no progress or invalid progress: remaining=%zu wrote=%zu key=%.*s\n",
                        remaining, wrote, (int)key.mv_size, (const char *)key.mv_data);
                exit(EXIT_FAILURE);
            }
            if (wrote != remaining) {
                fprintf(stderr, "df multiple: partial MDB_MULTIPLE wrote=%zu remaining=%zu key=%.*s\n",
                        wrote, remaining, (int)key.mv_size, (const char *)key.mv_data);
            }
            remaining -= wrote;
            p += wrote * DF_ELEM_SIZE;
        }
    }

    mdb_cursor_close(cur);

    verify_against_oracle(txn, dbi, "dupfixed multiple");
    CHECK(mdb_txn_commit(txn), "dupfixed multiple: commit");

    CHECK(mdb_txn_begin(env, NULL, MDB_RDONLY, &txn), "dupfixed multiple: ro begin");
    verify_against_oracle(txn, dbi, "dupfixed multiple (ro)");
    mdb_txn_abort(txn);

    mdb_dbi_close(env, dbi);
    mdb_env_close(env);
}

static void test_dupfixed_merge_and_collapse(void)
{
    /* Stress DUPFIXED duplicate storage transitions and main-tree merges:
     *  - Build many keys with many fixed-size duplicates (subpage/subDB pressure)
     *  - Shrink many dupsets down to 1 element (merge/collapse inside dupset)
     *  - Delete a middle band of keys (forces main-tree merges)
     *  - Regrow duplicates for survivors (splits again)
     */
    char dir[256];
    make_test_dir(dir, sizeof(dir), "t10_dupfixed_merge_collapse");
    MDB_env *env = open_env_dir(dir, 16, 256u * 1024u * 1024u, 1);

    MDB_txn *txn = NULL;
    MDB_dbi dbi;

    CHECK(mdb_txn_begin(env, NULL, 0, &txn), "df merge/collapse: begin");
    CHECK(mdb_dbi_open(txn, "df", MDB_CREATE | MDB_DUPSORT | MDB_DUPFIXED | AGG_SCHEMA, &dbi),
          "df merge/collapse: open");

    MDB_cursor *cur = NULL;
    CHECK(mdb_cursor_open(txn, dbi, &cur), "df merge/collapse: cursor_open");

#ifdef MDB_DEBUG_AGG_INTEGRITY
    const unsigned KEYS = 50;
    const unsigned DUPS = 96;
    const unsigned REGROW = 48;
#else
    const unsigned KEYS = 140;
    const unsigned DUPS = 256;
    const unsigned REGROW = 96;
#endif

    char kbuf[64];
    unsigned char *elems = (unsigned char *)malloc((size_t)DUPS * DF_ELEM_SIZE);
    REQUIRE(elems != NULL, "df merge/collapse: malloc elems");
    MDB_val dvals[2];
    dvals[0].mv_size = DF_ELEM_SIZE;

    /* Populate */
    for (unsigned i = 0; i < KEYS; i++) {
        make_key(kbuf, sizeof(kbuf), "K", i);
        MDB_val key = { strlen(kbuf), kbuf };

        for (unsigned j = 0; j < DUPS; j++)
            fill_hash_prefix((uint8_t *)&elems[(size_t)j * DF_ELEM_SIZE], ((uint64_t)i<<20) ^ j);

        size_t remaining = DUPS;
        unsigned char *p = elems;
        while (remaining) {
            dvals[0].mv_data = p;
            dvals[1].mv_size = remaining;
            CHECK(mdb_cursor_put(cur, &key, dvals, MDB_MULTIPLE), "df merge/collapse: cursor_put");
            size_t wrote = dvals[1].mv_size;
            REQUIRE(wrote > 0 && wrote <= remaining, "df merge/collapse: MDB_MULTIPLE progress");
            remaining -= wrote;
            p += wrote * DF_ELEM_SIZE;
        }

        if ((i & 15u) == 0u)
            DBG_CHECK(txn, dbi, "df merge/collapse: dbg build");
    }

    DBG_CHECK(txn, dbi, "df merge/collapse: dbg build end");

    /* Shrink most dupsets to a single element (cursor_del). */
    for (unsigned i = 0; i < KEYS; i++) {
        make_key(kbuf, sizeof(kbuf), "K", i);
        MDB_val key = { strlen(kbuf), kbuf };
        MDB_val d = { 0, NULL };

        int rc = mdb_cursor_get(cur, &key, &d, MDB_SET_KEY);
        if (rc == MDB_NOTFOUND)
            continue;
        CHECK(rc, "df merge/collapse: set_key");

        /* Position at first dup and delete DUPS-1 duplicates. */
        CHECK(mdb_cursor_get(cur, &key, &d, MDB_FIRST_DUP), "df merge/collapse: first_dup");

        unsigned to_del = (DUPS > 1u) ? (DUPS - 1u) : 0u;
        /* For some keys, keep more than 1 to maintain variety. */
        if ((i % 7u) == 0u && to_del > 10u)
            to_del -= 10u;

        for (unsigned j = 0; j < to_del; j++) {
            rc = mdb_cursor_del(cur, 0);
            if (rc == MDB_NOTFOUND)
                break;
            CHECK(rc, "df merge/collapse: cursor_del");
            if ((j & 31u) == 0u)
                DBG_CHECK(txn, dbi, "df merge/collapse: dbg shrink");
        }
    }

    DBG_CHECK(txn, dbi, "df merge/collapse: dbg after shrink");

    /* Delete a middle band of keys completely (main-tree merges). */
    unsigned lo = KEYS / 3u;
    unsigned hi = (2u * KEYS) / 3u;
    for (unsigned i = lo; i < hi; i++) {
        make_key(kbuf, sizeof(kbuf), "K", i);
        MDB_val key = { strlen(kbuf), kbuf };
        (void)mdb_del(txn, dbi, &key, NULL);
        if ((i & 15u) == 0u)
            DBG_CHECK(txn, dbi, "df merge/collapse: dbg del keys");
    }

    DBG_CHECK(txn, dbi, "df merge/collapse: dbg after key deletes");

    /* Regrow duplicates for survivors (splits again). */
    for (unsigned i = 0; i < KEYS; i++) {
        if (i >= lo && i < hi)
            continue; /* deleted */
        if ((i & 3u) != 0u)
            continue; /* regrow subset */

        make_key(kbuf, sizeof(kbuf), "K", i);
        MDB_val key = { strlen(kbuf), kbuf };

        /* Add REGROW new unique elements. */
        for (unsigned j = 0; j < REGROW; j++)
            fill_hash_prefix((uint8_t *)&elems[(size_t)j * DF_ELEM_SIZE], ((uint64_t)i<<20) ^ (0x10000u + j));

        size_t remaining = REGROW;
        unsigned char *p = elems;
        while (remaining) {
            dvals[0].mv_data = p;
            dvals[1].mv_size = remaining;
            CHECK(mdb_cursor_put(cur, &key, dvals, MDB_MULTIPLE), "df merge/collapse: regrow put");
            size_t wrote = dvals[1].mv_size;
            REQUIRE(wrote > 0 && wrote <= remaining, "df merge/collapse: regrow progress");
            remaining -= wrote;
            p += wrote * DF_ELEM_SIZE;
        }

        DBG_CHECK(txn, dbi, "df merge/collapse: dbg regrow");
    }

    mdb_cursor_close(cur);
    free(elems);

    verify_against_oracle(txn, dbi, "dupfixed merge/collapse");
    CHECK(mdb_txn_commit(txn), "df merge/collapse: commit");

    CHECK(mdb_txn_begin(env, NULL, MDB_RDONLY, &txn), "df merge/collapse: ro begin");
    verify_against_oracle(txn, dbi, "dupfixed merge/collapse (ro)");
    mdb_txn_abort(txn);

    mdb_dbi_close(env, dbi);
    mdb_env_close(env);
}

static void test_upgrade_policy_named_db(void)
{
    /* Create a named DB without some aggregate bits, then attempt to open with expanded schema. */
    char dir[256];
    make_test_dir(dir, sizeof(dir), "t15_upgrade_named");
    MDB_env *env = open_env_dir(dir, 8, 64u * 1024u * 1024u, 1);

    MDB_txn *txn = NULL;
    MDB_dbi dbi1, dbi2;

    CHECK(mdb_txn_begin(env, NULL, 0, &txn), "upgrade named: begin");
    CHECK(mdb_dbi_open(txn, "u", MDB_CREATE | MDB_AGG_ENTRIES, &dbi1), "upgrade named: open entries-only");

    /* insert enough to likely create branches */
    char kbuf[64];
    uint8_t vbuf[MDB_HASH_SIZE + 128];
    for (unsigned i = 0; i < 600; i++) {
        make_key(kbuf, sizeof(kbuf), "K", i);
        MDB_val k = { strlen(kbuf), kbuf };
        MDB_val d;
        make_val_with_hash(vbuf, sizeof(vbuf), "V", i, &d);
        CHECK(mdb_put(txn, dbi1, &k, &d, 0), "upgrade named: put");
        DBG_CHECK(txn, dbi1, "upgrade named: dbg agg");
    }

    CHECK(mdb_txn_commit(txn), "upgrade named: commit");

    CHECK(mdb_txn_begin(env, NULL, 0, &txn), "upgrade named: begin2");
    /* Opening same named DB with expanded schema should be incompatible. */
    int rc = mdb_dbi_open(txn, "u", MDB_AGG_ENTRIES | MDB_AGG_KEYS | MDB_AGG_HASHSUM, &dbi2);
    expect_rc(rc, MDB_INCOMPATIBLE, "upgrade named: schema expansion");
    mdb_txn_abort(txn);

    mdb_dbi_close(env, dbi1);
    mdb_env_close(env);
}



/* ------------------------------ additional coverage tests (unit4) ------------------------------ */

static void prefix_vs_range_check(MDB_txn *txn, MDB_dbi dbi, const char *label, const MDB_val *qk)
{
    unsigned schema = 0;
    CHECK(mdb_agg_info(txn, dbi, &schema), "mdb_agg_info");

    MDB_agg p0, p1, r0, r1;
    CHECK(mdb_agg_prefix(txn, dbi, qk, NULL, 0, &p0), "mdb_agg_prefix excl");
    CHECK(mdb_agg_range(txn, dbi, NULL, NULL, qk, NULL, 0, &r0), "mdb_agg_range upper excl");

    {
        char msg[256];
        snprintf(msg, sizeof(msg), "%s: prefix(excl) == range(upper_excl)", label);
        agg_require_equal(schema, &p0, &r0, msg);
    }

    CHECK(mdb_agg_prefix(txn, dbi, qk, NULL, MDB_AGG_PREFIX_INCL, &p1), "mdb_agg_prefix incl");
    CHECK(mdb_agg_range(txn, dbi, NULL, NULL, qk, NULL, MDB_RANGE_UPPER_INCL, &r1), "mdb_agg_range upper incl");

    {
        char msg[256];
        snprintf(msg, sizeof(msg), "%s: prefix(incl) == range(upper_incl)", label);
        agg_require_equal(schema, &p1, &r1, msg);
    }
}



static void prefix_vs_range_check_pair(MDB_txn *txn, MDB_dbi dbi, const char *label,
                                      const MDB_val *qk, const MDB_val *qd)
{
    unsigned schema = 0;
    CHECK(mdb_agg_info(txn, dbi, &schema), "mdb_agg_info");

    MDB_agg p0, p1, r0, r1;
    CHECK(mdb_agg_prefix(txn, dbi, qk, qd, 0, &p0), "mdb_agg_prefix excl");
    CHECK(mdb_agg_range(txn, dbi, NULL, NULL, qk, qd, 0, &r0), "mdb_agg_range upper excl");

    {
        char msg[256];
        snprintf(msg, sizeof(msg), "%s: prefix(excl) == range(upper_excl)", label);
        agg_require_equal(schema, &p0, &r0, msg);
    }

    CHECK(mdb_agg_prefix(txn, dbi, qk, qd, MDB_AGG_PREFIX_INCL, &p1), "mdb_agg_prefix incl");
    CHECK(mdb_agg_range(txn, dbi, NULL, NULL, qk, qd, MDB_RANGE_UPPER_INCL, &r1), "mdb_agg_range upper incl");

    {
        char msg[256];
        snprintf(msg, sizeof(msg), "%s: prefix(incl) == range(upper_incl)", label);
        agg_require_equal(schema, &p1, &r1, msg);
    }
}

static void test_prefix_matches_range(void)
{
    char dir[256];
    make_test_dir(dir, sizeof(dir), "t16_prefix_range");
    MDB_env *env = open_env_dir(dir, 16, 96u * 1024u * 1024u, 1);

    MDB_txn *txn = NULL;
    MDB_dbi dbp, dbd;

    CHECK(mdb_txn_begin(env, NULL, 0, &txn), "prefix: begin");
    CHECK(mdb_dbi_open(txn, "plain", MDB_CREATE | AGG_SCHEMA, &dbp), "prefix: open plain");
    CHECK(mdb_dbi_open(txn, "dups", MDB_CREATE | MDB_DUPSORT | AGG_SCHEMA, &dbd), "prefix: open dups");

    /* Fill plain */
    char kbuf[64];
    uint8_t vbuf[MDB_HASH_SIZE + 128];
    for (unsigned i = 0; i < 200; i++) {
        make_key(kbuf, sizeof(kbuf), "K", i);
        MDB_val k = { strlen(kbuf), kbuf };
        MDB_val d;
        make_val_with_hash_salted(vbuf, sizeof(vbuf), "P", i, 0, &d);
        CHECK(mdb_put(txn, dbp, &k, &d, 0), "prefix: put plain");
        DBG_CHECK(txn, dbp, "prefix: dbg plain");
    }

    /* Fill dupsort: 100 keys, 3 dups each */
    for (unsigned i = 0; i < 100; i++) {
        make_key(kbuf, sizeof(kbuf), "K", i);
        MDB_val k = { strlen(kbuf), kbuf };
        for (unsigned j = 0; j < 3; j++) {
            MDB_val d;
            make_val_with_hash_salted(vbuf, sizeof(vbuf), "D", (i * 10u) + j, (uint64_t)j, &d);
            CHECK(mdb_put(txn, dbd, &k, &d, 0), "prefix: put dups");
            DBG_CHECK(txn, dbd, "prefix: dbg dups");
        }
    }

    /* Query keys: before-first, exact-first, between, after-last */
    {
        const char *q1s = "J00000000"; /* before K... */
        MDB_val q1 = { strlen(q1s), (void *)q1s };
        prefix_vs_range_check(txn, dbp, "plain before-first", &q1);
        prefix_vs_range_check(txn, dbd, "dups before-first", &q1);

        const char *q2s = "K00000000";
        MDB_val q2 = { strlen(q2s), (void *)q2s };
        prefix_vs_range_check(txn, dbp, "plain exact-first", &q2);
        prefix_vs_range_check(txn, dbd, "dups exact-first", &q2);

        /* Record-order prefix/range check at an in-dupset boundary for the first key. */
        {
            MDB_val qd;
            make_val_with_hash_salted(vbuf, sizeof(vbuf), "D", 1u, 1u, &qd); /* (i=0,j=1) */
            prefix_vs_range_check_pair(txn, dbd, "dups exact-first (mid-dup)", &q2, &qd);
        }


        const char *q3s = "K00000050a"; /* between K00000050 and K00000051 */
        MDB_val q3 = { strlen(q3s), (void *)q3s };
        prefix_vs_range_check(txn, dbp, "plain between", &q3);
        prefix_vs_range_check(txn, dbd, "dups between", &q3);

        /* Record-order prefix/range check at an in-dupset boundary for an interior key. */
        {
            const char *q50s = "K00000050";
            MDB_val q50 = { strlen(q50s), (void *)q50s };
            MDB_val qd;
            make_val_with_hash_salted(vbuf, sizeof(vbuf), "D", (50u * 10u) + 1u, 1u, &qd); /* (i=50,j=1) */
            prefix_vs_range_check_pair(txn, dbd, "dups K50 (mid-dup)", &q50, &qd);
        }


        const char *q4s = "L00000000"; /* after K... */
        MDB_val q4 = { strlen(q4s), (void *)q4s };
        prefix_vs_range_check(txn, dbp, "plain after-last", &q4);
        prefix_vs_range_check(txn, dbd, "dups after-last", &q4);
    }

    verify_against_oracle(txn, dbp, "prefix plain");
    verify_against_oracle(txn, dbd, "prefix dups");

    CHECK(mdb_txn_commit(txn), "prefix: commit");

    mdb_dbi_close(env, dbp);
    mdb_dbi_close(env, dbd);
    mdb_env_close(env);
}

static size_t oracle_lower_bound_key(MDB_txn *txn, MDB_dbi dbi, const RecVec *vv, const MDB_val *qk)
{
    for (size_t i = 0; i < vv->len; i++) {
        int c = rec_key_cmp(txn, dbi, &vv->v[i], qk);
        if (c >= 0)
            return i;
    }
    return vv->len;
}

static void test_rank_input_shape_rules(void)
{
    char dir[256];
    make_test_dir(dir, sizeof(dir), "t17_rank_shape");
    MDB_env *env = open_env_dir(dir, 16, 96u * 1024u * 1024u, 1);

    MDB_txn *txn = NULL;
    MDB_dbi dbp, dbd;

    CHECK(mdb_txn_begin(env, NULL, 0, &txn), "rankshape: begin");
    CHECK(mdb_dbi_open(txn, "plain", MDB_CREATE | AGG_SCHEMA, &dbp), "rankshape: open plain");
    CHECK(mdb_dbi_open(txn, "dups", MDB_CREATE | MDB_DUPSORT | AGG_SCHEMA, &dbd), "rankshape: open dups");

    char kbuf[64];
    uint8_t vbuf[MDB_HASH_SIZE + 128];

    /* plain: 10 keys */
    for (unsigned i = 0; i < 10; i++) {
        make_key(kbuf, sizeof(kbuf), "K", i);
        MDB_val k = { strlen(kbuf), kbuf };
        MDB_val d;
        make_val_with_hash_salted(vbuf, sizeof(vbuf), "P", i, 0, &d);
        CHECK(mdb_put(txn, dbp, &k, &d, 0), "rankshape: put plain");
        DBG_CHECK(txn, dbp, "rankshape: dbg plain");
    }

    /* dupsort: 5 keys, 3 dups each */
    for (unsigned i = 0; i < 5; i++) {
        make_key(kbuf, sizeof(kbuf), "K", i);
        MDB_val k = { strlen(kbuf), kbuf };
        for (unsigned j = 0; j < 3; j++) {
            MDB_val d;
            make_val_with_hash_salted(vbuf, sizeof(vbuf), "D", (i * 10u) + j, (uint64_t)j, &d);
            CHECK(mdb_put(txn, dbd, &k, &d, 0), "rankshape: put dups");
            DBG_CHECK(txn, dbd, "rankshape: dbg dups");
        }
    }

    /* (A) plain: non-empty data is invalid for rank (both weights) */
    {
        const char *ks = "K00000005";
        MDB_val qk = { strlen(ks), (void *)ks };
        uint8_t db = 0xAA;
        MDB_val qd = { 1, &db };
        uint64_t rr = 0, di = 0;
        int rc = mdb_agg_rank(txn, dbp, &qk, &qd, MDB_AGG_WEIGHT_ENTRIES, MDB_AGG_RANK_EXACT, &rr, &di);
        expect_rc(rc, EINVAL, "rankshape: plain entries with non-empty data");

        qk.mv_size = strlen(ks); qk.mv_data = (void *)ks;
        qd.mv_size = 1; qd.mv_data = &db;
        rc = mdb_agg_rank(txn, dbp, &qk, &qd, MDB_AGG_WEIGHT_KEYS, MDB_AGG_RANK_EXACT, &rr, &di);
        expect_rc(rc, EINVAL, "rankshape: plain keys with non-empty data");
    }

    /* (B) dupsort: keys-weight requires empty data */
    {
        const char *ks = "K00000001";
        MDB_val qk = { strlen(ks), (void *)ks };
        uint8_t db = 0xBB;
        MDB_val qd = { 1, &db };
        uint64_t rr = 0, di = 0;
        int rc = mdb_agg_rank(txn, dbd, &qk, &qd, MDB_AGG_WEIGHT_KEYS, MDB_AGG_RANK_EXACT, &rr, &di);
        expect_rc(rc, EINVAL, "rankshape: dupsort keys-weight with data");
    }

    /* (C) dupsort: entries-weight exact requires specified data */
    {
        const char *ks = "K00000002";
        MDB_val qk = { strlen(ks), (void *)ks };
        MDB_val qd = { 0, NULL }; /* unspecified */
        uint64_t rr = 0, di = 0;
        int rc = mdb_agg_rank(txn, dbd, &qk, &qd, MDB_AGG_WEIGHT_ENTRIES, MDB_AGG_RANK_EXACT, &rr, &di);
        expect_rc(rc, EINVAL, "rankshape: dupsort entries exact with unspecified data");
    }

    /* (D) dupsort: entries-weight set-range with unspecified data behaves like SET_RANGE on key */
    {
        RecVec vv;
        oracle_build(txn, dbd, &vv);

        const char *ks = "K00000002a"; /* between K2 and K3 */
        MDB_val qk = { strlen(ks), (void *)ks };
        MDB_val qd = { 0, NULL }; /* unspecified */

        size_t exp = oracle_lower_bound_key(txn, dbd, &vv, &qk);
        REQUIRE(exp < vv.len, "rankshape: expected SET_RANGE to find something");

        uint64_t rr = UINT64_MAX, di = UINT64_MAX;
        int rc = mdb_agg_rank(txn, dbd, &qk, &qd, MDB_AGG_WEIGHT_ENTRIES, MDB_AGG_RANK_SET_RANGE, &rr, &di);
        CHECK(rc, "rankshape: dupsort entries set-range (unspecified data)");

        if (rr != (uint64_t)exp) {
            fprintf(stderr, "rankshape: expected rr=%zu got=%" PRIu64 "\n", exp, rr);
            exit(EXIT_FAILURE);
        }
        const Rec *or = &vv.v[exp];
        if (qk.mv_size != or->klen || memcmp(qk.mv_data, or->k, or->klen) != 0 ||
            qd.mv_size != or->dlen || memcmp(qd.mv_data, or->d, or->dlen) != 0) {
            fprintf(stderr, "rankshape: set-range did not update to expected record\n");
            exit(EXIT_FAILURE);
        }
        /* Should land on first dup of the next key, so dup index should be 0. */
        if (di != 0) {
            fprintf(stderr, "rankshape: expected dup_index=0 got=%" PRIu64 "\n", di);
            exit(EXIT_FAILURE);
        }

        recvec_free(&vv);
    }

    verify_against_oracle(txn, dbp, "rankshape plain");
    verify_against_oracle(txn, dbd, "rankshape dups");

    CHECK(mdb_txn_commit(txn), "rankshape: commit");

    mdb_dbi_close(env, dbp);
    mdb_dbi_close(env, dbd);
    mdb_env_close(env);
}

static void test_hashsum_reserve_incompatible(void)
{
    char dir[256];
    make_test_dir(dir, sizeof(dir), "t18_hashsum_reserve");
    MDB_env *env = open_env_dir(dir, 8, 64u * 1024u * 1024u, 1);

    MDB_txn *txn = NULL;
    MDB_dbi dbi;
    CHECK(mdb_txn_begin(env, NULL, 0, &txn), "hsreserve: begin");
    CHECK(mdb_dbi_open(txn, "plain", MDB_CREATE | AGG_SCHEMA, &dbi), "hsreserve: open");

    char kbuf[64];
    uint8_t vbuf[MDB_HASH_SIZE + 128];

    /* baseline record */
    make_key(kbuf, sizeof(kbuf), "K", 1);
    MDB_val k1 = { strlen(kbuf), kbuf };
    MDB_val d1;
    make_val_with_hash_salted(vbuf, sizeof(vbuf), "V", 1, 0, &d1);
    CHECK(mdb_put(txn, dbi, &k1, &d1, 0), "hsreserve: put baseline");
    DBG_CHECK(txn, dbi, "hsreserve: dbg");

    unsigned schema = 0;
    CHECK(mdb_agg_info(txn, dbi, &schema), "hsreserve: agg_info");

    MDB_agg before, after;
    CHECK(mdb_agg_totals(txn, dbi, &before), "hsreserve: totals before");

    /* attempt MDB_RESERVE: must be incompatible when hashsum is enabled */
    make_key(kbuf, sizeof(kbuf), "K", 2);
    MDB_val k2 = { strlen(kbuf), kbuf };
    MDB_val d2 = { MDB_HASH_SIZE + 16u, NULL };
    int rc = mdb_put(txn, dbi, &k2, &d2, MDB_RESERVE);
    expect_rc(rc, MDB_INCOMPATIBLE, "hsreserve: put MDB_RESERVE incompatible");

    CHECK(mdb_agg_totals(txn, dbi, &after), "hsreserve: totals after");
    agg_require_equal(schema, &before, &after, "hsreserve: totals unchanged");

    verify_against_oracle(txn, dbi, "hsreserve");
    CHECK(mdb_txn_commit(txn), "hsreserve: commit");

    mdb_dbi_close(env, dbi);
    mdb_env_close(env);
}

static void test_hashsum_writemap_incompatible(void)
{
    char dir[256];
    make_test_dir(dir, sizeof(dir), "t19_hashsum_writemap");

    /* WRITEMAP should make HASHSUM DBIs incompatible for updates. */
    MDB_env *env = open_env_dir_flags(dir, 16, 128u * 1024u * 1024u, 1, MDB_WRITEMAP);

    MDB_txn *txn = NULL;
    MDB_dbi dbhs, dbok;

    CHECK(mdb_txn_begin(env, NULL, 0, &txn), "hswritemap: begin");
    CHECK(mdb_dbi_open(txn, "hs", MDB_CREATE | AGG_SCHEMA, &dbhs), "hswritemap: open hashsum");
    CHECK(mdb_dbi_open(txn, "ok", MDB_CREATE | (MDB_AGG_ENTRIES|MDB_AGG_KEYS), &dbok), "hswritemap: open no-hashsum");

    char kbuf[64];
    uint8_t vbuf[MDB_HASH_SIZE + 128];

    make_key(kbuf, sizeof(kbuf), "K", 1);
    MDB_val k = { strlen(kbuf), kbuf };
    MDB_val d;
    make_val_with_hash_salted(vbuf, sizeof(vbuf), "V", 1, 0, &d);

    int rc = mdb_put(txn, dbhs, &k, &d, 0);
    expect_rc(rc, MDB_INCOMPATIBLE, "hswritemap: put incompatible in hashsum DB");

    /* control DB without hashsum should accept puts */
    for (unsigned i = 0; i < 200; i++) {
        make_key(kbuf, sizeof(kbuf), "K", i);
        MDB_val kk = { strlen(kbuf), kbuf };
        MDB_val dd;
        make_val_with_hash_salted(vbuf, sizeof(vbuf), "C", i, 0, &dd);
        CHECK(mdb_put(txn, dbok, &kk, &dd, 0), "hswritemap: put control");
        DBG_CHECK(txn, dbok, "hswritemap: dbg control");
    }

    verify_against_oracle(txn, dbhs, "hswritemap hashsum (empty)");
    verify_against_oracle(txn, dbok, "hswritemap control");

    CHECK(mdb_txn_commit(txn), "hswritemap: commit");

    mdb_dbi_close(env, dbhs);
    mdb_dbi_close(env, dbok);
    mdb_env_close(env);
}

static void test_hashsum_nonzero_offset_plain(void)
{
    char dir[256];
    make_test_dir(dir, sizeof(dir), "t20_hashoff_plain");
    MDB_env *env = open_env_dir(dir, 8, 64u * 1024u * 1024u, 1);

    const size_t off = 64u; /* > MDB_HASH_SIZE so any "old behavior" (offset 0) is obviously wrong. */
    const uint64_t salt0 = 0;
    const uint64_t salt1 = 0x1111111111111111ULL;

    MDB_txn *txn = NULL;
    MDB_dbi dbi;

    CHECK(mdb_txn_begin(env, NULL, 0, &txn), "hashoff plain: begin");
    CHECK(mdb_dbi_open(txn, "plain", MDB_CREATE | AGG_SCHEMA, &dbi), "hashoff plain: open");
    CHECK(mdb_set_hash_offset(txn, dbi, (unsigned)off), "hashoff plain: set_hash_offset");

    uint8_t exp[MDB_HASH_SIZE] = {0};
    uint8_t wrong[MDB_HASH_SIZE] = {0};
    uint8_t tmp[MDB_HASH_SIZE];

    char kbuf[64];
    uint8_t vbuf[512];

    /* Insert 20 keys with hash-at-offset, compute expected hashsum from that region. */
    for (unsigned i = 0; i < 20; i++) {
        make_key(kbuf, sizeof(kbuf), "K", i);
        MDB_val k = { strlen(kbuf), kbuf };
        MDB_val d;
        make_val_with_hash_offset_salted(vbuf, sizeof(vbuf), off, "V", i, salt0, &d);

        /* expected: sum hash bytes at offset */
        expect_hash_for_id_salt(tmp, i, salt0);
        mdb_hashsum_add(exp, tmp);

        /* wrong: sum first MDB_HASH_SIZE bytes (prefix-only, since off >= MDB_HASH_SIZE) */
        expect_prefix_first_hashbytes(tmp, i);
        mdb_hashsum_add(wrong, tmp);

        CHECK(mdb_put(txn, dbi, &k, &d, 0), "hashoff plain: put");
    }

    /* Overwrite one key, changing hash bytes (salt changes), prefix stays the same. */
    {
        const unsigned i = 7;
        make_key(kbuf, sizeof(kbuf), "K", i);
        MDB_val k = { strlen(kbuf), kbuf };
        MDB_val d;
        make_val_with_hash_offset_salted(vbuf, sizeof(vbuf), off, "V", i, salt1, &d);

        expect_hash_for_id_salt(tmp, i, salt0);
        mdb_hashsum_sub(exp, tmp);
        expect_hash_for_id_salt(tmp, i, salt1);
        mdb_hashsum_add(exp, tmp);

        CHECK(mdb_put(txn, dbi, &k, &d, 0), "hashoff plain: overwrite");
    }

    DBG_CHECK(txn, dbi, "hashoff plain: dbg");

    /* Totals must match the offset-hash oracle and must not match the prefix oracle. */
    {
        MDB_agg got;
        CHECK(mdb_agg_totals(txn, dbi, &got), "hashoff plain: totals");
        REQUIRE(got.mv_agg_entries == 20u, "hashoff plain: expected entries==20");
        REQUIRE(got.mv_agg_keys == 20u, "hashoff plain: expected keys==20");
        REQUIRE(memcmp(got.mv_agg_hashes, exp, MDB_HASH_SIZE) == 0, "hashoff plain: totals hashsum mismatch");
        REQUIRE(memcmp(got.mv_agg_hashes, wrong, MDB_HASH_SIZE) != 0, "hashoff plain: totals unexpectedly matches prefix-sum");
    }

    /* A deterministic inclusive key range should also hash from the offset region. */
    {
        uint8_t exp_sub[MDB_HASH_SIZE] = {0};
        for (unsigned i = 5; i <= 14; i++) {
            uint64_t s = (i == 7) ? salt1 : salt0;
            expect_hash_for_id_salt(tmp, i, s);
            mdb_hashsum_add(exp_sub, tmp);
        }

        char lbuf[64], hbuf[64];
        make_key(lbuf, sizeof(lbuf), "K", 5);
        make_key(hbuf, sizeof(hbuf), "K", 14);
        MDB_val lk = { strlen(lbuf), lbuf };
        MDB_val hk = { strlen(hbuf), hbuf };

        MDB_agg got;
        CHECK(mdb_agg_range(txn, dbi, &lk, NULL, &hk, NULL,
                             MDB_RANGE_LOWER_INCL | MDB_RANGE_UPPER_INCL, &got),
              "hashoff plain: range subset");
        REQUIRE(got.mv_agg_entries == 10u, "hashoff plain: expected 10 entries in [5,14]");
        REQUIRE(got.mv_agg_keys == 10u, "hashoff plain: expected 10 keys in [5,14]");
        REQUIRE(memcmp(got.mv_agg_hashes, exp_sub, MDB_HASH_SIZE) == 0, "hashoff plain: range hashsum mismatch");
    }

    CHECK(mdb_txn_commit(txn), "hashoff plain: commit");

    /* Re-check on a read txn (persistent correctness). */
    {
        MDB_txn *rtxn = NULL;
        CHECK(mdb_txn_begin(env, NULL, MDB_RDONLY, &rtxn), "hashoff plain: rtxn");
        MDB_agg got;
        CHECK(mdb_agg_totals(rtxn, dbi, &got), "hashoff plain: totals ro");
        REQUIRE(memcmp(got.mv_agg_hashes, exp, MDB_HASH_SIZE) == 0, "hashoff plain: totals ro hashsum mismatch");
        mdb_txn_abort(rtxn);
    }

    mdb_dbi_close(env, dbi);
    mdb_env_close(env);
}

static void test_hashsum_nonzero_offset_dupfixed(void)
{
    char dir[256];
    make_test_dir(dir, sizeof(dir), "t21_hashoff_dupfixed");
    MDB_env *env = open_env_dir(dir, 8, 64u * 1024u * 1024u, 1);

    const size_t off = 64u; /* > MDB_HASH_SIZE so old behavior is obviously wrong. */
    const uint64_t salt0 = 0;

    MDB_txn *txn = NULL;
    MDB_dbi dbi;

    CHECK(mdb_txn_begin(env, NULL, 0, &txn), "hashoff dupfixed: begin");
    CHECK(mdb_dbi_open(txn, "dups", MDB_CREATE | MDB_DUPSORT | MDB_DUPFIXED | AGG_SCHEMA, &dbi),
          "hashoff dupfixed: open");
    CHECK(mdb_set_hash_offset(txn, dbi, (unsigned)off), "hashoff dupfixed: set_hash_offset");

    uint8_t exp[MDB_HASH_SIZE] = {0};
    uint8_t wrong[MDB_HASH_SIZE] = {0};
    uint8_t tmp[MDB_HASH_SIZE];

    const char *ks = "K00000000";
    MDB_val k = { strlen(ks), (void *)ks };

    uint8_t vbuf[512];

    /* Insert 20 duplicates under the same key. */
    for (unsigned i = 0; i < 20; i++) {
        MDB_val d;
        make_val_with_hash_offset_salted(vbuf, sizeof(vbuf), off, "D", i, salt0, &d);

        expect_hash_for_id_salt(tmp, i, salt0);
        mdb_hashsum_add(exp, tmp);

        expect_prefix_first_hashbytes(tmp, i);
        mdb_hashsum_add(wrong, tmp);

        CHECK(mdb_put(txn, dbi, &k, &d, 0), "hashoff dupfixed: put dup");
    }

    DBG_CHECK(txn, dbi, "hashoff dupfixed: dbg");

    /* Totals: entries==20, keys==1, hashsum uses offset. */
    {
        MDB_agg got;
        CHECK(mdb_agg_totals(txn, dbi, &got), "hashoff dupfixed: totals");
        REQUIRE(got.mv_agg_entries == 20u, "hashoff dupfixed: expected entries==20");
        REQUIRE(got.mv_agg_keys == 1u, "hashoff dupfixed: expected keys==1");
        REQUIRE(memcmp(got.mv_agg_hashes, exp, MDB_HASH_SIZE) == 0, "hashoff dupfixed: totals hashsum mismatch");
        REQUIRE(memcmp(got.mv_agg_hashes, wrong, MDB_HASH_SIZE) != 0, "hashoff dupfixed: totals unexpectedly matches prefix-sum");

        MDB_agg all;
        CHECK(mdb_agg_range(txn, dbi, NULL, NULL, NULL, NULL, 0, &all), "hashoff dupfixed: range all");
        REQUIRE(memcmp(all.mv_agg_hashes, exp, MDB_HASH_SIZE) == 0, "hashoff dupfixed: range(all) hashsum mismatch");
    }

    CHECK(mdb_txn_commit(txn), "hashoff dupfixed: commit");

    mdb_dbi_close(env, dbi);
    mdb_env_close(env);
}

static void test_put_nooverwrite_no_agg_change(void)
{
    char dir[256];
    make_test_dir(dir, sizeof(dir), "t20_nooverwrite");
    MDB_env *env = open_env_dir(dir, 8, 64u * 1024u * 1024u, 1);

    MDB_txn *txn = NULL;
    MDB_dbi dbi;

    CHECK(mdb_txn_begin(env, NULL, 0, &txn), "nooverwrite: begin");
    CHECK(mdb_dbi_open(txn, "plain", MDB_CREATE | AGG_SCHEMA, &dbi), "nooverwrite: open");

    char kbuf[64];
    uint8_t vbuf[MDB_HASH_SIZE + 128];

    make_key(kbuf, sizeof(kbuf), "K", 1);
    MDB_val k = { strlen(kbuf), kbuf };
    MDB_val d1;
    make_val_with_hash_salted(vbuf, sizeof(vbuf), "V", 1, 0, &d1);
    CHECK(mdb_put(txn, dbi, &k, &d1, 0), "nooverwrite: put first");
    DBG_CHECK(txn, dbi, "nooverwrite: dbg");

    unsigned schema = 0;
    CHECK(mdb_agg_info(txn, dbi, &schema), "nooverwrite: agg_info");

    MDB_agg before, after;
    CHECK(mdb_agg_totals(txn, dbi, &before), "nooverwrite: totals before");

    MDB_val d2;
    make_val_with_hash_salted(vbuf, sizeof(vbuf), "V", 1, 0x1111111111111111ULL, &d2);
    int rc = mdb_put(txn, dbi, &k, &d2, MDB_NOOVERWRITE);
    expect_rc(rc, MDB_KEYEXIST, "nooverwrite: MDB_NOOVERWRITE keyexists");

    CHECK(mdb_agg_totals(txn, dbi, &after), "nooverwrite: totals after");
    agg_require_equal(schema, &before, &after, "nooverwrite: totals unchanged");

    verify_against_oracle(txn, dbi, "nooverwrite");
    CHECK(mdb_txn_commit(txn), "nooverwrite: commit");

    mdb_dbi_close(env, dbi);
    mdb_env_close(env);
}

static void test_dupsort_put_nodupdata_no_agg_change(void)
{
    char dir[256];
    make_test_dir(dir, sizeof(dir), "t21_nodupdata");
    MDB_env *env = open_env_dir(dir, 8, 64u * 1024u * 1024u, 1);

    MDB_txn *txn = NULL;
    MDB_dbi dbi;

    CHECK(mdb_txn_begin(env, NULL, 0, &txn), "nodupdata: begin");
    CHECK(mdb_dbi_open(txn, "dups", MDB_CREATE | MDB_DUPSORT | AGG_SCHEMA, &dbi), "nodupdata: open");

    const char *ks = "HOT";
    MDB_val k = { strlen(ks), (void *)ks };
    uint8_t vbuf[MDB_HASH_SIZE + 64];
    MDB_val d;
    make_val_with_hash_salted(vbuf, sizeof(vbuf), "D", 1, 0, &d);

    CHECK(mdb_put(txn, dbi, &k, &d, 0), "nodupdata: put first");
    DBG_CHECK(txn, dbi, "nodupdata: dbg");

    unsigned schema = 0;
    CHECK(mdb_agg_info(txn, dbi, &schema), "nodupdata: agg_info");

    MDB_agg before, after;
    CHECK(mdb_agg_totals(txn, dbi, &before), "nodupdata: totals before");

    int rc = mdb_put(txn, dbi, &k, &d, MDB_NODUPDATA);
    expect_rc(rc, MDB_KEYEXIST, "nodupdata: MDB_NODUPDATA keyexists");

    CHECK(mdb_agg_totals(txn, dbi, &after), "nodupdata: totals after");
    agg_require_equal(schema, &before, &after, "nodupdata: totals unchanged");

    verify_against_oracle(txn, dbi, "nodupdata");
    CHECK(mdb_txn_commit(txn), "nodupdata: commit");

    mdb_dbi_close(env, dbi);
    mdb_env_close(env);
}

static void make_ordered_dup_val_micro(uint8_t *buf, size_t cap, unsigned dupidx, unsigned tweak, MDB_val *out)
{
    /* Make DUPSORT values where the first byte defines ordering between dups.
     * We can alter bytes [1..] to change HASHSUM without changing sort position.
     */
    REQUIRE(cap >= (size_t)MDB_HASH_SIZE + 32u, "buffer too small for ordered dup val");
    memset(buf, 0, MDB_HASH_SIZE);
    buf[0] = (uint8_t)dupidx;
    buf[1] = (uint8_t)tweak;
    int n = snprintf((char *)buf + MDB_HASH_SIZE, cap - (size_t)MDB_HASH_SIZE,
                     "dup:%03u:%03u", dupidx, tweak);
    REQUIRE(n > 0, "snprintf failed");
    out->mv_data = buf;
    out->mv_size = (size_t)MDB_HASH_SIZE + (size_t)n;
}

static void test_cursor_current_update_hashsum_delta(void)
{
    char dir[256];
    make_test_dir(dir, sizeof(dir), "t22_cursor_current");
    MDB_env *env = open_env_dir(dir, 16, 96u * 1024u * 1024u, 1);

    MDB_txn *txn = NULL;
    MDB_dbi dbp, dbd;

    CHECK(mdb_txn_begin(env, NULL, 0, &txn), "curcur: begin");
    CHECK(mdb_dbi_open(txn, "plain", MDB_CREATE | AGG_SCHEMA, &dbp), "curcur: open plain");
    CHECK(mdb_dbi_open(txn, "dups", MDB_CREATE | MDB_DUPSORT | AGG_SCHEMA, &dbd), "curcur: open dups");

    char kbuf[64];
    uint8_t vbuf[MDB_HASH_SIZE + 128];

    /* plain: 20 keys */
    for (unsigned i = 0; i < 20; i++) {
        make_key(kbuf, sizeof(kbuf), "K", i);
        MDB_val k = { strlen(kbuf), kbuf };
        MDB_val d;
        make_val_with_hash_salted(vbuf, sizeof(vbuf), "P", i, 0, &d);
        CHECK(mdb_put(txn, dbp, &k, &d, 0), "curcur: put plain");
        DBG_CHECK(txn, dbp, "curcur: dbg plain");
    }

    /* dupsort: one key with 10 ordered dups */
    {
        const char *ks = "HOT";
        MDB_val k = { strlen(ks), (void *)ks };
        uint8_t dbuf[MDB_HASH_SIZE + 64];
        for (unsigned i = 0; i < 10; i++) {
            MDB_val d;
            make_ordered_dup_val_micro(dbuf, sizeof(dbuf), i, 0, &d);
            CHECK(mdb_put(txn, dbd, &k, &d, 0), "curcur: put dups");
            DBG_CHECK(txn, dbd, "curcur: dbg dups");
        }
    }

    unsigned schema_p = 0, schema_d = 0;
    CHECK(mdb_agg_info(txn, dbp, &schema_p), "curcur: agg_info plain");
    CHECK(mdb_agg_info(txn, dbd, &schema_d), "curcur: agg_info dups");

    /* plain: update one record via MDB_CURRENT and verify hashsum delta */
    if (schema_p & MDB_AGG_HASHSUM) {
        MDB_agg before, after;
        CHECK(mdb_agg_totals(txn, dbp, &before), "curcur: totals before plain");

        MDB_cursor *cur = NULL;
        CHECK(mdb_cursor_open(txn, dbp, &cur), "curcur: cursor_open plain");

        const char *ks = "K00000005";
        MDB_val k = { strlen(ks), (void *)ks };
        MDB_val d = { 0, NULL };
        CHECK(mdb_cursor_get(cur, &k, &d, MDB_SET_KEY), "curcur: cursor set_key");

        uint8_t oldhs[MDB_HASH_SIZE];
        CHECK(mdb_hashsum_extract_bytes(d.mv_data, d.mv_size, 0, oldhs), "curcur: extract oldhs");

        MDB_val nd;
        make_val_with_hash_salted(vbuf, sizeof(vbuf), "P", 5, 0x2222222222222222ULL, &nd);
        uint8_t newhs[MDB_HASH_SIZE];
        CHECK(mdb_hashsum_extract_bytes(nd.mv_data, nd.mv_size, 0, newhs), "curcur: extract newhs");

        CHECK(mdb_cursor_put(cur, &k, &nd, MDB_CURRENT), "curcur: cursor_put MDB_CURRENT plain");
        DBG_CHECK(txn, dbp, "curcur: dbg plain after current");

        CHECK(mdb_agg_totals(txn, dbp, &after), "curcur: totals after plain");
        if (after.mv_agg_entries != before.mv_agg_entries || after.mv_agg_keys != before.mv_agg_keys)
            REQUIRE(0, "curcur: plain current changed entries/keys");

        uint8_t expect[MDB_HASH_SIZE];
        memcpy(expect, before.mv_agg_hashes, MDB_HASH_SIZE);
        mdb_hashsum_sub(expect, oldhs);
        mdb_hashsum_add(expect, newhs);
        if (memcmp(expect, after.mv_agg_hashes, MDB_HASH_SIZE) != 0)
            REQUIRE(0, "curcur: plain current hashsum delta mismatch");

        mdb_cursor_close(cur);
    }

    /* dupsort: update one duplicate via MDB_CURRENT and verify hashsum delta */
    if (schema_d & MDB_AGG_HASHSUM) {
        MDB_agg before, after;
        CHECK(mdb_agg_totals(txn, dbd, &before), "curcur: totals before dups");

        MDB_cursor *cur = NULL;
        CHECK(mdb_cursor_open(txn, dbd, &cur), "curcur: cursor_open dups");

        const char *ks = "HOT";
        MDB_val k = { strlen(ks), (void *)ks };

        uint8_t obuf[MDB_HASH_SIZE + 64];
        uint8_t nbuf[MDB_HASH_SIZE + 64];
        MDB_val od, nd;
        make_ordered_dup_val_micro(obuf, sizeof(obuf), 5, 0, &od);
        make_ordered_dup_val_micro(nbuf, sizeof(nbuf), 5, 7, &nd);

        /* position at (k,od) */
        CHECK(mdb_cursor_get(cur, &k, &od, MDB_GET_BOTH), "curcur: cursor get_both");

        uint8_t oldhs[MDB_HASH_SIZE];
        CHECK(mdb_hashsum_extract_bytes(od.mv_data, od.mv_size, 0, oldhs), "curcur: extract oldhs dup");
        uint8_t newhs[MDB_HASH_SIZE];
        CHECK(mdb_hashsum_extract_bytes(nd.mv_data, nd.mv_size, 0, newhs), "curcur: extract newhs dup");

        CHECK(mdb_cursor_put(cur, &k, &nd, MDB_CURRENT), "curcur: cursor_put MDB_CURRENT dups");
        DBG_CHECK(txn, dbd, "curcur: dbg dups after current");

        CHECK(mdb_agg_totals(txn, dbd, &after), "curcur: totals after dups");
        if (after.mv_agg_entries != before.mv_agg_entries || after.mv_agg_keys != before.mv_agg_keys)
            REQUIRE(0, "curcur: dups current changed entries/keys");

        uint8_t expect[MDB_HASH_SIZE];
        memcpy(expect, before.mv_agg_hashes, MDB_HASH_SIZE);
        mdb_hashsum_sub(expect, oldhs);
        mdb_hashsum_add(expect, newhs);
        if (memcmp(expect, after.mv_agg_hashes, MDB_HASH_SIZE) != 0)
            REQUIRE(0, "curcur: dups current hashsum delta mismatch");

        mdb_cursor_close(cur);
    }

    verify_against_oracle(txn, dbp, "curcur plain");
    verify_against_oracle(txn, dbd, "curcur dups");

    CHECK(mdb_txn_commit(txn), "curcur: commit");

    mdb_dbi_close(env, dbp);
    mdb_dbi_close(env, dbd);
    mdb_env_close(env);
}

static void test_nested_txn_abort_commit(void)
{
    char dir[256];
    make_test_dir(dir, sizeof(dir), "t23_nested_txn");
    MDB_env *env = open_env_dir(dir, 8, 128u * 1024u * 1024u, 1);

    MDB_txn *ptxn = NULL;
    MDB_dbi dbi;

    CHECK(mdb_txn_begin(env, NULL, 0, &ptxn), "nested: parent begin");
    CHECK(mdb_dbi_open(ptxn, "plain", MDB_CREATE | AGG_SCHEMA, &dbi), "nested: open");

    char kbuf[64];
    uint8_t vbuf[MDB_HASH_SIZE + 128];

    /* baseline */
    for (unsigned i = 0; i < 200; i++) {
        make_key(kbuf, sizeof(kbuf), "K", i);
        MDB_val k = { strlen(kbuf), kbuf };
        MDB_val d;
        make_val_with_hash_salted(vbuf, sizeof(vbuf), "B", i, 0, &d);
        CHECK(mdb_put(ptxn, dbi, &k, &d, 0), "nested: put baseline");
        DBG_CHECK(ptxn, dbi, "nested: dbg baseline");
    }
    verify_against_oracle(ptxn, dbi, "nested baseline");

    /* child txn - abort */
    {
        MDB_txn *ctxn = NULL;
        CHECK(mdb_txn_begin(env, ptxn, 0, &ctxn), "nested: child begin (abort)");

        /* inserts */
        for (unsigned i = 400; i < 450; i++) {
            make_key(kbuf, sizeof(kbuf), "K", i);
            MDB_val k = { strlen(kbuf), kbuf };
            MDB_val d;
            make_val_with_hash_salted(vbuf, sizeof(vbuf), "C", i, 0, &d);
            CHECK(mdb_put(ctxn, dbi, &k, &d, 0), "nested: child put");
            DBG_CHECK(ctxn, dbi, "nested: child dbg put");
        }

        /* deletes */
        for (unsigned i = 80; i < 100; i++) {
            make_key(kbuf, sizeof(kbuf), "K", i);
            MDB_val k = { strlen(kbuf), kbuf };
            CHECK(mdb_del(ctxn, dbi, &k, NULL), "nested: child del");
            DBG_CHECK(ctxn, dbi, "nested: child dbg del");
        }

        /* overwrite one key (hashsum delta) */
        make_key(kbuf, sizeof(kbuf), "K", 10);
        MDB_val k = { strlen(kbuf), kbuf };
        MDB_val d;
        make_val_with_hash_salted(vbuf, sizeof(vbuf), "B", 10, 0x7777777777777777ULL, &d);
        CHECK(mdb_put(ctxn, dbi, &k, &d, 0), "nested: child overwrite");
        DBG_CHECK(ctxn, dbi, "nested: child dbg overwrite");

        verify_against_oracle(ctxn, dbi, "nested child (abort) view");
        mdb_txn_abort(ctxn);
    }

    /* parent must see baseline unchanged */
    verify_against_oracle(ptxn, dbi, "nested parent after child abort");

    /* child txn - commit */
    {
        MDB_txn *ctxn = NULL;
        CHECK(mdb_txn_begin(env, ptxn, 0, &ctxn), "nested: child begin (commit)");

        /* inserts */
        for (unsigned i = 500; i < 520; i++) {
            make_key(kbuf, sizeof(kbuf), "K", i);
            MDB_val k = { strlen(kbuf), kbuf };
            MDB_val d;
            make_val_with_hash_salted(vbuf, sizeof(vbuf), "C", i, 0, &d);
            CHECK(mdb_put(ctxn, dbi, &k, &d, 0), "nested: child2 put");
            DBG_CHECK(ctxn, dbi, "nested: child2 dbg put");
        }

        /* deletes */
        for (unsigned i = 20; i < 40; i++) {
            make_key(kbuf, sizeof(kbuf), "K", i);
            MDB_val k = { strlen(kbuf), kbuf };
            CHECK(mdb_del(ctxn, dbi, &k, NULL), "nested: child2 del");
            DBG_CHECK(ctxn, dbi, "nested: child2 dbg del");
        }

        /* overwrite one key */
        make_key(kbuf, sizeof(kbuf), "K", 15);
        MDB_val k = { strlen(kbuf), kbuf };
        MDB_val d;
        make_val_with_hash_salted(vbuf, sizeof(vbuf), "B", 15, 0x8888888888888888ULL, &d);
        CHECK(mdb_put(ctxn, dbi, &k, &d, 0), "nested: child2 overwrite");
        DBG_CHECK(ctxn, dbi, "nested: child2 dbg overwrite");

        verify_against_oracle(ctxn, dbi, "nested child (commit) view");
        CHECK(mdb_txn_commit(ctxn), "nested: child2 commit");
    }

    /* parent should now see child changes */
    verify_against_oracle(ptxn, dbi, "nested parent after child commit");

    CHECK(mdb_txn_commit(ptxn), "nested: parent commit");

    /* verify on reopened read txn */
    MDB_txn *rtxn = NULL;
    CHECK(mdb_txn_begin(env, NULL, MDB_RDONLY, &rtxn), "nested: ro begin");
    verify_against_oracle(rtxn, dbi, "nested ro view");
    mdb_txn_abort(rtxn);

    mdb_dbi_close(env, dbi);
    mdb_env_close(env);
}


/*
 * Regression: stale named-DB snapshot used by mdb_dbi_open() after an aborted write txn.
 *
 * Background:
 *   - DBI handles are environment-global, but txn->mt_dbs[dbi] is a txn-local snapshot
 *     (root/depth/flags/etc.). For named DBIs, LMDB marks that snapshot DB_STALE at
 *     txn start and expects it to be refreshed lazily via a cursor walk in MAIN_DBI.
 *   - The multi-aggregate extension performs aggregate-root validation during mdb_dbi_open().
 *     Before Patch 22, the validation could consult txn->mt_dbs[dbi].md_root before the
 *     DB_STALE snapshot was refreshed, so a reused write-txn object (me_txn0) could carry
 *     a stale md_root from a previous *aborted* txn. That stale root pgno may be >= mt_next_pgno
 *     of the new txn, leading to MDB_PAGE_NOTFOUND inside mdb_page_get().
 *
 * This test forces an aborted txn to allocate a new root (depth > 1) and then immediately
 * begins a new write txn that opens the same named DB. Without the "refresh stale snapshot
 * before agg-root checks" fix, this reliably fails with MDB_PAGE_NOTFOUND.
 */
static void test_stale_named_dbi_open_after_abort(void)
{
    char dir[256];
    make_test_dir(dir, sizeof(dir), "t24_stale_named_dbi_open");
    MDB_env *env = open_env_dir(dir, 8, 64u * 1024u * 1024u, 1);

    MDB_txn *txn = NULL;
    MDB_dbi dbi;

    char kbuf[96];
    uint8_t vbuf[MDB_HASH_SIZE + 256];
    uint8_t bigbuf[MDB_HASH_SIZE + 1024];

    /* txn1: create the named DB and commit a small baseline record. */
    CHECK(mdb_txn_begin(env, NULL, 0, &txn), "stale: txn1 begin");
    CHECK(mdb_dbi_open(txn, "plain", MDB_CREATE | AGG_SCHEMA, &dbi), "stale: txn1 open");
    {
        make_key(kbuf, sizeof(kbuf), "K", 0);
        MDB_val k = { strlen(kbuf), kbuf };
        MDB_val d;
        make_val_with_hash_salted(vbuf, sizeof(vbuf), "B", 0, 0, &d);
        CHECK(mdb_put(txn, dbi, &k, &d, 0), "stale: txn1 put baseline");
        DBG_CHECK(txn, dbi, "stale: txn1 dbg baseline");
    }
    CHECK(mdb_txn_commit(txn), "stale: txn1 commit");

    /*
     * txn2 (abort): insert medium-sized in-page values until the sub-DB depth increases (>1),
     * i.e. the root leaf splits and a new root page is allocated. Then abort. The env will
     * likely reuse the same write-txn object for the next txn, preserving txn-local snapshots.
     */
    CHECK(mdb_txn_begin(env, NULL, 0, &txn), "stale: txn2 begin");
    CHECK(mdb_dbi_open(txn, "plain", 0, &dbi), "stale: txn2 open");

    MDB_stat st;
    memset(&st, 0, sizeof(st));

    unsigned inserted = 0;
    for (unsigned i = 1; i < 5000; i++) {
        make_key(kbuf, sizeof(kbuf), "SPLIT", i);
        MDB_val k = { strlen(kbuf), kbuf };
        MDB_val d;
        /* extra=768 keeps values in-page (no overflow), but large enough to force splits quickly. */
        make_big_val_with_hash(bigbuf, sizeof(bigbuf), i, 768u, &d);
        CHECK(mdb_put(txn, dbi, &k, &d, 0), "stale: txn2 put (force split)");
        inserted++;

        if ((i % 25u) == 0) {
            CHECK(mdb_stat(txn, dbi, &st), "stale: txn2 stat");
            if (st.ms_depth > 1)
                break;
        }
    }
    CHECK(mdb_stat(txn, dbi, &st), "stale: txn2 final stat");

    REQUIRE(st.ms_depth > 1, "stale: could not force root split (depth > 1); increase loop or value size");
    REQUIRE(inserted > 0, "stale: inserted == 0 (unexpected)");

    mdb_txn_abort(txn);

    /*
     * txn3: the regression trigger. mdb_dbi_open() must succeed. On the buggy build it can
     * fail with MDB_PAGE_NOTFOUND due to stale txn->mt_dbs[dbi].md_root being consulted.
     */
    CHECK(mdb_txn_begin(env, NULL, 0, &txn), "stale: txn3 begin");
    CHECK(mdb_dbi_open(txn, "plain", 0, &dbi), "stale: txn3 open (after abort)");
    mdb_txn_abort(txn);

    mdb_dbi_close(env, dbi);
    mdb_env_close(env);
}



/* ------------------------------ API contract / edge behavior checks ------------------------------ */

static void test_api_contracts_prefix_range(void)
{
    char dir[256];
    make_test_dir(dir, sizeof(dir), "t25_api_contracts");
    MDB_env *env = open_env_dir(dir, 16, 64u * 1024u * 1024u, 1);

    MDB_txn *txn = NULL;
    MDB_dbi plain, dups, noagg;

    char kbuf[64];
    uint8_t vbuf[MDB_HASH_SIZE + 256];

    CHECK(mdb_txn_begin(env, NULL, 0, &txn), "contracts: begin");
    CHECK(mdb_dbi_open(txn, "plain", MDB_CREATE | AGG_SCHEMA, &plain), "contracts: open plain");
    CHECK(mdb_dbi_open(txn, "dups", MDB_CREATE | MDB_DUPSORT | AGG_SCHEMA, &dups), "contracts: open dups");
    CHECK(mdb_dbi_open(txn, "noagg", MDB_CREATE, &noagg), "contracts: open noagg");

    /* plain: a few keys */
    for (unsigned i = 0; i < 8; i++) {
        make_key(kbuf, sizeof(kbuf), "K", i);
        MDB_val k = { strlen(kbuf), kbuf };
        MDB_val d;
        make_val_with_hash_salted(vbuf, sizeof(vbuf), "P", i, 0, &d);
        CHECK(mdb_put(txn, plain, &k, &d, 0), "contracts: put plain");
    }

    /* dups: a few keys with a few dups each */
    for (unsigned i = 0; i < 4; i++) {
        make_key(kbuf, sizeof(kbuf), "K", i);
        MDB_val k = { strlen(kbuf), kbuf };
        for (unsigned j = 0; j < 4; j++) {
            MDB_val d;
            make_val_with_hash_salted(vbuf, sizeof(vbuf), "D", (i<<8) ^ j, 0, &d);
            CHECK(mdb_put(txn, dups, &k, &d, 0), "contracts: put dups");
        }
    }

    CHECK(mdb_txn_commit(txn), "contracts: commit");

    CHECK(mdb_txn_begin(env, NULL, MDB_RDONLY, &txn), "contracts: ro begin");
    CHECK(mdb_dbi_open(txn, "plain", 0, &plain), "contracts: ro open plain");
    CHECK(mdb_dbi_open(txn, "dups", 0, &dups), "contracts: ro open dups");
    CHECK(mdb_dbi_open(txn, "noagg", 0, &noagg), "contracts: ro open noagg");

    /* noagg: all aggregate calls must fail with MDB_INCOMPATIBLE */
    {
        MDB_agg a;
        expect_rc(mdb_agg_totals(txn, noagg, &a), MDB_INCOMPATIBLE, "contracts: totals(noagg)");
        expect_rc(mdb_agg_prefix(txn, noagg, &(MDB_val){1,(void*)"K"}, NULL, 0, &a), MDB_INCOMPATIBLE, "contracts: prefix(noagg)");
        expect_rc(mdb_agg_prefix(txn, noagg, &(MDB_val){1,(void*)"K"}, NULL, 0, &a), MDB_INCOMPATIBLE, "contracts: prefix(noagg)");
        expect_rc(mdb_agg_range(txn, noagg, NULL, NULL, NULL, NULL, 0, &a), MDB_INCOMPATIBLE, "contracts: range(noagg)");
        expect_rc(mdb_agg_range(txn, noagg, NULL, NULL, NULL, NULL, 0, &a), MDB_INCOMPATIBLE, "contracts: range(noagg)");
    }

    /* invalid args/flags: EINVAL */
    {
        MDB_agg a;
        MDB_val k = { 1, (void*)"K" };
        expect_rc(mdb_agg_prefix(txn, plain, NULL, NULL, 0, &a), EINVAL, "contracts: prefix(NULL key)");
        expect_rc(mdb_agg_prefix(txn, plain, &k, NULL, 0x80000000u, &a), EINVAL, "contracts: prefix(bad flags)");
        expect_rc(mdb_agg_range(txn, plain, NULL, NULL, NULL, NULL, 0x80000000u, &a), EINVAL, "contracts: range(bad flags)");
    }

    /* plain DB: prefix(key,data) must match prefix(key) (data ignored / fallback). */
    {
        MDB_agg p2, p1;
        make_key(kbuf, sizeof(kbuf), "K", 3);
        MDB_val k = { strlen(kbuf), kbuf };
        MDB_val dummy;
        make_val_with_hash_salted(vbuf, sizeof(vbuf), "X", 999, 0x1234, &dummy);
        CHECK(mdb_agg_prefix(txn, plain, &k, &dummy, 0, &p2), "contracts: plain prefix");
        CHECK(mdb_agg_prefix(txn, plain, &k, NULL, 0, &p1), "contracts: plain prefix");
        agg_require_equal(p1.mv_flags, &p2, &p1, "contracts: plain prefix == prefix");
    }

    /* plain DB: range(low,data,high,data) must match range(low,high). */
    {
        MDB_agg r2, r1;
        MDB_val k0 = { 9, (void*)"K00000001" };
        MDB_val k1 = { 9, (void*)"K00000006" };
        MDB_val dummy;
        make_val_with_hash_salted(vbuf, sizeof(vbuf), "Y", 123, 0, &dummy);
        unsigned rf = MDB_RANGE_LOWER_INCL | MDB_RANGE_UPPER_INCL;
        CHECK(mdb_agg_range(txn, plain, &k0, &dummy, &k1, &dummy, rf, &r2), "contracts: plain range");
        CHECK(mdb_agg_range(txn, plain, &k0, NULL, &k1, NULL, rf, &r1), "contracts: plain range");
        agg_require_equal(r1.mv_flags, &r2, &r1, "contracts: plain range == range");
    }

    /* DUPSORT: range(key-only) must match range(key-only). */
    {
        MDB_agg r2, r1;
        MDB_val k0 = { 9, (void*)"K00000001" };
        MDB_val k1 = { 9, (void*)"K00000003" };
        unsigned rf = MDB_RANGE_LOWER_INCL | MDB_RANGE_UPPER_INCL;
        CHECK(mdb_agg_range(txn, dups, &k0, NULL, &k1, NULL, rf, &r2), "contracts: dups range key-only");
        CHECK(mdb_agg_range(txn, dups, &k0, NULL, &k1, NULL, rf, &r1), "contracts: dups range key-only");
        agg_require_equal(r1.mv_flags, &r2, &r1, "contracts: dups range(key-only)==range");
    }

    /* Data with NULL key is currently ignored; enforce self-consistency. */
    {
        MDB_agg a, b;
        MDB_val hk = { 9, (void*)"K00000002" };
        MDB_val hd;
        make_val_with_hash_salted(vbuf, sizeof(vbuf), "D", (2u<<8) ^ 2u, 0, &hd);

        unsigned rf = MDB_RANGE_UPPER_INCL;
        CHECK(mdb_agg_range(txn, dups, NULL, &hd, &hk, &hd, rf, &a), "contracts: range(NULL,hd, hk,hd)");
        CHECK(mdb_agg_range(txn, dups, NULL, NULL, &hk, &hd, rf, &b), "contracts: range(NULL,NULL, hk,hd)");
        agg_require_equal(b.mv_flags, &a, &b, "contracts: NULL-key data ignored (low)");
    }

    mdb_txn_abort(txn);
    mdb_dbi_close(env, plain);
    mdb_dbi_close(env, dups);
    mdb_dbi_close(env, noagg);
    mdb_env_close(env);
}



static void test_api_window_rank_seek(void)
{
    char dir[256];
    make_test_dir(dir, sizeof(dir), "t26_window_rank_seek");
    MDB_env *env = open_env_dir(dir, 16, 64u * 1024u * 1024u, 1);

    MDB_txn *txn = NULL;
    MDB_dbi plain, dups;

    char kbuf[64];
    uint8_t vbuf[MDB_HASH_SIZE + 256];

    CHECK(mdb_txn_begin(env, NULL, 0, &txn), "window: begin");
    CHECK(mdb_dbi_open(txn, "plain", MDB_CREATE | AGG_SCHEMA, &plain), "window: open plain");
    CHECK(mdb_dbi_open(txn, "dups", MDB_CREATE | MDB_DUPSORT | AGG_SCHEMA, &dups), "window: open dups");

    /* Build deterministic content large enough for multiple pages. */
    for (unsigned i = 0; i < 64; i++) {
        make_key(kbuf, sizeof(kbuf), "K", i);
        MDB_val k = { strlen(kbuf), kbuf };
        MDB_val d;
        make_val_with_hash_salted(vbuf, sizeof(vbuf), "P", i, 0, &d);
        CHECK(mdb_put(txn, plain, &k, &d, 0), "window: put plain");
    }

    for (unsigned i = 0; i < 32; i++) {
        make_key(kbuf, sizeof(kbuf), "K", i);
        MDB_val k = { strlen(kbuf), kbuf };
        for (unsigned j = 0; j < 4; j++) {
            MDB_val d;
            make_val_with_hash_salted(vbuf, sizeof(vbuf), "D", (i<<8) ^ j, 0, &d);
            CHECK(mdb_put(txn, dups, &k, &d, 0), "window: put dups");
        }
    }

    CHECK(mdb_txn_commit(txn), "window: commit");

    CHECK(mdb_txn_begin(env, NULL, MDB_RDONLY, &txn), "window: ro begin");
    CHECK(mdb_dbi_open(txn, "plain", 0, &plain), "window: ro open plain");
    CHECK(mdb_dbi_open(txn, "dups", 0, &dups), "window: ro open dups");

    MDB_dbi dbis[2] = { plain, dups };
    const char *dbnames[2] = { "plain", "dups" };

    for (unsigned which = 0; which < 2; which++) {
        MDB_dbi dbi = dbis[which];

        RecVec vv;
        recvec_init(&vv);
        oracle_build(txn, dbi, &vv);

        /* Key-only window bounds used by LMDBAggSlice: [low, high) with flags. */
        MDB_val low = { 9, (void*)"K00000010" };
        MDB_val high = { 9, (void*)"K00000020" };

        unsigned rfs[] = {
            MDB_RANGE_LOWER_INCL,                           /* [low, high) */
            MDB_RANGE_LOWER_INCL | MDB_RANGE_UPPER_INCL,    /* [low, high] */
            0u,                                             /* (low, high) */
            MDB_RANGE_UPPER_INCL                            /* (low, high] */
        };

        for (unsigned rfi = 0; rfi < sizeof(rfs)/sizeof(rfs[0]); rfi++) {
            unsigned rf = rfs[rfi];

            size_t exp_lo = 0, exp_hi = 0;
            oracle_window_bounds_key_only(txn, dbi, &vv, &low, &high, rf, &exp_lo, &exp_hi);

            /* Step 2 contract: window_rank must initialize a zeroed window too. */
            MDB_agg_window w0;
            memset(&w0, 0, sizeof(w0));
            {
                MDB_val probe = { 9, (void*)"K00000000" };
                uint64_t rel = 12345;
                CHECK(mdb_agg_window_rank(txn, dbi,
                                          &low, NULL,
                                          &high, NULL,
                                          rf,
                                          &w0,
                                          &probe, NULL,
                                          &rel),
                      "window_rank init");
                REQUIRE((size_t)w0.mv_abs_lo == exp_lo, "window_rank: abs_lo mismatch");
                REQUIRE((size_t)w0.mv_abs_hi == exp_hi, "window_rank: abs_hi mismatch");
                REQUIRE((size_t)w0.mv_total_entries == vv.len, "window_rank: total_entries mismatch");
            }

            MDB_agg_window w;
            memset(&w, 0, sizeof(w));

            /* Choose a non-trivial relative interval inside the window. */
            size_t win_size = (exp_hi >= exp_lo) ? (exp_hi - exp_lo) : 0;
            uint64_t rel_begin = 0, rel_end = 0;
            if (win_size > 0) {
                rel_begin = (uint64_t)(win_size / 3);
                size_t max_len = win_size - (size_t)rel_begin;
                size_t want = (max_len > 10) ? 10 : max_len;
                rel_end = rel_begin + (uint64_t)want;
            }

            MDB_agg got;
            CHECK(mdb_agg_window_fingerprint(txn, dbi,
                                             &low, NULL,
                                             &high, NULL,
                                             rf,
                                             &w,
                                             rel_begin, rel_end,
                                             &got),
                  "window_fingerprint");

            REQUIRE((size_t)w.mv_abs_lo == exp_lo, "window_fingerprint: abs_lo mismatch");
            REQUIRE((size_t)w.mv_abs_hi == exp_hi, "window_fingerprint: abs_hi mismatch");
            REQUIRE((size_t)w.mv_total_entries == vv.len, "window_fingerprint: total_entries mismatch");

            MDB_agg exp;
            oracle_rank_range_agg(&vv, exp_lo + (size_t)rel_begin, exp_lo + (size_t)rel_end, &exp);
            agg_require_equal(exp.mv_flags, &got, &exp, "window_fingerprint: rank-range agg mismatch");

            /* Step 2: window_rank(rel) must match oracle lower-bound within window (key-only). */
            {
                const char *probes[] = {
                    "K00000000",
                    "K00000010",
                    "K00000011",
                    "K00000020",
                    "K00000021",
                    "K00000099"
                };
                for (unsigned pi = 0; pi < sizeof(probes)/sizeof(probes[0]); pi++) {
                    MDB_val pk = { 9, (void*)probes[pi] };
                    uint64_t rel = 0;
                    CHECK(mdb_agg_window_rank(txn, dbi,
                                               &low, NULL,
                                               &high, NULL,
                                               rf,
                                               &w,
                                               &pk, NULL,
                                               &rel),
                          "window_rank");
                    uint64_t exp_rel = oracle_window_rel_rank_key_only(txn, dbi, &vv, exp_lo, exp_hi, &pk);
                    if (rel != exp_rel) {
                        fprintf(stderr, "window_rank mismatch db=%s rf=0x%x probe=%s got=%llu exp=%llu\n",
                                dbnames[which], rf, probes[pi],
                                (unsigned long long)rel, (unsigned long long)exp_rel);
                        REQUIRE(0, "window_rank mismatch");
                    }
                }
            }

            /* Step 3: cursor seek by absolute rank must match oracle and leave cursor ready for MDB_NEXT. */
            {
                MDB_cursor *mc = NULL;
                CHECK(mdb_cursor_open(txn, dbi, &mc), "cursor_open");

                size_t picks[6];
                size_t np = 0;

                if (vv.len > 0) picks[np++] = 0;
                if (vv.len > 1) picks[np++] = 1;
                if (exp_lo < vv.len) picks[np++] = exp_lo;
                if (exp_lo + 1 < vv.len) picks[np++] = exp_lo + 1;
                if (exp_hi > 0 && exp_hi - 1 < vv.len) picks[np++] = exp_hi - 1;
                if (vv.len > 0) picks[np++] = vv.len - 1;

                for (size_t qi = 0; qi < np; qi++) {
                    size_t rank = picks[qi];
                    MDB_val k = {0}, d = {0};
                    int rc = mdb_agg_cursor_seek_rank(mc, (uint64_t)rank, &k, &d);
                    REQUIRE(rc == MDB_SUCCESS, "mdb_agg_cursor_seek_rank failed");
                    require_mdbvals_eq_rec(&k, &d, &vv.v[rank], "cursor_seek_rank record mismatch");

                    if (rank + 1 < vv.len) {
                        MDB_val k2 = k, d2 = d;
                        CHECK(mdb_cursor_get(mc, &k2, &d2, MDB_NEXT), "cursor_next after seek_rank");
                        require_mdbvals_eq_rec(&k2, &d2, &vv.v[rank + 1], "cursor_next after seek_rank mismatch");
                    } else {
                        MDB_val k2 = k, d2 = d;
                        expect_rc(mdb_cursor_get(mc, &k2, &d2, MDB_NEXT), MDB_NOTFOUND, "cursor_next on last");
                    }
                }

                /* Out-of-range must return MDB_NOTFOUND. */
                {
                    MDB_val k = {0}, d = {0};
                    expect_rc(mdb_agg_cursor_seek_rank(mc, (uint64_t)vv.len, &k, &d), MDB_NOTFOUND, "seek_rank out-of-range");
                }

                mdb_cursor_close(mc);
            }
        }

        recvec_free(&vv);
    }

    mdb_txn_abort(txn);
    mdb_dbi_close(env, plain);
    mdb_dbi_close(env, dups);
    mdb_env_close(env);
}

/* ------------------------------ DUPSORT subpage/subDB helpers ------------------------------ */

static size_t dups_count_for_key(MDB_txn *txn, MDB_dbi dbi, const MDB_val *key)
{
    MDB_cursor *mc = NULL;
    MDB_val k = *key, d;
    mdb_size_t cnt = 0;
    int rc;

    CHECK(mdb_cursor_open(txn, dbi, &mc), "dups_count: cursor_open");
    rc = mdb_cursor_get(mc, &k, &d, MDB_SET_KEY);
    if (rc == MDB_NOTFOUND) {
        mdb_cursor_close(mc);
        return 0;
    }
    CHECK(rc, "dups_count: cursor_set_key");
    CHECK(mdb_cursor_count(mc, &cnt), "dups_count: cursor_count");
    mdb_cursor_close(mc);
    return (size_t)cnt;
}

static void dups_hashsum_for_key(MDB_txn *txn, MDB_dbi dbi, const MDB_val *key,
                                uint8_t acc[MDB_HASH_SIZE], size_t *entries_out)
{
    MDB_cursor *mc = NULL;
    MDB_val k = *key, d;
    int rc;

    memset(acc, 0, MDB_HASH_SIZE);
    if (entries_out) *entries_out = 0;

    CHECK(mdb_cursor_open(txn, dbi, &mc), "dups_hs: cursor_open");
    rc = mdb_cursor_get(mc, &k, &d, MDB_SET_KEY);
    if (rc == MDB_NOTFOUND) {
        mdb_cursor_close(mc);
        return;
    }
    CHECK(rc, "dups_hs: cursor_set_key");

    /* Iterate all duplicates for this key. */
    rc = mdb_cursor_get(mc, &k, &d, MDB_FIRST_DUP);
    CHECK(rc, "dups_hs: first_dup");
    do {
        REQUIRE(d.mv_size >= (size_t)MDB_HASH_SIZE, "dups_hs: value too small for hashsum");
        mdb_hashsum_add(acc, (const uint8_t *)d.mv_data);
        if (entries_out) (*entries_out)++;
        rc = mdb_cursor_get(mc, &k, &d, MDB_NEXT_DUP);
    } while (rc == MDB_SUCCESS);

    REQUIRE(rc == MDB_NOTFOUND, "dups_hs: next_dup unexpected rc");
    mdb_cursor_close(mc);
}

/* Key-local record-range sanity check:
 * range((key,-inf) .. (key,+inf)) must touch exactly that dupset. */
static void check_key_local_range(MDB_txn *txn, MDB_dbi dbi, const MDB_val *key,
                                   const uint8_t hi_sentinel[MDB_HASH_SIZE], const char *msg)
{
    MDB_agg got;
    MDB_val hi = { MDB_HASH_SIZE, (void *)hi_sentinel };
    unsigned rf = MDB_RANGE_LOWER_INCL | MDB_RANGE_UPPER_INCL;

    CHECK(mdb_agg_range(txn, dbi, key, NULL, key, &hi, rf, &got), msg);
    agg_expect_flags(msg, got.mv_flags, AGG_SCHEMA);

    size_t exp_entries = 0;
    uint8_t exp_hs[MDB_HASH_SIZE];
    dups_hashsum_for_key(txn, dbi, key, exp_hs, &exp_entries);
    size_t exp_keys = exp_entries ? 1u : 0u;

    if ((AGG_SCHEMA & MDB_AGG_ENTRIES) && got.mv_agg_entries != exp_entries) {
        fprintf(stderr, "%s: entries mismatch a=%" PRIu64 " b=%zu\n",
                msg, (uint64_t)got.mv_agg_entries, exp_entries);
        exit(EXIT_FAILURE);
    }
    if ((AGG_SCHEMA & MDB_AGG_KEYS) && got.mv_agg_keys != exp_keys) {
        fprintf(stderr, "%s: keys mismatch a=%" PRIu64 " b=%zu\n",
                msg, (uint64_t)got.mv_agg_keys, exp_keys);
        exit(EXIT_FAILURE);
    }
    if ((AGG_SCHEMA & MDB_AGG_HASHSUM) && memcmp(got.mv_agg_hashes, exp_hs, MDB_HASH_SIZE) != 0) {
        fprintf(stderr, "%s: hashsum mismatch\n", msg);
        exit(EXIT_FAILURE);
    }
}

static void shrink_dups_to_keep_last(MDB_txn *txn, MDB_dbi dbi, const MDB_val *key, size_t keep)
{
    MDB_cursor *mc = NULL;
    MDB_val k = *key, d;
    mdb_size_t cnt = 0;
    int rc;

    CHECK(mdb_cursor_open(txn, dbi, &mc), "shrink_dups: cursor_open");
    rc = mdb_cursor_get(mc, &k, &d, MDB_SET_KEY);
    if (rc == MDB_NOTFOUND) {
        mdb_cursor_close(mc);
        return;
    }
    CHECK(rc, "shrink_dups: set_key");
    CHECK(mdb_cursor_count(mc, &cnt), "shrink_dups: count");

    while ((size_t)cnt > keep) {
        rc = mdb_cursor_get(mc, &k, &d, MDB_LAST_DUP);
        CHECK(rc, "shrink_dups: last_dup");
        CHECK(mdb_cursor_del(mc, 0), "shrink_dups: del");
        cnt--;
    }

    mdb_cursor_close(mc);
}

/* ------------------------------ DUPSORT oscillation (subpage <-> subDB <-> collapse) ------------------------------ */

#ifdef MDB_DEBUG_AGG_INTEGRITY
#define OSC_BASE_KEYS 80u
#define OSC_DUPS 250u
#else
#define OSC_BASE_KEYS 160u
#define OSC_DUPS 900u
#endif
#define OSC_KEEP 2u


static void test_dupsort_subpage_subdb_oscillation(void)
{
    char dir[512];
    make_test_dir(dir, sizeof(dir), "t23_dups_subpage_subdb_osc");
    MDB_env *env = open_env_dir(dir, 16, 256u * 1024u * 1024u, 1);

    MDB_txn *txn = NULL;
    MDB_dbi dbi = 0;

    CHECK(mdb_txn_begin(env, NULL, 0, &txn), "osc: txn_begin");
    CHECK(mdb_dbi_open(txn, "dups", MDB_CREATE | MDB_DUPSORT | AGG_SCHEMA, &dbi), "osc: dbi_open");

    /* Heuristic: choose a value size that is close to the environment's maxkeysize
     * (but still safely within our stack buffers) to force dup-subDB growth. */
    uint8_t bigbuf[2048];
    uint8_t smallbuf[256];
    size_t big_extra;
    {
        int maxkey = mdb_env_get_maxkeysize(env);
        size_t want = (size_t)(maxkey > 0 ? maxkey : 511);
        /* Keep within maxkeysize; for DUPSORT, dup-subDB records are keys. */
        if (want > (size_t)MDB_HASH_SIZE + 800) want = (size_t)MDB_HASH_SIZE + 800;
        big_extra = want - (size_t)MDB_HASH_SIZE;
        if ((size_t)MDB_HASH_SIZE + big_extra > sizeof(bigbuf))
            big_extra = sizeof(bigbuf) - (size_t)MDB_HASH_SIZE;
    }

    uint8_t hi_sentinel[MDB_HASH_SIZE];
    memset(hi_sentinel, 0xFF, sizeof(hi_sentinel));

    /* Seed the main tree so that the parent leaf is busy when the dup-subDB changes. */
    for (unsigned i = 0; i < 30u; i++) {
        char kbuf[32];
        MDB_val k, d;
        snprintf(kbuf, sizeof(kbuf), "K%08u", i);
        k.mv_data = kbuf;
        k.mv_size = strlen(kbuf) + 1u;
        make_val_with_hash(smallbuf, sizeof(smallbuf), "seed", i, &d);
        CHECK(mdb_put(txn, dbi, &k, &d, 0), "osc: seed put");
    }
    DBG_CHECK(txn, dbi, "osc: after seed");

    /* Hot key (not among seeded keys). */
    MDB_val hotk;
    {
        static char hkbuf[] = "K00000042";
        hotk.mv_data = hkbuf;
        hotk.mv_size = sizeof(hkbuf);
    }

    /* Oscillate three times: small dupset (subpage) -> large dupset (dup-subDB) -> shrink.
     *
     * We also add key-local range checks during the “critical window” (when the dup-subDB
     * is large and likely to split) to catch aggregate divergence earlier than end-of-txn. */
    for (unsigned cycle = 0; cycle < 3u; cycle++) {
        /* Ensure the key doesn't exist from a previous cycle. */
        {
            int rc = mdb_del(txn, dbi, &hotk, NULL);
            REQUIRE(rc == MDB_SUCCESS || rc == MDB_NOTFOUND, "osc: unexpected rc from del(all)");
        }

        /* Start with a tiny dupset with small values (encourages subpage form). */
        for (unsigned j = 0; j < 5u; j++) {
            MDB_val d;
            make_val_with_hash_salted(smallbuf, sizeof(smallbuf), "osc.tiny", j, (uint64_t)(1000u + cycle), &d);
            CHECK(mdb_put(txn, dbi, &hotk, &d, 0), "osc: put tiny dup");
        }
        DBG_CHECK(txn, dbi, "osc: after tiny dupset");
        check_key_local_range(txn, dbi, &hotk, hi_sentinel, "osc: range local (tiny)");

        /* Grow to a large dupset with big values.
         * Cycle 0 uses monotonic DUPSORT order to encourage right-heavy splits. */
        const unsigned crit_start = 64u; /* after this, check more aggressively */
        for (unsigned j = 0; j < OSC_DUPS; j++) {
            MDB_val d;
            if (cycle == 0)
                make_big_val_with_monotonic_hash(bigbuf, sizeof(bigbuf), (unsigned)(cycle * 100000u + j), big_extra, &d);
            else
                make_big_val_with_hash(bigbuf, sizeof(bigbuf), (unsigned)(cycle * 100000u + j), big_extra, &d);

            CHECK(mdb_put(txn, dbi, &hotk, &d, 0), "osc: put dup");

            if (j >= crit_start) {
                /* Critical window: check every iteration under MDB_DEBUG_AGG_INTEGRITY builds. */
                DBG_CHECK(txn, dbi, "osc: dbg check (grow)");
                if ((j % 17u) == 0u)
                    check_key_local_range(txn, dbi, &hotk, hi_sentinel, "osc: range local (grow)");
            } else if ((j % 50u) == 0u) {
                DBG_CHECK(txn, dbi, "osc: dbg check (grow sparse)");
            }
        }
        DBG_CHECK(txn, dbi, "osc: after grow");
        check_key_local_range(txn, dbi, &hotk, hi_sentinel, "osc: range local (after grow)");

        /* Shrink aggressively: delete largest values down to a tiny remainder.
         * This is where dup-subDB collapse / merge paths get exercised. */
        shrink_dups_to_keep_last(txn, dbi, &hotk, OSC_KEEP);
        DBG_CHECK(txn, dbi, "osc: after shrink");
        check_key_local_range(txn, dbi, &hotk, hi_sentinel, "osc: range local (after shrink)");
        REQUIRE(dups_count_for_key(txn, dbi, &hotk) == OSC_KEEP, "osc: dupcount mismatch after shrink");

        /* Re-seed as subpage for the next cycle by deleting and inserting small values. */
        if (cycle != 2u) {
            CHECK(mdb_del(txn, dbi, &hotk, NULL), "osc: delete hot key");
            for (unsigned j = 0; j < 7u; j++) {
                MDB_val d;
                make_val_with_hash_salted(smallbuf, sizeof(smallbuf), "osc.reseed", j, (uint64_t)(2000u + cycle), &d);
                CHECK(mdb_put(txn, dbi, &hotk, &d, 0), "osc: re-seed dup");
            }
            DBG_CHECK(txn, dbi, "osc: after reseed");
            check_key_local_range(txn, dbi, &hotk, hi_sentinel, "osc: range local (reseed)");
        }
    }

    verify_against_oracle(txn, dbi, "dups subpage<->subdb oscillation");
    CHECK(mdb_txn_commit(txn), "osc: txn_commit");
    mdb_env_close(env);
}



/* ------------------------------ driver ------------------------------ */

typedef void (*test_fn)(void);

typedef struct {
    const char *name;
    test_fn fn;
} Test;


/* ------------------------------ Additional subpage<->subDB stress tests ------------------------------ */

static void test_dupsort_subpage_subdb_root_split_repair_regress(void)
{
    char dir[512];
    make_test_dir(dir, sizeof(dir), "t24_dups_subdb_rootsplit_repair");
    MDB_env *env = open_env_dir(dir, 16, 256u * 1024u * 1024u, 1);

    MDB_txn *txn = NULL;
    MDB_dbi dbi = 0;

    CHECK(mdb_txn_begin(env, NULL, 0, &txn), "rootsplit: txn_begin");
    CHECK(mdb_dbi_open(txn, "dups", MDB_CREATE | MDB_DUPSORT | AGG_SCHEMA, &dbi), "rootsplit: dbi_open");

    uint8_t bigbuf[2048];
    uint8_t smallbuf[256];
    size_t big_extra;
    {
        int maxkey = mdb_env_get_maxkeysize(env);
        size_t want = (size_t)(maxkey > 0 ? maxkey : 511);
        if (want > (size_t)MDB_HASH_SIZE + 800) want = (size_t)MDB_HASH_SIZE + 800;
        big_extra = want - (size_t)MDB_HASH_SIZE;
        if ((size_t)MDB_HASH_SIZE + big_extra > sizeof(bigbuf))
            big_extra = sizeof(bigbuf) - (size_t)MDB_HASH_SIZE;
    }

    uint8_t hi_sentinel[MDB_HASH_SIZE];
    memset(hi_sentinel, 0xFF, sizeof(hi_sentinel));

    /* Fill the main tree a bit. */
    for (unsigned i = 0; i < 48u; i++) {
        char kbuf[32];
        MDB_val k, d;
        snprintf(kbuf, sizeof(kbuf), "K%08u", i);
        k.mv_data = kbuf;
        k.mv_size = strlen(kbuf) + 1u;
        make_val_with_hash(smallbuf, sizeof(smallbuf), "rootsplit.seed", i, &d);
        CHECK(mdb_put(txn, dbi, &k, &d, 0), "rootsplit: seed put");
    }

    MDB_val hotk;
    {
        static char hkbuf[] = "K00000042";
        hotk.mv_data = hkbuf;
        hotk.mv_size = sizeof(hkbuf);
    }

    /* Start with a tiny subpage dupset. */
    for (unsigned j = 0; j < 3u; j++) {
        MDB_val d;
        make_val_with_hash_salted(smallbuf, sizeof(smallbuf), "rootsplit.tiny", j, (uint64_t)777u, &d);
        CHECK(mdb_put(txn, dbi, &hotk, &d, 0), "rootsplit: put tiny dup");
    }
    DBG_CHECK(txn, dbi, "rootsplit: after tiny");

    /* Force deep dup-subDB growth and root splits with monotonic inserts. */
    const unsigned target = OSC_DUPS + 40u;
    for (unsigned j = 0; j < target; j++) {
        MDB_val d;
        make_big_val_with_monotonic_hash(bigbuf, sizeof(bigbuf), (unsigned)(900000u + j), big_extra, &d);
        CHECK(mdb_put(txn, dbi, &hotk, &d, 0), "rootsplit: put big dup");
        if (j >= 64u) {
            DBG_CHECK(txn, dbi, "rootsplit: dbg check (grow)");
            if ((j % 23u) == 0u)
                check_key_local_range(txn, dbi, &hotk, hi_sentinel, "rootsplit: range local (grow)");
        }
    }
    DBG_CHECK(txn, dbi, "rootsplit: after grow");
    check_key_local_range(txn, dbi, &hotk, hi_sentinel, "rootsplit: range local (after grow)");

    /* Shrink hard and then regrow to shake merge/split repair paths. */
    shrink_dups_to_keep_last(txn, dbi, &hotk, 2u);
    DBG_CHECK(txn, dbi, "rootsplit: after shrink");
    check_key_local_range(txn, dbi, &hotk, hi_sentinel, "rootsplit: range local (after shrink)");

    for (unsigned j = 0; j < target; j++) {
        MDB_val d;
        make_big_val_with_hash(bigbuf, sizeof(bigbuf), (unsigned)(1200000u + (j * 7u)), big_extra, &d);
        CHECK(mdb_put(txn, dbi, &hotk, &d, 0), "rootsplit: regrow put");
        if ((j % 37u) == 0u)
            check_key_local_range(txn, dbi, &hotk, hi_sentinel, "rootsplit: range local (regrow)");
    }
    DBG_CHECK(txn, dbi, "rootsplit: after regrow");

    verify_against_oracle(txn, dbi, "dups subdb root split repair");
    CHECK(mdb_txn_commit(txn), "rootsplit: txn_commit");
    mdb_env_close(env);
}

// #ifdef MDB_DEBUG_AGG_INTEGRITY
// #define HAMMER_ROUNDS 18u
// #define HAMMER_HOT_KEYS 6u
// #define HAMMER_HI 260u
// #define HAMMER_KEEP 3u
// #else
#define HAMMER_ROUNDS 60u
#define HAMMER_HOT_KEYS 12u
#define HAMMER_HI 900u
#define HAMMER_KEEP 4u
// #endif 
#define HAMMER_SMALL_INIT 7u

static void test_dupsort_subpage_subdb_oscillation_hammer(void)
{
    char dir[512];
    make_test_dir(dir, sizeof(dir), "t25_dups_subpage_subdb_hammer");
    MDB_env *env = open_env_dir(dir, 16, 512u * 1024u * 1024u, 1);

    MDB_txn *txn = NULL;
    MDB_dbi dbi = 0;

    CHECK(mdb_txn_begin(env, NULL, 0, &txn), "hammer: txn_begin");
    CHECK(mdb_dbi_open(txn, "dups", MDB_CREATE | MDB_DUPSORT | AGG_SCHEMA, &dbi), "hammer: dbi_open");

    uint8_t bigbuf[2048];
    uint8_t smallbuf[256];
    size_t big_extra;
    {
        int maxkey = mdb_env_get_maxkeysize(env);
        size_t want = (size_t)(maxkey > 0 ? maxkey : 511);
        if (want > (size_t)MDB_HASH_SIZE + 800) want = (size_t)MDB_HASH_SIZE + 800;
        big_extra = want - (size_t)MDB_HASH_SIZE;
        if ((size_t)MDB_HASH_SIZE + big_extra > sizeof(bigbuf))
            big_extra = sizeof(bigbuf) - (size_t)MDB_HASH_SIZE;
    }

    uint8_t hi_sentinel[MDB_HASH_SIZE];
    memset(hi_sentinel, 0xFF, sizeof(hi_sentinel));

    /* Baseline main-tree noise keys. */
    for (unsigned i = 0; i < 96u; i++) {
        char kbuf[32];
        MDB_val k, d;
        snprintf(kbuf, sizeof(kbuf), "N%08u", i);
        k.mv_data = kbuf;
        k.mv_size = strlen(kbuf) + 1u;
        make_val_with_hash(smallbuf, sizeof(smallbuf), "hammer.seed", (unsigned)(50000u + i), &d);
        CHECK(mdb_put(txn, dbi, &k, &d, 0), "hammer: seed put");
    }

    for (unsigned round = 0; round < HAMMER_ROUNDS; round++) {
        char hkbuf[32];
        MDB_val hotk;
        snprintf(hkbuf, sizeof(hkbuf), "K%08u", (unsigned)(1000u + (round % HAMMER_HOT_KEYS)));
        hotk.mv_data = hkbuf;
        hotk.mv_size = strlen(hkbuf) + 1u;

        /* Reset the key, then build subpage -> subDB -> shrink, repeatedly. */
        {
            int rc = mdb_del(txn, dbi, &hotk, NULL);
            REQUIRE(rc == MDB_SUCCESS || rc == MDB_NOTFOUND, "hammer: unexpected rc from del(all)");
        }

        for (unsigned j = 0; j < HAMMER_SMALL_INIT; j++) {
            MDB_val d;
            make_val_with_hash_salted(smallbuf, sizeof(smallbuf), "hammer.tiny", j, (uint64_t)(0xAB00u + round), &d);
            CHECK(mdb_put(txn, dbi, &hotk, &d, 0), "hammer: put tiny");
        }

        for (unsigned j = 0; j < HAMMER_HI; j++) {
            MDB_val d;
            unsigned id = (unsigned)(round * 100000u + (round % HAMMER_HOT_KEYS) * 1000u + j);
            if ((round & 1u) == 0u)
                make_big_val_with_monotonic_hash(bigbuf, sizeof(bigbuf), id, big_extra, &d);
            else
                make_big_val_with_hash(bigbuf, sizeof(bigbuf), id, big_extra, &d);

            CHECK(mdb_put(txn, dbi, &hotk, &d, 0), "hammer: put big");
            if (j >= 64u) {
                DBG_CHECK(txn, dbi, "hammer: dbg check (grow)");
                if ((j % 31u) == 0u)
                    check_key_local_range(txn, dbi, &hotk, hi_sentinel, "hammer: range local (grow)");
            }
        }

        shrink_dups_to_keep_last(txn, dbi, &hotk, HAMMER_KEEP);
        DBG_CHECK(txn, dbi, "hammer: after shrink");
        check_key_local_range(txn, dbi, &hotk, hi_sentinel, "hammer: range local (after shrink)");

        /* Interleave some non-hot updates to keep the main tree active. */
        if ((round % 3u) == 0u) {
            char kbuf[32];
            MDB_val k, d;
            snprintf(kbuf, sizeof(kbuf), "N%08u", (unsigned)(200u + round));
            k.mv_data = kbuf;
            k.mv_size = strlen(kbuf) + 1u;
            make_val_with_hash(smallbuf, sizeof(smallbuf), "hammer.noise", (unsigned)(90000u + round), &d);
            CHECK(mdb_put(txn, dbi, &k, &d, 0), "hammer: noise put");
        }
        if ((round % 7u) == 0u) {
            char kbuf[32];
            MDB_val k;
            snprintf(kbuf, sizeof(kbuf), "N%08u", (unsigned)(round)); /* likely exists from seeding */
            k.mv_data = kbuf;
            k.mv_size = strlen(kbuf) + 1u;
            (void)mdb_del(txn, dbi, &k, NULL);
        }

        if ((round % 4u) == 0u)
            verify_against_oracle(txn, dbi, "dups subpage<->subdb hammer (checkpoint)");
    }

    verify_against_oracle(txn, dbi, "dups subpage<->subdb hammer");
    CHECK(mdb_txn_commit(txn), "hammer: txn_commit");
    mdb_env_close(env);
}


static void
test_dupsort_subdb_invariants_focus(void)
{
	const char *dir = "/tmp/testdb_agg_unit/t17_dups_focus";
	MDB_env *env;
	MDB_txn *txn;
	MDB_dbi dbi;
	MDB_val hotk, d;
	uint8_t bigbuf[1024];
	uint8_t smallbuf[256];

	create_clean_dir(dir);
	CHECK(mdb_env_create(&env), "env_create");
	CHECK(mdb_env_set_maxdbs(env, 16), "env_set_maxdbs");
	CHECK(mdb_env_set_mapsize(env, 268435456u), "env_set_mapsize");
	CHECK(mdb_env_open(env, dir, 0, 0664), "env_open");

	CHECK(mdb_txn_begin(env, NULL, 0, &txn), "txn_begin");
	CHECK(mdb_dbi_open(txn, "d", MDB_CREATE|MDB_DUPSORT|
		MDB_AGG_ENTRIES|MDB_AGG_KEYS|MDB_AGG_HASHSUM, &dbi), "dbi_open");
	make_key_string(&hotk, "KFOCUS0000");

	/* Seed unrelated keys to keep the main DB from being too trivial. */
	seed_main_keys(txn, dbi, "seed", 64);

	for (unsigned cycle = 0; cycle < 30; cycle++) {
		/* Reset the hot key to force repeated subpage<->subdb conversions. */
		(void)mdb_del(txn, dbi, &hotk, NULL);

		/* Build a deep dup-subdb with non-monotone insertion order to create
		   internal splits (and occasionally recursive parent splits). */
		const unsigned N = 1600;
		for (unsigned j = 0; j < N; j++) {
			unsigned id = (j & 1u) ? (N - 1u - j) : j;
			make_big_val_with_hash(bigbuf, sizeof(bigbuf),
				id ^ (unsigned)(cycle * 0x9e3779b9u),
				511u - (size_t)MDB_HASH_SIZE, &d);
			CHECK(mdb_put(txn, dbi, &hotk, &d, 0), "focus: put big");
			if ((j % 200u) == 199u)
				DBG_CHECK(txn, dbi, "focus: periodic");
		}
		DBG_CHECK(txn, dbi, "focus: after_build");

		/* Shrink aggressively to trigger subdb merges and possibly depth reduction. */
		shrink_dups_to_keep_last(txn, dbi, &hotk, 6);
		DBG_CHECK(txn, dbi, "focus: after_shrink");

		/* Grow a bit again to re-trigger subpage growth and re-splitting. */
		for (unsigned j = 0; j < 48; j++) {
			make_val_with_hash_salted(smallbuf, sizeof(smallbuf),
				"focus", (unsigned)(100000u + j), (uint64_t)cycle, &d);
			CHECK(mdb_put(txn, dbi, &hotk, &d, 0), "focus: put small");
		}
		DBG_CHECK(txn, dbi, "focus: after_regrow");

		/* Sprinkle a few deletes of the full dupset to exercise the "delete all dups" path. */
		if ((cycle % 5u) == 4u) {
			CHECK(mdb_del(txn, dbi, &hotk, NULL), "focus: del all dups");
			DBG_CHECK(txn, dbi, "focus: after_del_all");
		}
	}

	CHECK(mdb_txn_commit(txn), "txn_commit");
	mdb_dbi_close(env, dbi);
	mdb_env_close(env);
}

static uint64_t
xorshift64(uint64_t *s)
{
	uint64_t x = *s;
	x ^= x << 13;
	x ^= x >> 7;
	x ^= x << 17;
	*s = x;
	return x;
}

static void
test_dupsort_subdb_agg_stress_txn_churn(void)
{
	const char *dir = "/tmp/testdb_agg_unit/t18_dups_stress_churn";
	MDB_env *env;
	MDB_txn *txn;
	MDB_dbi dbi;
	uint8_t bigbuf[1024];
	uint8_t smallbuf[256];

	create_clean_dir(dir);
	CHECK(mdb_env_create(&env), "env_create");
	CHECK(mdb_env_set_maxdbs(env, 16), "env_set_maxdbs");
	CHECK(mdb_env_set_mapsize(env, 268435456u), "env_set_mapsize");
	CHECK(mdb_env_open(env, dir, 0, 0664), "env_open");

	/* Open DBI once. */
	CHECK(mdb_txn_begin(env, NULL, 0, &txn), "txn_begin");
	CHECK(mdb_dbi_open(txn, "d", MDB_CREATE|MDB_DUPSORT|
		MDB_AGG_ENTRIES|MDB_AGG_KEYS|MDB_AGG_HASHSUM, &dbi), "dbi_open");
	CHECK(mdb_txn_commit(txn), "txn_commit");

	uint64_t rng = 0x243f6a8885a308d3ULL;

	for (unsigned round = 0; round < 120; round++) {
		CHECK(mdb_txn_begin(env, NULL, 0, &txn), "txn_begin");

		for (unsigned op = 0; op < 220; op++) {
			uint64_t r = xorshift64(&rng);
			unsigned kidx = (unsigned)(r % 8u);
			char kbuf[32];
			snprintf(kbuf, sizeof(kbuf), "KSTRESS%02u", kidx);

			MDB_val k, d;
			make_key_string(&k, kbuf);

			/* 0..6: insert, 7: delete all dups for that key */
			if ((r & 7u) != 7u) {
				if (r & 1u) {
					make_big_val_with_hash(bigbuf, sizeof(bigbuf),
						(unsigned)(r & 0xffffffffu) ^ (unsigned)(round * 0x9e3779b9u),
						511u - (size_t)MDB_HASH_SIZE, &d);
				} else {
					make_val_with_hash_salted(smallbuf, sizeof(smallbuf),
						"stress", (unsigned)(r & 0xffffffffu),
						(uint64_t)op, &d);
				}
				CHECK(mdb_put(txn, dbi, &k, &d, 0), "stress: put");
			} else {
				(void)mdb_del(txn, dbi, &k, NULL);
			}
		}

		if ((round % 10u) == 0u)
			DBG_CHECK(txn, dbi, "stress: periodic write");

		CHECK(mdb_txn_commit(txn), "txn_commit");

		/* Also check on a read-only txn occasionally, to ensure persistent state is consistent. */
		if ((round % 15u) == 0u) {
			MDB_txn *rtxn;
			CHECK(mdb_txn_begin(env, NULL, MDB_RDONLY, &rtxn), "rtxn_begin");
			DBG_CHECK(rtxn, dbi, "stress: periodic read");
			mdb_txn_abort(rtxn);
		}
	}

	mdb_dbi_close(env, dbi);
	mdb_env_close(env);
}

static void run_test(const Test *t)
{
    fprintf(stdout, "[RUN] %s\n", t->name);
    t->fn();
    fprintf(stdout, "[OK ] %s\n", t->name);
}

int main(void)
{
    const Test tests[] = {
        {"hashsum algebra", test_hashsum_algebra},
        {"plain right splis", test_plain_rightsplit},
        {"plain left splits", test_plain_leftsplit},
        {"plain middle fill", test_plain_middlefill},
        {"plain delete middle", test_plain_delete_middle},
        {"plain root collapse", test_plain_root_collapse},
        {"plain overwrite", test_plain_overwrite},
        {"plain overflow values", test_plain_overflow_values_hashsum},
        {"plain deep split/merge oscillation", test_plain_deep_split_merge_oscillation},
        {"plain merge update_key growth", test_plain_merge_update_key_growth_split},
        {"regress del big overflow same-txn", test_regress_del_bigdata_same_txn_after_freelist_loaded},
        {"dupsort divergence", test_dupsort_divergence},
        {"dupsort round robin", test_dupsort_round_robin},
        {"dupsort large dupset subdb build", test_dupsort_large_dups_subdb_build},
        {"dupsort subdb merges/collapse", test_dupsort_subdb_merge_and_collapse},
        {"dupsort set_range within dupset", test_dupsort_set_range_within_dups},
        {"dupsort prefix/range partial dupset", test_dupsort_prefix_range_partial},
        {"dupsort prefix/range corner-cases", test_dupsort_prefix_range_cornercases},
        {"dupsort delete middle keys", test_dupsort_main_tree_delete_middle},
        {"dupsort merge update_key growth", test_dupsort_main_merge_update_key_growth_split},
        {"dupfixed merge/collapse", test_dupfixed_merge_and_collapse},
        {"dupfixed MDB_MULTIPLE", test_dupfixed_multiple},
        {"upgrade policy named DB", test_upgrade_policy_named_db},

        {"prefix == range (upper)", test_prefix_matches_range},
        {"rank input shape rules", test_rank_input_shape_rules},
        {"hashsum: MDB_RESERVE incompatible", test_hashsum_reserve_incompatible},
        {"hashsum: WRITEMAP incompatible", test_hashsum_writemap_incompatible},
        {"hashsum: nonzero offset (plain)", test_hashsum_nonzero_offset_plain},
        {"hashsum: nonzero offset (dupfixed)", test_hashsum_nonzero_offset_dupfixed},
        {"put: NOOVERWRITE no agg change", test_put_nooverwrite_no_agg_change},
        {"dupsort: NODUPDATA no agg change", test_dupsort_put_nodupdata_no_agg_change},
        {"cursor MDB_CURRENT hashsum delta", test_cursor_current_update_hashsum_delta},
        {"api contracts prefix/range", test_api_contracts_prefix_range},
        {"api window/rank/seek (step1-3)", test_api_window_rank_seek},
        {"dupsort subpage<->subDB oscillation", test_dupsort_subpage_subdb_oscillation},

        {"dupsort dup-subDB roott split repair (regress)", test_dupsort_subpage_subdb_root_split_repair_regress},
        {"dupsort subpage<->subDB oscillation hammer", test_dupsort_subpage_subdb_oscillation_hammer},

        {"dupsort subdb invariants focus", test_dupsort_subdb_invariants_focus},
        {"dupsort subdb agg stress txn churn", test_dupsort_subdb_agg_stress_txn_churn},

        {"stale named DBI open after abort", test_stale_named_dbi_open_after_abort},
        {"nested write txn abort/commit", test_nested_txn_abort_commit},

    };

    for (size_t i = 0; i < sizeof(tests)/sizeof(tests[0]); i++)
        run_test(&tests[i]);

    fprintf(stdout, "All unit tests passed.\n");
    return 0;
}
