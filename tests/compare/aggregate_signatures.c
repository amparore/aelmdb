#include <assert.h>
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "lmdb.h"

#ifndef TEST_IMPL_NAME
#define TEST_IMPL_NAME "unknown"
#endif

typedef struct Rec {
	MDB_val k, v;
	unsigned char *kb, *vb;
} Rec;

typedef struct Vec {
	Rec *v;
	size_t n, cap;
} Vec;

static const char *g_phase = "startup";
static uint64_t g_signature = UINT64_C(1469598103934665603);


/* Long shared stress defaults.  They can be overridden at compile time.
 * The workload is deterministic for a given tuple (seed, rounds, ops).
 * Keep these defaults large enough to exercise repeated split/merge/re-growth
 * cycles, but small enough that a full AELMDB comparison remains a
 * practical developer test. */
#ifndef AGG_COMPARE_LONG_SEED
#define AGG_COMPARE_LONG_SEED UINT64_C(0x6d64626167676c31)
#endif
#ifndef AGG_COMPARE_LONG_ROUNDS
#define AGG_COMPARE_LONG_ROUNDS 80u
#endif
#ifndef AGG_COMPARE_LONG_OPS
#define AGG_COMPARE_LONG_OPS 400u
#endif
#ifndef AGG_COMPARE_LONG_VERIFY_EVERY
#define AGG_COMPARE_LONG_VERIFY_EVERY 8u
#endif
#ifndef AGG_COMPARE_LONG_REOPEN_EVERY
#define AGG_COMPARE_LONG_REOPEN_EVERY 16u
#endif

/* Large-key structural stress defaults.  AGG_COMPARE_LARGE_KEYS_COUNT is the
 * number of sparse (even) logical keys; the interior-fill phase inserts the
 * same number of odd keys, so the full tree contains twice this many primary
 * keys.  The default is chosen to make depth >= 4 realistic on 4 KiB pages
 * while keeping the shared AELMDB test compact. */
#ifndef AGG_COMPARE_LARGE_KEYS_SEED
#define AGG_COMPARE_LARGE_KEYS_SEED UINT64_C(0x6c617267656b6579)
#endif
#ifndef AGG_COMPARE_LARGE_KEYS_COUNT
#define AGG_COMPARE_LARGE_KEYS_COUNT 768u
#endif
#ifndef AGG_COMPARE_LARGE_KEYS_ROUNDS
#define AGG_COMPARE_LARGE_KEYS_ROUNDS 4u
#endif
#ifndef AGG_COMPARE_LARGE_KEYS_MIN_DEPTH
#define AGG_COMPARE_LARGE_KEYS_MIN_DEPTH 4u
#endif
#ifndef AGG_COMPARE_LARGE_KEYS_VARIABLE
#define AGG_COMPARE_LARGE_KEYS_VARIABLE 0
#endif

#if defined(AGG_COMPARE_STRESS_LONG) || defined(AGG_COMPARE_STRESS_LARGE_KEYS)
static char g_phase_buf[128];
#endif

static void sig_bytes(const void *p, size_t n)
{
	const unsigned char *s = (const unsigned char *)p;
	size_t i;
	for (i = 0; i < n; ++i) {
		g_signature ^= s[i];
		g_signature *= UINT64_C(1099511628211);
	}
}

static void sig_u64(uint64_t v)
{
	unsigned char b[8];
	unsigned i;
	for (i = 0; i < 8; ++i)
		b[i] = (unsigned char)(v >> (i * 8));
	sig_bytes(b, sizeof(b));
}

static void die_rc(const char *where, int rc)
{
	if (rc) {
		fprintf(stderr, "FAIL[%s/%s]: %s: %d %s\n",
			TEST_IMPL_NAME, g_phase, where, rc, mdb_strerror(rc));
		exit(2);
	}
}

static void check(int ok, const char *msg)
{
	if (!ok) {
		fprintf(stderr, "FAIL[%s/%s]: %s\n", TEST_IMPL_NAME, g_phase, msg);
		exit(3);
	}
}

static void vec_push(Vec *x, const MDB_val *k, const MDB_val *v)
{
	Rec *r;
	if (x->n == x->cap) {
		x->cap = x->cap ? x->cap * 2 : 128;
		x->v = (Rec *)realloc(x->v, x->cap * sizeof(*x->v));
		check(x->v != NULL, "realloc vec");
	}
	r = &x->v[x->n++];
	r->kb = (unsigned char *)malloc(k->mv_size ? k->mv_size : 1);
	r->vb = (unsigned char *)malloc(v->mv_size ? v->mv_size : 1);
	check(r->kb && r->vb, "malloc record");
	memcpy(r->kb, k->mv_data, k->mv_size);
	memcpy(r->vb, v->mv_data, v->mv_size);
	r->k.mv_size = k->mv_size;
	r->k.mv_data = r->kb;
	r->v.mv_size = v->mv_size;
	r->v.mv_data = r->vb;
}

static void vec_free(Vec *x)
{
	size_t i;
	for (i = 0; i < x->n; ++i) {
		free(x->v[i].kb);
		free(x->v[i].vb);
	}
	free(x->v);
	memset(x, 0, sizeof(*x));
}

static void be32(unsigned char *p, uint32_t x)
{
	p[0] = (unsigned char)(x >> 24);
	p[1] = (unsigned char)(x >> 16);
	p[2] = (unsigned char)(x >> 8);
	p[3] = (unsigned char)x;
}

static uint32_t load_be32(const unsigned char *p)
{
	return ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) |
		((uint32_t)p[2] << 8) | (uint32_t)p[3];
}

static void make_gap_after_8(const MDB_val *src, unsigned char gap[8])
{
	uint32_t x;
	check(src->mv_size == 8, "gap key size");
	memcpy(gap, src->mv_data, 8);
	x = load_be32(gap + 4);
	check(x != UINT32_MAX, "gap key overflow");
	be32(gap + 4, x + 1u);
}

static int valcmp(const MDB_val *a, const MDB_val *b)
{
	size_t n = a->mv_size < b->mv_size ? a->mv_size : b->mv_size;
	int c = memcmp(a->mv_data, b->mv_data, n);
	if (c)
		return c;
	return a->mv_size < b->mv_size ? -1 : a->mv_size > b->mv_size;
}

static size_t slice_start(size_t n, int off)
{
	if (off >= 0) {
		check((size_t)off + MDB_HASH_SIZE <= n, "positive slice");
		return (size_t)off;
	}
	{
		size_t back = (size_t)(-(off + 1));
		check(MDB_HASH_SIZE + back <= n, "negative slice");
		return n - MDB_HASH_SIZE - back;
	}
}

static void hadd(unsigned char *dst, const unsigned char *src)
{
	unsigned carry = 0;
	size_t i;
	for (i = 0; i < MDB_HASH_SIZE; i++) {
		unsigned s = dst[i] + src[i] + carry;
		dst[i] = (unsigned char)s;
		carry = s >> 8;
	}
}

static void expected(const Vec *x, int dupsort, int keyhash, int hashoff,
	const MDB_val *lk, const MDB_val *ld, int lincl,
	const MDB_val *hk, const MDB_val *hd, int hincl,
	MDB_agg *out)
{
	size_t i, last_key = (size_t)-1;
	memset(out, 0, sizeof(*out));
	for (i = 0; i < x->n; i++) {
		const Rec *r = &x->v[i];
		int in = 1, c;
		if (lk) {
			c = valcmp(&r->k, lk);
			if (dupsort && ld && c == 0)
				c = valcmp(&r->v, ld);
			if (c < 0 || (c == 0 && !lincl))
				in = 0;
		}
		if (hk) {
			c = valcmp(&r->k, hk);
			if (dupsort && hd && c == 0)
				c = valcmp(&r->v, hd);
			if (c > 0 || (c == 0 && !hincl))
				in = 0;
		}
		if (!in)
			continue;
		out->mv_agg_entries++;
		if (last_key == (size_t)-1 || valcmp(&x->v[last_key].k, &r->k) != 0) {
			out->mv_agg_keys++;
			last_key = i;
		}
		{
			const MDB_val *s = keyhash ? &r->k : &r->v;
			hadd(out->mv_agg_hashes,
				(const unsigned char *)s->mv_data + slice_start(s->mv_size, hashoff));
		}
	}
}

static void sig_agg(const MDB_agg *a)
{
	sig_u64(a->mv_flags);
	sig_u64(a->mv_agg_entries);
	sig_u64(a->mv_agg_keys);
	sig_bytes(a->mv_agg_hashes, MDB_HASH_SIZE);
}

static void expect_agg(const MDB_agg *got, const MDB_agg *exp,
	unsigned schema, const char *where)
{
	if (got->mv_flags != schema ||
		got->mv_agg_entries != exp->mv_agg_entries ||
		got->mv_agg_keys != exp->mv_agg_keys ||
		memcmp(got->mv_agg_hashes, exp->mv_agg_hashes, MDB_HASH_SIZE)) {
		fprintf(stderr,
			"FAIL[%s/%s]: agg mismatch %s flags=%x/%x e=%llu/%llu k=%llu/%llu\n",
			TEST_IMPL_NAME, g_phase, where, got->mv_flags, schema,
			(unsigned long long)got->mv_agg_entries,
			(unsigned long long)exp->mv_agg_entries,
			(unsigned long long)got->mv_agg_keys,
			(unsigned long long)exp->mv_agg_keys);
		exit(4);
	}
	sig_agg(got);
}

static void snapshot(MDB_txn *txn, MDB_dbi dbi, Vec *x)
{
	MDB_cursor *c;
	MDB_val k = {0}, v = {0};
	int rc;
	die_rc("cursor_open", mdb_cursor_open(txn, dbi, &c));
	for (rc = mdb_cursor_get(c, &k, &v, MDB_FIRST);
		rc == 0;
		rc = mdb_cursor_get(c, &k, &v, MDB_NEXT)) {
		vec_push(x, &k, &v);
		sig_bytes(k.mv_data, k.mv_size);
		sig_bytes(v.mv_data, v.mv_size);
	}
	check(rc == MDB_NOTFOUND, "snapshot end");
	mdb_cursor_close(c);
}

#ifdef AGG_COMPARE_TRACE
static uint64_t vec_fingerprint(const Vec *x)
{
	uint64_t h = UINT64_C(1469598103934665603);
	size_t i, j;
	for (i = 0; i < x->n; ++i) {
		const unsigned char *p = x->v[i].k.mv_data;
		for (j = 0; j < x->v[i].k.mv_size; ++j) { h ^= p[j]; h *= UINT64_C(1099511628211); }
		p = x->v[i].v.mv_data;
		for (j = 0; j < x->v[i].v.mv_size; ++j) { h ^= p[j]; h *= UINT64_C(1099511628211); }
	}
	return h;
}
#endif

static void fill_value(unsigned char *p, size_t n, uint32_t a, uint32_t b)
{
	size_t i;
	memset(p, 0, n);
	be32(p, a);
	if (n >= 8)
		be32(p + 4, b);
	for (i = 8; i < n; i++)
		p[i] = (unsigned char)(a * 17u + b * 29u + i * 13u);
}

static void make_plain_key(unsigned char kb[8], unsigned i)
{
	memset(kb, 0, 8);
	be32(kb + 4, i * 2 + 1);
}

static void make_dup_key(unsigned char kb[8], unsigned i)
{
	memset(kb, 0, 8);
	be32(kb + 4, i * 3 + 2);
}

static void put_plain(MDB_txn *txn, MDB_dbi dbi, unsigned n, size_t vlen)
{
	unsigned i;
	unsigned char kb[8], *vb = (unsigned char *)malloc(vlen);
	MDB_val k = {8, kb}, v = {vlen, vb};
	check(vb != NULL, "plain value alloc");
	for (i = 0; i < n; i++) {
		make_plain_key(kb, i);
		fill_value(vb, vlen, i, 7);
		die_rc("plain put", mdb_put(txn, dbi, &k, &v, 0));
	}
	free(vb);
}

static void put_dups(MDB_txn *txn, MDB_dbi dbi, unsigned keys,
	unsigned base_dups, size_t vlen)
{
	unsigned i, j;
	unsigned char kb[8], *vb = (unsigned char *)malloc(vlen);
	MDB_val k = {8, kb}, v = {vlen, vb};
	check(vb != NULL, "dup value alloc");
	for (i = 0; i < keys; i++) {
		unsigned nd = (i % 9 == 0) ? base_dups * 8 : 1 + (i % base_dups);
		make_dup_key(kb, i);
		for (j = 0; j < nd; j++) {
			fill_value(vb, vlen, j + 1, i + 3);
			die_rc("dup put", mdb_put(txn, dbi, &k, &v, 0));
		}
	}
	free(vb);
}


#ifdef AGG_COMPARE_STRESS_GROWTH
static void grow_plain(MDB_txn *txn, MDB_dbi dbi, size_t vlen)
{
	unsigned i;
	unsigned char kb[8], *vb = (unsigned char *)malloc(vlen);
	MDB_val k = {8, kb}, v = {vlen, vb};
	check(vb != NULL, "grow plain alloc");
	for (i = 100; i < 900; i += 19) {
		make_plain_key(kb, i);
		fill_value(vb, vlen, i, 91);
		die_rc("plain growth replace", mdb_put(txn, dbi, &k, &v, 0));
	}
	for (i = 900; i < 1180; ++i) {
		make_plain_key(kb, i);
		fill_value(vb, vlen, i, 123);
		die_rc("plain growth append", mdb_put(txn, dbi, &k, &v, 0));
	}
	free(vb);
}

static void grow_dups(MDB_txn *txn, MDB_dbi dbi, size_t vlen)
{
	unsigned i, j;
	unsigned char kb[8], *vb = (unsigned char *)malloc(vlen);
	MDB_val k = {8, kb}, v = {vlen, vb};
	check(vb != NULL, "grow dup alloc");
	for (i = 0; i < 75; i += 8) {
		make_dup_key(kb, i);
		for (j = 100; j < 118; ++j) {
			fill_value(vb, vlen, j, i + 3);
			die_rc("dup growth existing", mdb_put(txn, dbi, &k, &v, 0));
		}
	}
	for (i = 75; i < 100; ++i) {
		make_dup_key(kb, i);
		for (j = 0; j < 7; ++j) {
			fill_value(vb, vlen, 200 + j, i + 3);
			die_rc("dup growth new", mdb_put(txn, dbi, &k, &v, 0));
		}
	}
	free(vb);
}

static void grow_keydb(MDB_txn *txn, MDB_dbi dbi, size_t klen)
{
	unsigned i;
	unsigned char *kb = (unsigned char *)malloc(klen), vb[16];
	MDB_val k = {klen, kb}, v = {sizeof(vb), vb};
	check(kb != NULL, "grow keydb alloc");
	for (i = 120; i < 200; ++i) {
		fill_value(kb, klen, i, 1);
		fill_value(vb, sizeof(vb), i, 2);
		die_rc("keydb growth add", mdb_put(txn, dbi, &k, &v, 0));
	}
	free(kb);
}

#endif

#ifdef AGG_COMPARE_STRESS_MUTATIONS
static void mutate_plain(MDB_txn *txn, MDB_dbi dbi, size_t vlen)
{
	unsigned i;
	unsigned char kb[8], *vb = (unsigned char *)malloc(vlen);
	MDB_val k = {8, kb}, v = {vlen, vb};
	check(vb != NULL, "mutate plain alloc");

	/* Large deletions from both sparse and contiguous regions exercise rebalance. */
	for (i = 0; i < 360; i += 3) {
		make_plain_key(kb, i);
		die_rc("plain sparse delete", mdb_del(txn, dbi, &k, NULL));
	}
	for (i = 610; i < 860; ++i) {
		make_plain_key(kb, i);
		die_rc("plain contiguous delete", mdb_del(txn, dbi, &k, NULL));
	}

	/* Replace surviving values and insert keys into gaps/new right edge. */
	for (i = 361; i < 610; i += 17) {
		make_plain_key(kb, i);
		fill_value(vb, vlen, i, 91);
		die_rc("plain replace", mdb_put(txn, dbi, &k, &v, 0));
	}
	for (i = 900; i < 1080; ++i) {
		make_plain_key(kb, i);
		fill_value(vb, vlen, i, 123);
		die_rc("plain append growth", mdb_put(txn, dbi, &k, &v, 0));
	}
	free(vb);
}

static void mutate_dups(MDB_txn *txn, MDB_dbi dbi, size_t vlen)
{
	unsigned i;
	unsigned char kb[8], *vb = (unsigned char *)malloc(vlen);
	MDB_val k = {8, kb}, v = {vlen, vb};
	check(vb != NULL, "mutate dup alloc");

	/* Delete complete dupsets for selected keys. */
	for (i = 5; i < 75; i += 17) {
		make_dup_key(kb, i);
		die_rc("dup delete key", mdb_del(txn, dbi, &k, NULL));
	}

	/* Delete one exact duplicate from many remaining dupsets. */
	for (i = 1; i < 75; i += 7) {
		if (i >= 5 && ((i - 5) % 17) == 0)
			continue;
		make_dup_key(kb, i);
		fill_value(vb, vlen, 1, i + 3);
		{
			int rc = mdb_del(txn, dbi, &k, &v);
			if (rc != MDB_NOTFOUND)
				die_rc("dup delete value", rc);
		}
	}

	/* Grow some existing dupsets and create new primary keys. */
	for (i = 0; i < 75; i += 10) {
		unsigned j;
		make_dup_key(kb, i);
		for (j = 100; j < 112; ++j) {
			fill_value(vb, vlen, j, i + 3);
			die_rc("dup regrow", mdb_put(txn, dbi, &k, &v, 0));
		}
	}
	for (i = 75; i < 90; ++i) {
		unsigned j;
		make_dup_key(kb, i);
		for (j = 0; j < 5; ++j) {
			fill_value(vb, vlen, 200 + j, i + 3);
			die_rc("dup new keys", mdb_put(txn, dbi, &k, &v, 0));
		}
	}
	free(vb);
}

static void mutate_keydb(MDB_txn *txn, MDB_dbi dbi, size_t klen)
{
	unsigned i;
	unsigned char *kb = (unsigned char *)malloc(klen), vb[16];
	MDB_val k = {klen, kb}, v = {sizeof(vb), vb};
	check(kb != NULL, "mutate keydb alloc");
	for (i = 0; i < 120; i += 4) {
		fill_value(kb, klen, i, 1);
		die_rc("keydb delete", mdb_del(txn, dbi, &k, NULL));
	}
	for (i = 120; i < 180; ++i) {
		fill_value(kb, klen, i, 1);
		fill_value(vb, sizeof(vb), i, 2);
		die_rc("keydb add", mdb_put(txn, dbi, &k, &v, 0));
	}
	free(kb);
}

#endif


#ifdef AGG_COMPARE_STRESS_LONG
static uint64_t g_rng = AGG_COMPARE_LONG_SEED;

static uint64_t rng64(void)
{
	/* xorshift64*: deterministic and independent of libc rand(). */
	uint64_t x = g_rng;
	x ^= x >> 12;
	x ^= x << 25;
	x ^= x >> 27;
	g_rng = x;
	return x * UINT64_C(2685821657736338717);
}

static unsigned rng_u(unsigned n)
{
	check(n != 0, "rng modulus");
	return (unsigned)(rng64() % n);
}

static void set_round_phase(const char *kind, unsigned round)
{
	snprintf(g_phase_buf, sizeof(g_phase_buf), "long-%s-%u", kind, round);
	g_phase = g_phase_buf;
}

static void set_op_phase(const char *kind, unsigned round, unsigned op)
{
	snprintf(g_phase_buf, sizeof(g_phase_buf), "long-r%u-op%u-%s", round, op, kind);
	g_phase = g_phase_buf;
}

static int key_exists(MDB_txn *txn, MDB_dbi dbi, const MDB_val *key)
{
	MDB_val k = *key, data = {0, NULL};
	int rc = mdb_get(txn, dbi, &k, &data);
	if (rc == MDB_SUCCESS)
		return 1;
	check(rc == MDB_NOTFOUND, "key existence lookup");
	return 0;
}

static int dup_pair_exists(MDB_txn *txn, MDB_dbi dbi,
	const MDB_val *key, const MDB_val *data)
{
	/* Do not use MDB_GET_BOTH here. This helper is an oracle for mutation
	 * semantics, so it must not depend on the exact-match API being tested.
	 * Scan only the duplicate run for the requested primary key. */
	MDB_cursor *c = NULL;
	MDB_val k = *key, v = {0, NULL};
	int rc;
	die_rc("dup existence cursor", mdb_cursor_open(txn, dbi, &c));
	rc = mdb_cursor_get(c, &k, &v, MDB_SET);
	while (rc == MDB_SUCCESS) {
		if (valcmp(&v, data) == 0) {
			mdb_cursor_close(c);
			return 1;
		}
		rc = mdb_cursor_get(c, &k, &v, MDB_NEXT_DUP);
	}
	mdb_cursor_close(c);
	check(rc == MDB_NOTFOUND, "dup pair existence scan");
	return 0;
}

typedef struct DbFingerprint {
	uint64_t h;
	uint64_t n;
} DbFingerprint;

typedef struct StateFingerprint {
	DbFingerprint plain, dup, keydb;
} StateFingerprint;

static uint64_t fp_mix_byte(uint64_t h, unsigned char b)
{
	h ^= b;
	return h * UINT64_C(1099511628211);
}

static uint64_t fp_mix_u64(uint64_t h, uint64_t v)
{
	unsigned i;
	for (i = 0; i < 8; ++i)
		h = fp_mix_byte(h, (unsigned char)(v >> (i * 8)));
	return h;
}

static DbFingerprint db_fingerprint(MDB_txn *txn, MDB_dbi dbi)
{
	DbFingerprint f = {UINT64_C(1469598103934665603), 0};
	MDB_cursor *c = NULL;
	MDB_val k = {0}, v = {0};
	int rc;
	die_rc("fingerprint cursor", mdb_cursor_open(txn, dbi, &c));
	for (rc = mdb_cursor_get(c, &k, &v, MDB_FIRST);
	     rc == MDB_SUCCESS;
	     rc = mdb_cursor_get(c, &k, &v, MDB_NEXT)) {
		size_t i;
		f.h = fp_mix_u64(f.h, k.mv_size);
		for (i = 0; i < k.mv_size; ++i)
			f.h = fp_mix_byte(f.h, ((const unsigned char *)k.mv_data)[i]);
		f.h = fp_mix_u64(f.h, v.mv_size);
		for (i = 0; i < v.mv_size; ++i)
			f.h = fp_mix_byte(f.h, ((const unsigned char *)v.mv_data)[i]);
		f.n++;
	}
	check(rc == MDB_NOTFOUND, "fingerprint scan end");
	mdb_cursor_close(c);
	f.h = fp_mix_u64(f.h, f.n);
	return f;
}

static StateFingerprint state_fingerprint(MDB_txn *txn, MDB_dbi plain,
	MDB_dbi dup, MDB_dbi keydb)
{
	StateFingerprint s;
	s.plain = db_fingerprint(txn, plain);
	s.dup = db_fingerprint(txn, dup);
	s.keydb = db_fingerprint(txn, keydb);
	return s;
}

static void expect_state_fingerprint(const StateFingerprint *a,
	const StateFingerprint *b, const char *where)
{
	if (a->plain.h != b->plain.h || a->plain.n != b->plain.n ||
	    a->dup.h != b->dup.h || a->dup.n != b->dup.n ||
	    a->keydb.h != b->keydb.h || a->keydb.n != b->keydb.n) {
		fprintf(stderr,
			"FAIL[%s/%s]: state fingerprint mismatch %s "
			"plain=%016llx/%llu:%016llx/%llu "
			"dup=%016llx/%llu:%016llx/%llu "
			"key=%016llx/%llu:%016llx/%llu\n",
			TEST_IMPL_NAME, g_phase, where,
			(unsigned long long)a->plain.h, (unsigned long long)a->plain.n,
			(unsigned long long)b->plain.h, (unsigned long long)b->plain.n,
			(unsigned long long)a->dup.h, (unsigned long long)a->dup.n,
			(unsigned long long)b->dup.h, (unsigned long long)b->dup.n,
			(unsigned long long)a->keydb.h, (unsigned long long)a->keydb.n,
			(unsigned long long)b->keydb.h, (unsigned long long)b->keydb.n);
		exit(5);
	}
}

static void long_plain_op(MDB_txn *txn, MDB_dbi dbi, size_t base_vlen,
	unsigned round, unsigned op)
{
	set_op_phase("plain", round, op);
	unsigned id;
	unsigned action = rng_u(100);
	unsigned char kb[8];
	MDB_val k = {8, kb};

	/* Mix a stable interior hot set with a larger universe and a moving right
	 * edge. This repeatedly revisits separators while also forcing growth. */
	if (action < 18)
		id = 700u + rng_u(350);
	else if (action < 28)
		id = 4096u + round * 13u + (op % 17u);
	else
		id = rng_u(4096);
	make_plain_key(kb, id);
	snprintf(g_phase_buf, sizeof(g_phase_buf),
		"long-r%u-op%u-plain-a%u-k%u", round, op, action, id);
	g_phase = g_phase_buf;

	if (action < 70) {
		size_t vlen;
		unsigned char *vb;
		MDB_val v;
		/* Variable-size replacement is deliberate: small/medium values perturb
		 * leaf packing; occasional >page values exercise overflow pages. */
		switch (rng_u(16)) {
		case 0:  vlen = 5200u + rng_u(2600); break;
		case 1:  vlen = 900u + rng_u(700); break;
		case 2:  vlen = base_vlen + 200u + rng_u(300); break;
		default: vlen = base_vlen + rng_u(64); break;
		}
		vb = (unsigned char *)malloc(vlen);
		check(vb != NULL, "long plain alloc");
		fill_value(vb, vlen, id ^ (round * 131u), op ^ (unsigned)rng64());
		v.mv_size = vlen; v.mv_data = vb;
		die_rc("long plain put", mdb_put(txn, dbi, &k, &v, 0));
		free(vb);
	} else {
		int existed = key_exists(txn, dbi, &k);
		int rc = mdb_del(txn, dbi, &k, NULL);
		check((existed && rc == MDB_SUCCESS) || (!existed && rc == MDB_NOTFOUND),
			"plain delete result matches pre-state");
		check(!key_exists(txn, dbi, &k), "plain delete removes key");
	}
}

static void long_dup_value(unsigned char *vb, size_t vlen,
	unsigned keyid, unsigned dupid)
{
	/* Same value family as the initial dup population for low dupids, so exact
	 * deletion hits both old and newly-created duplicates. */
	fill_value(vb, vlen, dupid + 1u, keyid + 3u);
}

static void long_dup_op(MDB_txn *txn, MDB_dbi dbi, size_t vlen,
	unsigned round, unsigned op)
{
	set_op_phase("dup", round, op);
	unsigned action = rng_u(100);
	unsigned keyid = (action < 20) ? rng_u(40) : rng_u(320);
	unsigned dupid = rng_u(96);
	snprintf(g_phase_buf, sizeof(g_phase_buf),
		"long-r%u-op%u-dup-a%u-k%u-d%u", round, op, action, keyid, dupid);
	g_phase = g_phase_buf;
	unsigned char kb[8], *vb = (unsigned char *)malloc(vlen);
	MDB_val k = {8, kb}, v = {vlen, vb};
	check(vb != NULL, "long dup alloc");
	make_dup_key(kb, keyid);
	long_dup_value(vb, vlen, keyid, dupid);

	if (action < 58) {
		die_rc("long dup put", mdb_put(txn, dbi, &k, &v, 0));
	} else if (action < 83) {
		int existed = dup_pair_exists(txn, dbi, &k, &v);
		int rc = mdb_del(txn, dbi, &k, &v);
		check((existed && rc == MDB_SUCCESS) || (!existed && rc == MDB_NOTFOUND),
			"DUPSORT exact delete result matches pre-state");
		check(!dup_pair_exists(txn, dbi, &k, &v),
			"DUPSORT exact delete removes requested pair");
	} else if (action < 91) {
		int existed = key_exists(txn, dbi, &k);
		int rc = mdb_del(txn, dbi, &k, NULL);
		check((existed && rc == MDB_SUCCESS) || (!existed && rc == MDB_NOTFOUND),
			"DUPSORT full-key delete result matches pre-state");
		check(!key_exists(txn, dbi, &k), "DUPSORT full-key delete removes key");
	} else {
		/* Burst growth makes a primary key repeatedly cross singleton, inline
		 * subpage and persistent duplicate-subDB representations. */
		unsigned j, base = 128u + ((round * 17u + op) & 127u);
		for (j = 0; j < 12; ++j) {
			long_dup_value(vb, vlen, keyid, base + j);
			die_rc("long dup burst", mdb_put(txn, dbi, &k, &v, 0));
		}
	}
	free(vb);
}

static void long_keydb_op(MDB_txn *txn, MDB_dbi dbi, size_t klen,
	unsigned round, unsigned op)
{
	set_op_phase("keydb", round, op);
	unsigned action = rng_u(100);
	unsigned id = (action < 15) ? (100u + rng_u(180)) : rng_u(900);
	unsigned char *kb = (unsigned char *)malloc(klen), vb[16];
	MDB_val k = {klen, kb}, v = {sizeof(vb), vb};
	check(kb != NULL, "long keydb alloc");
	fill_value(kb, klen, id, 1);
	snprintf(g_phase_buf, sizeof(g_phase_buf),
		"long-r%u-op%u-keydb-a%u-k%u", round, op, action, id);
	g_phase = g_phase_buf;
	if (action < 68) {
		fill_value(vb, sizeof(vb), id ^ round, op + 2u);
		die_rc("long keydb put", mdb_put(txn, dbi, &k, &v, 0));
	} else {
		int existed = key_exists(txn, dbi, &k);
		int rc = mdb_del(txn, dbi, &k, NULL);
		check((existed && rc == MDB_SUCCESS) || (!existed && rc == MDB_NOTFOUND),
			"key-source delete result matches pre-state");
		check(!key_exists(txn, dbi, &k), "key-source delete removes key");
	}
	free(kb);
}

static void long_mutate_batch(MDB_txn *txn, MDB_dbi plain, MDB_dbi dup,
	MDB_dbi keydb, size_t vlen, size_t klen, unsigned round, unsigned nops)
{
	unsigned i;
	for (i = 0; i < nops; ++i) {
		unsigned which = rng_u(100);
		if (which < 52)
			long_plain_op(txn, plain, vlen, round, i);
		else if (which < 84)
			long_dup_op(txn, dup, vlen, round, i);
		else
			long_keydb_op(txn, keydb, klen, round, i);
	}
}

static void reopen_env(const char *path, MDB_env **envp)
{
	MDB_env *env;
	mdb_env_close(*envp);
	*envp = NULL;
	die_rc("long reopen env_create", mdb_env_create(&env));
	die_rc("long reopen mapsize", mdb_env_set_mapsize(env, 256u * 1024u * 1024u));
	die_rc("long reopen maxdbs", mdb_env_set_maxdbs(env, 8));
	die_rc("long reopen env_open", mdb_env_open(env, path, 0, 0664));
	*envp = env;
}
#endif

#ifdef AGG_COMPARE_STRESS_LARGE_KEYS
typedef struct LargeTreeStats {
	unsigned peak_depth;
	size_t peak_branch_pages;
	size_t peak_leaf_pages;
	unsigned depth_changes;
	unsigned last_depth;
} LargeTreeStats;

static uint64_t large_mix64(uint64_t x)
{
	x ^= x >> 30;
	x *= UINT64_C(0xbf58476d1ce4e5b9);
	x ^= x >> 27;
	x *= UINT64_C(0x94d049bb133111eb);
	x ^= x >> 31;
	return x;
}

static void large_be64(unsigned char *p, uint64_t x)
{
	unsigned i;
	for (i = 0; i < 8; ++i)
		p[i] = (unsigned char)(x >> (56u - i * 8u));
}

static size_t large_near_max(size_t maxkey)
{
	size_t slack = maxkey >= 64 ? 16u : 0u;
	check(maxkey >= 16, "large-key maxkey unexpectedly small");
	return maxkey - slack;
}

static size_t large_key_size(size_t maxkey, uint64_t id)
{
	size_t hi = large_near_max(maxkey);
#if AGG_COMPARE_LARGE_KEYS_VARIABLE
	size_t lo = (maxkey + 1u) / 2u;
	size_t span;
	if (lo < 8u)
		lo = 8u;
	check(lo <= hi, "large-key variable interval");
	span = hi - lo + 1u;
	return lo + (size_t)(large_mix64(id ^ AGG_COMPARE_LARGE_KEYS_SEED) % span);
#else
	(void)id;
	return hi;
#endif
}

static void large_make_key(unsigned char *buf, size_t len, uint64_t id)
{
	uint64_t x;
	size_t i;
	check(len >= 8u, "large-key key length");
	large_be64(buf, id);
	x = large_mix64(id ^ AGG_COMPARE_LARGE_KEYS_SEED ^ (uint64_t)len);
	for (i = 8; i < len; ++i) {
		if ((i & 7u) == 0u)
			x = large_mix64(x ^ (uint64_t)i);
		buf[i] = (unsigned char)(x >> ((i & 7u) * 8u));
	}
}

static void large_make_value(unsigned char *buf, size_t len,
	uint64_t id, unsigned tag)
{
	size_t i;
	uint64_t x = large_mix64(id ^ ((uint64_t)tag << 48) ^ AGG_COMPARE_LARGE_KEYS_SEED);
	check(len >= 8u, "large-key value length");
	for (i = 0; i < len; ++i) {
		if ((i & 7u) == 0u)
			x = large_mix64(x ^ (uint64_t)i);
		buf[i] = (unsigned char)(x >> ((i & 7u) * 8u));
	}
}

static int large_delete_selected(uint64_t id, unsigned round)
{
	/* Alternating contiguous blocks deliberately empty whole runs of leaves.
	 * Varying the block width moves merge boundaries between rounds. */
	uint64_t block = 24u + (uint64_t)(round % 4u) * 11u;
	return (((id / block) + round) & 1u) == 0u;
}

static void large_put_ids(MDB_txn *txn, MDB_dbi plain, MDB_dbi dup,
	size_t maxkey, size_t vlen, unsigned count, int odd, int reverse)
{
	unsigned char *kb = (unsigned char *)malloc(maxkey);
	unsigned char *pv = (unsigned char *)malloc(vlen);
	unsigned char *dv = (unsigned char *)malloc(vlen);
	unsigned i;
	check(kb && pv && dv, "large-key put buffers");

	for (i = 0; i < count; ++i) {
		unsigned pos = reverse ? count - 1u - i : i;
		uint64_t id = (uint64_t)pos * 2u + (unsigned)odd;
		size_t klen = large_key_size(maxkey, id);
		MDB_val k = {klen, kb}, p = {vlen, pv}, d = {vlen, dv};
		large_make_key(kb, klen, id);
		large_make_value(pv, vlen, id, 1u);
		large_make_value(dv, vlen, id, 2u);
		die_rc("large plain put", mdb_put(txn, plain, &k, &p, 0));
		die_rc("large dup put", mdb_put(txn, dup, &k, &d, 0));
	}

	free(dv);
	free(pv);
	free(kb);
}

static void large_delete_wave(MDB_txn *txn, MDB_dbi plain, MDB_dbi dup,
	size_t maxkey, size_t vlen, unsigned count, unsigned round,
	int do_plain, int do_dup)
{
	unsigned char *kb = (unsigned char *)malloc(maxkey);
	unsigned char *dv = (unsigned char *)malloc(vlen);
	uint64_t total = (uint64_t)count * 2u;
	uint64_t id;
	check(kb && dv, "large-key delete buffers");

	if (do_plain) {
		for (id = 0; id < total; ++id) {
			MDB_val k, got = {0, NULL};
			int rc;
			size_t klen;
			if (!large_delete_selected(id, round))
				continue;
			snprintf(g_phase_buf, sizeof(g_phase_buf),
				"large-plain-r%u-del-k%llu", round,
				(unsigned long long)id);
			g_phase = g_phase_buf;
			klen = large_key_size(maxkey, id);
			large_make_key(kb, klen, id);
			k.mv_size = klen; k.mv_data = kb;
			rc = mdb_get(txn, plain, &k, &got);
			check(rc == MDB_SUCCESS, "large plain pre-delete exists");
			die_rc("large plain delete", mdb_del(txn, plain, &k, NULL));
			rc = mdb_get(txn, plain, &k, &got);
			check(rc == MDB_NOTFOUND, "large plain delete removes key");
		}
	}

	if (do_dup) {
		for (id = 0; id < total; ++id) {
			MDB_val k, d, got = {0, NULL};
			int rc;
			size_t klen;
			if (!large_delete_selected(id, round))
				continue;
			snprintf(g_phase_buf, sizeof(g_phase_buf),
				"large-dup-r%u-del-k%llu", round,
				(unsigned long long)id);
			g_phase = g_phase_buf;
			klen = large_key_size(maxkey, id);
			large_make_key(kb, klen, id);
			large_make_value(dv, vlen, id, 2u);
			k.mv_size = klen; k.mv_data = kb;
			d.mv_size = vlen; d.mv_data = dv;
			/* One duplicate value per primary key: an exact DUPSORT delete
			 * removes the whole outer-tree key while exercising exact-pair delete. */
			die_rc("large dup exact delete", mdb_del(txn, dup, &k, &d));
			rc = mdb_get(txn, dup, &k, &got);
			check(rc == MDB_NOTFOUND, "large dup exact delete removes outer key");
		}
	}

	free(dv);
	free(kb);
}

static void large_regrow_wave(MDB_txn *txn, MDB_dbi plain, MDB_dbi dup,
	size_t maxkey, size_t vlen, unsigned count, unsigned round,
	int do_plain, int do_dup)
{
	unsigned char *kb = (unsigned char *)malloc(maxkey);
	unsigned char *pv = (unsigned char *)malloc(vlen);
	unsigned char *dv = (unsigned char *)malloc(vlen);
	uint64_t total = (uint64_t)count * 2u;
	uint64_t n;
	check(kb && pv && dv, "large-key regrow buffers");

	/* Reverse traversal ensures re-growth is not just the inverse of the
	 * original ascending construction. */
	for (n = total; n != 0; --n) {
		uint64_t id = n - 1u;
		size_t klen;
		MDB_val k, p, d;
		if (!large_delete_selected(id, round))
			continue;
		snprintf(g_phase_buf, sizeof(g_phase_buf),
			"large-%s-r%u-regrow-k%llu",
			do_plain ? "plain" : "dup", round,
			(unsigned long long)id);
		g_phase = g_phase_buf;
		klen = large_key_size(maxkey, id);
		large_make_key(kb, klen, id);
		large_make_value(pv, vlen, id, 1u);
		large_make_value(dv, vlen, id, 2u);
		k.mv_size = klen; k.mv_data = kb;
		p.mv_size = vlen; p.mv_data = pv;
		d.mv_size = vlen; d.mv_data = dv;
		if (do_plain)
			die_rc("large plain regrow", mdb_put(txn, plain, &k, &p, 0));
		if (do_dup)
			die_rc("large dup regrow", mdb_put(txn, dup, &k, &d, 0));
	}

	free(dv);
	free(pv);
	free(kb);
}

static void large_note_stats(MDB_txn *txn, MDB_dbi dbi, const char *db,
	const char *phase, size_t maxkey, LargeTreeStats *track)
{
	MDB_stat st;
	die_rc("large mdb_stat", mdb_stat(txn, dbi, &st));
	if (track->last_depth && track->last_depth != st.ms_depth)
		track->depth_changes++;
	track->last_depth = st.ms_depth;
	if (st.ms_depth > track->peak_depth)
		track->peak_depth = st.ms_depth;
	if (st.ms_branch_pages > track->peak_branch_pages)
		track->peak_branch_pages = st.ms_branch_pages;
	if (st.ms_leaf_pages > track->peak_leaf_pages)
		track->peak_leaf_pages = st.ms_leaf_pages;
	fprintf(stderr,
		"STRUCT impl=%s profile=%s phase=%s db=%s maxkey=%zu key_min=%zu key_max=%zu "
		"depth=%u branch=%zu leaf=%zu overflow=%zu entries=%zu\n",
		TEST_IMPL_NAME,
#if AGG_COMPARE_LARGE_KEYS_VARIABLE
		"large-key-half-to-near-max",
#else
		"large-key-near-max",
#endif
		phase, db, maxkey,
#if AGG_COMPARE_LARGE_KEYS_VARIABLE
		(maxkey + 1u) / 2u,
#else
		large_near_max(maxkey),
#endif
		large_near_max(maxkey), st.ms_depth, st.ms_branch_pages,
		st.ms_leaf_pages, st.ms_overflow_pages, st.ms_entries);
}
#endif

static void verify_queries(MDB_txn *txn, MDB_dbi dbi, int dupsort,
	int keyhash, int hashoff, unsigned schema)
{
	Vec x = {0};
	MDB_agg got, exp;
	size_t i;

	snapshot(txn, dbi, &x);
#ifdef AGG_COMPARE_TRACE
	fprintf(stderr, "TRACE_DB impl=%s phase=%s dbi=%u n=%zu fp=%016llx\n",
		TEST_IMPL_NAME, g_phase, (unsigned)dbi, x.n,
		(unsigned long long)vec_fingerprint(&x));
#endif
	check(x.n > 20, "enough records");

	expected(&x, dupsort, keyhash, hashoff, NULL, NULL, 0, NULL, NULL, 0, &exp);
	die_rc("totals", mdb_agg_totals(txn, dbi, &got));
	expect_agg(&got, &exp, schema, "totals");
	die_rc("range open", mdb_agg_range(txn, dbi, NULL, NULL, NULL, NULL, 0, &got));
	expect_agg(&got, &exp, schema, "range open");

	for (i = 0; i < x.n; i += x.n / 11 + 1) {
		unsigned f = (i & 1) ? MDB_AGG_PREFIX_INCL : 0;
		expected(&x, dupsort, keyhash, hashoff, NULL, NULL, 0,
			&x.v[i].k, dupsort ? &x.v[i].v : NULL, !!f, &exp);
		die_rc("prefix record", mdb_agg_prefix(txn, dbi, &x.v[i].k,
			dupsort ? &x.v[i].v : NULL, f, &got));
		expect_agg(&got, &exp, schema, "prefix record");
		if (dupsort) {
			expected(&x, dupsort, keyhash, hashoff, NULL, NULL, 0,
				&x.v[i].k, NULL, !!f, &exp);
			die_rc("prefix key", mdb_agg_prefix(txn, dbi, &x.v[i].k, NULL, f, &got));
			expect_agg(&got, &exp, schema, "prefix key");
		}
	}

	for (i = 0; i < 12; i++) {
		size_t a = (i * 7) % x.n;
		size_t b = x.n - 1 - ((i * 11) % x.n);
		size_t t;
		unsigned f = 0;
		if (a > b) { t = a; a = b; b = t; }
		if (i & 1) f |= MDB_RANGE_LOWER_INCL;
		if (i & 2) f |= MDB_RANGE_UPPER_INCL;
		expected(&x, dupsort, keyhash, hashoff,
			&x.v[a].k, dupsort ? &x.v[a].v : NULL, !!(f & MDB_RANGE_LOWER_INCL),
			&x.v[b].k, dupsort ? &x.v[b].v : NULL, !!(f & MDB_RANGE_UPPER_INCL),
			&exp);
		die_rc("range record", mdb_agg_range(txn, dbi,
			&x.v[a].k, dupsort ? &x.v[a].v : NULL,
			&x.v[b].k, dupsort ? &x.v[b].v : NULL, f, &got));
		expect_agg(&got, &exp, schema, "range record");
	}

	for (i = 0; i < x.n; i += x.n / 17 + 1) {
		MDB_val k = {0}, v = {0};
		uint64_t di = 999, rank = 999;
		die_rc("select entries", mdb_agg_select(txn, dbi,
			MDB_AGG_WEIGHT_ENTRIES, i, &k, &v, &di));
		check(valcmp(&k, &x.v[i].k) == 0 && valcmp(&v, &x.v[i].v) == 0,
			"select entries value");
		sig_u64(i); sig_bytes(k.mv_data, k.mv_size); sig_bytes(v.mv_data, v.mv_size);
		k = x.v[i].k;
		v = dupsort ? x.v[i].v : (MDB_val){0, NULL};
		die_rc("rank entries", mdb_agg_rank(txn, dbi, &k, &v,
			MDB_AGG_WEIGHT_ENTRIES, MDB_AGG_RANK_EXACT, &rank, &di));
		check(rank == i, "rank entries exact");
		sig_u64(rank); sig_u64(di);
	}

	{
		MDB_val k = x.v[x.n / 3].k;
		MDB_val v = dupsort ? x.v[x.n / 3].v : (MDB_val){0, NULL};
		uint64_t rank = UINT64_MAX, di = UINT64_MAX;
		die_rc("rank set-range exact", mdb_agg_rank(txn, dbi, &k, &v,
			MDB_AGG_WEIGHT_ENTRIES, MDB_AGG_RANK_SET_RANGE, &rank, &di));
		check(rank == x.n / 3, "rank set-range exact index");
		sig_u64(rank); sig_u64(di);
		if (!dupsort && x.v[x.n / 3].k.mv_size == 8) {
			unsigned char gap[8];
			MDB_val gk = {8, gap}, gd = {0, NULL};
			make_gap_after_8(&x.v[x.n / 3].k, gap);
			die_rc("rank set-range gap", mdb_agg_rank(txn, dbi, &gk, &gd,
				MDB_AGG_WEIGHT_ENTRIES, MDB_AGG_RANK_SET_RANGE, &rank, &di));
			check(rank == x.n / 3 + 1, "rank set-range gap index");
			sig_u64(rank); sig_u64(di);
		}
	}

	{
		size_t ki = 0, pos = 0;
		while (pos < x.n) {
			size_t first = pos;
			MDB_val k = {0}, v = {0}, empty = {0, NULL};
			uint64_t rank = 999, di = 999;
			if ((ki % 5) == 0) {
				die_rc("select keys", mdb_agg_select(txn, dbi,
					MDB_AGG_WEIGHT_KEYS, ki, &k, &v, &di));
				check(valcmp(&k, &x.v[first].k) == 0, "select key");
				sig_u64(ki); sig_bytes(k.mv_data, k.mv_size);
				k = x.v[first].k;
				die_rc("rank keys", mdb_agg_rank(txn, dbi, &k, &empty,
					MDB_AGG_WEIGHT_KEYS, MDB_AGG_RANK_EXACT, &rank, &di));
				check(rank == ki && di == 0, "rank keys exact");
				sig_u64(rank); sig_u64(di);
			}
			pos++;
			while (pos < x.n && valcmp(&x.v[pos].k, &x.v[first].k) == 0)
				pos++;
			ki++;
		}
	}

	/* Common boundary/error semantics. */
	{
		MDB_val k = {0}, v = {0};
		uint64_t di = 0;
		int rc = mdb_agg_select(txn, dbi, MDB_AGG_WEIGHT_ENTRIES, x.n, &k, &v, &di);
		check(rc == MDB_NOTFOUND, "select past end");
		if (!dupsort && x.v[x.n / 2].k.mv_size == 8) {
			unsigned char gap[8]; MDB_val gk = {8, gap}, gd = {0, NULL};
			uint64_t rank = 0;
			make_gap_after_8(&x.v[x.n / 2].k, gap);
			rc = mdb_agg_rank(txn, dbi, &gk, &gd, MDB_AGG_WEIGHT_ENTRIES,
				MDB_AGG_RANK_EXACT, &rank, &di);
			check(rc == MDB_NOTFOUND, "rank exact missing");
		}
		{
			MDB_agg z, ez;
			expected(&x, dupsort, keyhash, hashoff,
				&x.v[x.n - 1].k, dupsort ? &x.v[x.n - 1].v : NULL, 1,
				&x.v[0].k, dupsort ? &x.v[0].v : NULL, 1, &ez);
			die_rc("range inverted", mdb_agg_range(txn, dbi,
				&x.v[x.n - 1].k, dupsort ? &x.v[x.n - 1].v : NULL,
				&x.v[0].k, dupsort ? &x.v[0].v : NULL,
				MDB_RANGE_LOWER_INCL | MDB_RANGE_UPPER_INCL, &z));
			expect_agg(&z, &ez, schema, "range inverted");
		}
	}

	{
		MDB_cursor *c;
		die_rc("seek cursor open", mdb_cursor_open(txn, dbi, &c));
		for (i = 0; i < x.n; i += x.n / 13 + 1) {
			MDB_val k = {0}, v = {0};
			die_rc("cursor seek rank", mdb_agg_cursor_seek_rank(c, i, &k, &v));
			check(valcmp(&k, &x.v[i].k) == 0 && valcmp(&v, &x.v[i].v) == 0,
				"cursor seek value");
			sig_u64(i); sig_bytes(k.mv_data, k.mv_size); sig_bytes(v.mv_data, v.mv_size);
			if (i + 1 < x.n) {
				die_rc("cursor next", mdb_cursor_get(c, &k, &v, MDB_NEXT));
				check(valcmp(&k, &x.v[i + 1].k) == 0 && valcmp(&v, &x.v[i + 1].v) == 0,
					"cursor next after seek");
			}
		}
		{
			MDB_val k = {0}, v = {0};
			int rc = mdb_agg_cursor_seek_rank(c, x.n, &k, &v);
			check(rc == MDB_NOTFOUND, "cursor seek past end");
		}
		mdb_cursor_close(c);
	}

	vec_free(&x);
}

static void verify_all(MDB_txn *txn, MDB_dbi plain, MDB_dbi dup, MDB_dbi keydb,
	unsigned schema)
{
	unsigned info = 0; int off = 0;
	die_rc("agg info plain", mdb_agg_info(txn, plain, &info));
	check(info == schema, "agg info plain schema");
	die_rc("hash offset plain", mdb_get_hash_offset(txn, plain, &off));
	check(off == -1, "hash offset plain value");
	die_rc("agg info dup", mdb_agg_info(txn, dup, &info));
	check(info == schema, "agg info dup schema");
	die_rc("hash offset dup", mdb_get_hash_offset(txn, dup, &off));
	check(off == 0, "hash offset dup value");
	die_rc("agg info keydb", mdb_agg_info(txn, keydb, &info));
	check(info == (schema | MDB_AGG_HASHSOURCE_FROM_KEY), "agg info keydb schema");
	die_rc("hash offset keydb", mdb_get_hash_offset(txn, keydb, &off));
	check(off == 0, "hash offset keydb value");
	verify_queries(txn, plain, 0, 0, -1, schema);
	verify_queries(txn, dup, 1, 0, 0, schema);
	verify_queries(txn, keydb, 0, 1, 0, schema | MDB_AGG_HASHSOURCE_FROM_KEY);
#ifdef AGG_COMPARE_TRACE
	fprintf(stderr, "TRACE impl=%s phase=%s sig=%016llx\n", TEST_IMPL_NAME, g_phase,
		(unsigned long long)g_signature);
#endif
}

static void cleanup_env(const char *path)
{
	char data[512], lock[512];
	snprintf(data, sizeof(data), "%s/data.mdb", path);
	snprintf(lock, sizeof(lock), "%s/lock.mdb", path);
	unlink(data);
	unlink(lock);
	rmdir(path);
}

#ifdef AGG_COMPARE_STRESS_LARGE_KEYS
static const char *large_profile_name(void)
{
#if AGG_COMPARE_LARGE_KEYS_VARIABLE
	return "large-key-half-to-near-max";
#else
	return "large-key-near-max";
#endif
}

static void verify_large_pair(MDB_txn *txn, MDB_dbi plain, MDB_dbi dup,
	unsigned schema)
{
	unsigned info = 0;
	int off = -999;
	die_rc("large agg info plain", mdb_agg_info(txn, plain, &info));
	check(info == schema, "large plain schema");
	die_rc("large hash plain", mdb_get_hash_offset(txn, plain, &off));
	check(off == 0, "large plain hash offset");
	die_rc("large agg info dup", mdb_agg_info(txn, dup, &info));
	check(info == schema, "large dup schema");
	die_rc("large hash dup", mdb_get_hash_offset(txn, dup, &off));
	check(off == 0, "large dup hash offset");
	verify_queries(txn, plain, 0, 0, 0, schema);
	verify_queries(txn, dup, 1, 0, 0, schema);
}

static void large_verify_phase(MDB_env *env, const char *phase, size_t maxkey,
	unsigned schema, LargeTreeStats *plain_stats, LargeTreeStats *dup_stats)
{
	MDB_txn *txn;
	MDB_dbi plain, dup;
	snprintf(g_phase_buf, sizeof(g_phase_buf), "large-%s", phase);
	g_phase = g_phase_buf;
	die_rc("large verify txn", mdb_txn_begin(env, NULL, MDB_RDONLY, &txn));
	die_rc("large verify open plain", mdb_dbi_open(txn, "large_plain", 0, &plain));
	die_rc("large verify open dup", mdb_dbi_open(txn, "large_dup", 0, &dup));
	verify_large_pair(txn, plain, dup, schema);
	large_note_stats(txn, plain, "plain", phase, maxkey, plain_stats);
	large_note_stats(txn, dup, "dup", phase, maxkey, dup_stats);
	mdb_txn_abort(txn);
}

static int run_large_key_stress(void)
{
	char path[] = "/tmp/agglargeXXXXXX";
	MDB_env *env;
	MDB_txn *txn;
	MDB_dbi plain, dup;
	LargeTreeStats plain_stats = {0,0,0,0,0};
	LargeTreeStats dup_stats = {0,0,0,0,0};
	unsigned schema = MDB_AGG_ENTRIES | MDB_AGG_KEYS | MDB_AGG_HASHSUM;
	size_t vlen = MDB_HASH_SIZE + 16u;
	size_t maxkey, min_key, max_key;
	unsigned round;

	g_phase = "large-create-env";
	check(mkdtemp(path) != NULL, "large mkdtemp");
	die_rc("large env_create", mdb_env_create(&env));
	die_rc("large mapsize", mdb_env_set_mapsize(env, 256u * 1024u * 1024u));
	die_rc("large maxdbs", mdb_env_set_maxdbs(env, 8));
	die_rc("large env_open", mdb_env_open(env, path, 0, 0664));
	maxkey = (size_t)mdb_env_get_maxkeysize(env);
	max_key = large_near_max(maxkey);
#if AGG_COMPARE_LARGE_KEYS_VARIABLE
	min_key = (maxkey + 1u) / 2u;
	if (min_key < 8u)
		min_key = 8u;
#else
	min_key = max_key;
#endif
	check(min_key <= max_key, "large key interval valid");

	/* Execution parameters are part of the deterministic cross-implementation
	 * signature, but physical page statistics are reported separately. */
	sig_u64(AGG_COMPARE_LARGE_KEYS_SEED);
	sig_u64(AGG_COMPARE_LARGE_KEYS_COUNT);
	sig_u64(AGG_COMPARE_LARGE_KEYS_ROUNDS);
	sig_u64(maxkey);
	sig_u64(min_key);
	sig_u64(max_key);
	sig_u64(AGG_COMPARE_LARGE_KEYS_VARIABLE);

	fprintf(stderr,
		"LARGE-CONFIG impl=%s profile=%s hash=%d count=%u rounds=%u "
		"maxkey=%zu key_min=%zu key_max=%zu min_depth=%u\n",
		TEST_IMPL_NAME, large_profile_name(), MDB_HASH_SIZE,
		(unsigned)AGG_COMPARE_LARGE_KEYS_COUNT,
		(unsigned)AGG_COMPARE_LARGE_KEYS_ROUNDS,
		maxkey, min_key, max_key, (unsigned)AGG_COMPARE_LARGE_KEYS_MIN_DEPTH);

	/* Phase 1: a sparse ordered tree.  Only even IDs are present, leaving one
	 * deterministic insertion point between every adjacent pair. */
	g_phase = "large-sparse-grow";
	die_rc("large sparse txn", mdb_txn_begin(env, NULL, 0, &txn));
	die_rc("large create plain", mdb_dbi_open(txn, "large_plain", MDB_CREATE | schema, &plain));
	die_rc("large plain hash", mdb_set_hash_offset(txn, plain, 0));
	die_rc("large create dup", mdb_dbi_open(txn, "large_dup", MDB_CREATE | MDB_DUPSORT | schema, &dup));
	die_rc("large dup hash", mdb_set_hash_offset(txn, dup, 0));
	large_put_ids(txn, plain, dup, maxkey, vlen,
		AGG_COMPARE_LARGE_KEYS_COUNT, 0, 0);
	die_rc("large sparse commit", mdb_txn_commit(txn));
	large_verify_phase(env, "sparse-grow", maxkey, schema, &plain_stats, &dup_stats);

	/* Phase 2: fill every gap with an odd key.  These are deliberately interior
	 * inserts rather than a right-edge append workload. */
	g_phase = "large-interior-fill";
	die_rc("large fill txn", mdb_txn_begin(env, NULL, 0, &txn));
	die_rc("large fill plain", mdb_dbi_open(txn, "large_plain", 0, &plain));
	die_rc("large fill dup", mdb_dbi_open(txn, "large_dup", 0, &dup));
	large_put_ids(txn, plain, dup, maxkey, vlen,
		AGG_COMPARE_LARGE_KEYS_COUNT, 1, 0);
	die_rc("large fill commit", mdb_txn_commit(txn));
	large_verify_phase(env, "interior-fill", maxkey, schema, &plain_stats, &dup_stats);

	/* Repeatedly empty alternating contiguous key blocks and rebuild them in
	 * reverse order.  Run the plain-key churn to completion first, then repeat
	 * the same structural workload through DUPSORT exact deletes.  Keeping the
	 * two paths in separate transactions means a DUPSORT-specific failure does
	 * not prevent the plain maintenance path from receiving the full stress. */
	for (round = 0; round < AGG_COMPARE_LARGE_KEYS_ROUNDS; ++round) {
		char phase[64];
		snprintf(g_phase_buf, sizeof(g_phase_buf), "large-plain-delete-%u", round);
		g_phase = g_phase_buf;
		die_rc("large plain delete txn", mdb_txn_begin(env, NULL, 0, &txn));
		die_rc("large plain delete open", mdb_dbi_open(txn, "large_plain", 0, &plain));
		die_rc("large plain delete dup-open", mdb_dbi_open(txn, "large_dup", 0, &dup));
		fprintf(stderr, "STRUCT-PHASE impl=%s profile=%s phase=plain-delete-%u\n",
			TEST_IMPL_NAME, large_profile_name(), round);
		large_delete_wave(txn, plain, dup, maxkey, vlen,
			AGG_COMPARE_LARGE_KEYS_COUNT, round, 1, 0);
		die_rc("large plain delete commit", mdb_txn_commit(txn));
		snprintf(phase, sizeof(phase), "plain-delete-%u", round);
		large_verify_phase(env, phase, maxkey, schema, &plain_stats, &dup_stats);

		snprintf(g_phase_buf, sizeof(g_phase_buf), "large-plain-regrow-%u", round);
		g_phase = g_phase_buf;
		die_rc("large plain regrow txn", mdb_txn_begin(env, NULL, 0, &txn));
		die_rc("large plain regrow open", mdb_dbi_open(txn, "large_plain", 0, &plain));
		die_rc("large plain regrow dup-open", mdb_dbi_open(txn, "large_dup", 0, &dup));
		fprintf(stderr, "STRUCT-PHASE impl=%s profile=%s phase=plain-regrow-%u\n",
			TEST_IMPL_NAME, large_profile_name(), round);
		large_regrow_wave(txn, plain, dup, maxkey, vlen,
			AGG_COMPARE_LARGE_KEYS_COUNT, round, 1, 0);
		die_rc("large plain regrow commit", mdb_txn_commit(txn));
		snprintf(phase, sizeof(phase), "plain-regrow-%u", round);
		large_verify_phase(env, phase, maxkey, schema, &plain_stats, &dup_stats);
	}

	for (round = 0; round < AGG_COMPARE_LARGE_KEYS_ROUNDS; ++round) {
		char phase[64];
		snprintf(g_phase_buf, sizeof(g_phase_buf), "large-dup-delete-%u", round);
		g_phase = g_phase_buf;
		die_rc("large dup delete txn", mdb_txn_begin(env, NULL, 0, &txn));
		die_rc("large dup delete plain-open", mdb_dbi_open(txn, "large_plain", 0, &plain));
		die_rc("large dup delete open", mdb_dbi_open(txn, "large_dup", 0, &dup));
		fprintf(stderr, "STRUCT-PHASE impl=%s profile=%s phase=dup-delete-%u\n",
			TEST_IMPL_NAME, large_profile_name(), round);
		large_delete_wave(txn, plain, dup, maxkey, vlen,
			AGG_COMPARE_LARGE_KEYS_COUNT, round, 0, 1);
		die_rc("large dup delete commit", mdb_txn_commit(txn));
		snprintf(phase, sizeof(phase), "dup-delete-%u", round);
		large_verify_phase(env, phase, maxkey, schema, &plain_stats, &dup_stats);

		snprintf(g_phase_buf, sizeof(g_phase_buf), "large-dup-regrow-%u", round);
		g_phase = g_phase_buf;
		die_rc("large dup regrow txn", mdb_txn_begin(env, NULL, 0, &txn));
		die_rc("large dup regrow plain-open", mdb_dbi_open(txn, "large_plain", 0, &plain));
		die_rc("large dup regrow open", mdb_dbi_open(txn, "large_dup", 0, &dup));
		fprintf(stderr, "STRUCT-PHASE impl=%s profile=%s phase=dup-regrow-%u\n",
			TEST_IMPL_NAME, large_profile_name(), round);
		large_regrow_wave(txn, plain, dup, maxkey, vlen,
			AGG_COMPARE_LARGE_KEYS_COUNT, round, 0, 1);
		die_rc("large dup regrow commit", mdb_txn_commit(txn));
		snprintf(phase, sizeof(phase), "dup-regrow-%u", round);
		large_verify_phase(env, phase, maxkey, schema, &plain_stats, &dup_stats);
	}

	/* A final environment reopen catches persistence problems independently of
	 * the write transaction that produced the final structure. */
	mdb_env_close(env);
	g_phase = "large-reopen-env";
	die_rc("large reopen env_create", mdb_env_create(&env));
	die_rc("large reopen mapsize", mdb_env_set_mapsize(env, 256u * 1024u * 1024u));
	die_rc("large reopen maxdbs", mdb_env_set_maxdbs(env, 8));
	die_rc("large reopen env_open", mdb_env_open(env, path, 0, 0664));
	check((size_t)mdb_env_get_maxkeysize(env) == maxkey, "large maxkey stable after reopen");
	large_verify_phase(env, "reopen-final", maxkey, schema, &plain_stats, &dup_stats);

	check(plain_stats.peak_depth >= AGG_COMPARE_LARGE_KEYS_MIN_DEPTH,
		"large plain reached requested structural depth");
	check(dup_stats.peak_depth >= AGG_COMPARE_LARGE_KEYS_MIN_DEPTH,
		"large dup reached requested structural depth");
	fprintf(stderr,
		"STRUCT-SUMMARY impl=%s profile=%s plain_peak_depth=%u plain_peak_branch=%zu "
		"plain_peak_leaf=%zu plain_depth_changes=%u dup_peak_depth=%u "
		"dup_peak_branch=%zu dup_peak_leaf=%zu dup_depth_changes=%u\n",
		TEST_IMPL_NAME, large_profile_name(),
		plain_stats.peak_depth, plain_stats.peak_branch_pages,
		plain_stats.peak_leaf_pages, plain_stats.depth_changes,
		dup_stats.peak_depth, dup_stats.peak_branch_pages,
		dup_stats.peak_leaf_pages, dup_stats.depth_changes);

	mdb_env_close(env);
	cleanup_env(path);
	printf("PASS impl=%s mode=%s hash=%d signature=%016llx\n",
		TEST_IMPL_NAME, large_profile_name(), MDB_HASH_SIZE,
		(unsigned long long)g_signature);
	return 0;
}
#endif

int main(void)
{
#ifdef AGG_COMPARE_STRESS_LARGE_KEYS
	return run_large_key_stress();
#endif
	char path[] = "/tmp/aggcmpXXXXXX";
	MDB_env *env;
	MDB_txn *txn;
	MDB_dbi plain, dup, keydb;
	size_t vlen = MDB_HASH_SIZE + 24;
	size_t klen = MDB_HASH_SIZE + 8;
	unsigned schema = MDB_AGG_ENTRIES | MDB_AGG_KEYS | MDB_AGG_HASHSUM;

	g_phase = "create";
	check(mkdtemp(path) != NULL, "mkdtemp");
	die_rc("env_create", mdb_env_create(&env));
	die_rc("mapsize", mdb_env_set_mapsize(env, 128u * 1024u * 1024u));
	die_rc("maxdbs", mdb_env_set_maxdbs(env, 8));
	die_rc("env_open", mdb_env_open(env, path, 0, 0664));

	die_rc("txn begin", mdb_txn_begin(env, NULL, 0, &txn));
	die_rc("open plain", mdb_dbi_open(txn, "plain", MDB_CREATE | schema, &plain));
	die_rc("hash plain", mdb_set_hash_offset(txn, plain, -1));
	put_plain(txn, plain, 900, vlen);

	die_rc("open dup", mdb_dbi_open(txn, "dup", MDB_CREATE | MDB_DUPSORT | schema, &dup));
	die_rc("hash dup", mdb_set_hash_offset(txn, dup, 0));
	put_dups(txn, dup, 75, 9, vlen);

	die_rc("open keydb", mdb_dbi_open(txn, "keydb",
		MDB_CREATE | schema | MDB_AGG_HASHSOURCE_FROM_KEY, &keydb));
	die_rc("hash keydb", mdb_set_hash_offset(txn, keydb, 0));
	{
		unsigned i;
		unsigned char *kb = (unsigned char *)malloc(klen), vb[16];
		MDB_val k = {klen, kb}, v = {sizeof(vb), vb};
		check(kb != NULL, "keydb alloc");
		for (i = 0; i < 120; i++) {
			fill_value(kb, klen, i, 1);
			fill_value(vb, sizeof(vb), i, 2);
			die_rc("keydb put", mdb_put(txn, keydb, &k, &v, 0));
		}
		free(kb);
	}
	die_rc("commit create", mdb_txn_commit(txn));

	g_phase = "initial-read";
	die_rc("read txn", mdb_txn_begin(env, NULL, MDB_RDONLY, &txn));
	die_rc("reopen plain", mdb_dbi_open(txn, "plain", 0, &plain));
	die_rc("reopen dup", mdb_dbi_open(txn, "dup", 0, &dup));
	die_rc("reopen keydb", mdb_dbi_open(txn, "keydb", 0, &keydb));
	verify_all(txn, plain, dup, keydb, schema);
	mdb_txn_abort(txn);


#ifdef AGG_COMPARE_STRESS_LONG
	{
		unsigned round;
		/* Make the execution parameters part of the deterministic signature. */
		sig_u64(AGG_COMPARE_LONG_SEED);
		sig_u64(AGG_COMPARE_LONG_ROUNDS);
		sig_u64(AGG_COMPARE_LONG_OPS);
		fprintf(stderr, "LONG_CONFIG impl=%s seed=%016llx rounds=%u ops=%u verify_every=%u reopen_every=%u\n",
			TEST_IMPL_NAME, (unsigned long long)AGG_COMPARE_LONG_SEED,
			(unsigned)AGG_COMPARE_LONG_ROUNDS, (unsigned)AGG_COMPARE_LONG_OPS,
			(unsigned)AGG_COMPARE_LONG_VERIFY_EVERY,
			(unsigned)AGG_COMPARE_LONG_REOPEN_EVERY);
		for (round = 0; round < AGG_COMPARE_LONG_ROUNDS; ++round) {
			int abort_round = ((round % 19u) == 7u);
			int reopen_verify = abort_round || (AGG_COMPARE_LONG_VERIFY_EVERY &&
				((round + 1u) % AGG_COMPARE_LONG_VERIFY_EVERY) == 0u);
			StateFingerprint pre_round = {{0,0},{0,0},{0,0}};
			StateFingerprint post_round = {{0,0},{0,0},{0,0}};
			set_round_phase("write", round);
			die_rc("long txn begin", mdb_txn_begin(env, NULL, 0, &txn));
			die_rc("long open plain", mdb_dbi_open(txn, "plain", 0, &plain));
			die_rc("long open dup", mdb_dbi_open(txn, "dup", 0, &dup));
			die_rc("long open keydb", mdb_dbi_open(txn, "keydb", 0, &keydb));
			if (abort_round)
				pre_round = state_fingerprint(txn, plain, dup, keydb);
			long_mutate_batch(txn, plain, dup, keydb, vlen, klen,
				round, AGG_COMPARE_LONG_OPS);

			/* Nested transactions are part of the common LMDB API. Alternate
			 * committed and aborted children while retaining the parent writer. */
			if ((round % 11u) == 3u) {
				MDB_txn *child = NULL;
				int child_abort = ((round / 11u) & 1u) != 0;
				StateFingerprint child_before = {{0,0},{0,0},{0,0}};
				if (child_abort)
					child_before = state_fingerprint(txn, plain, dup, keydb);
				set_round_phase("child", round);
				die_rc("long child begin", mdb_txn_begin(env, txn, 0, &child));
				long_mutate_batch(child, plain, dup, keydb, vlen, klen,
					round ^ 0x4000u, AGG_COMPARE_LONG_OPS / 8u + 1u);
				if (child_abort) {
					StateFingerprint child_after;
					mdb_txn_abort(child);
					child_after = state_fingerprint(txn, plain, dup, keydb);
					expect_state_fingerprint(&child_before, &child_after,
						"nested abort isolation");
				} else {
					die_rc("long child commit", mdb_txn_commit(child));
				}
			}

			if (!abort_round && AGG_COMPARE_LONG_VERIFY_EVERY &&
			    (round % AGG_COMPARE_LONG_VERIFY_EVERY) == 0u) {
				set_round_phase("live-verify", round);
				verify_all(txn, plain, dup, keydb, schema);
			}

			if (!abort_round && reopen_verify)
				post_round = state_fingerprint(txn, plain, dup, keydb);

			set_round_phase(abort_round ? "abort" : "commit", round);
			if (abort_round)
				mdb_txn_abort(txn);
			else
				die_rc("long txn commit", mdb_txn_commit(txn));

			/* Periodically close/reopen the environment, not merely the txn. */
			if (AGG_COMPARE_LONG_REOPEN_EVERY &&
			    ((round + 1u) % AGG_COMPARE_LONG_REOPEN_EVERY) == 0u) {
				set_round_phase("env-reopen", round);
				reopen_env(path, &env);
			}

			if (reopen_verify) {
				StateFingerprint reopened;
				set_round_phase("reopen-verify", round);
				die_rc("long verify txn", mdb_txn_begin(env, NULL, MDB_RDONLY, &txn));
				die_rc("long verify plain", mdb_dbi_open(txn, "plain", 0, &plain));
				die_rc("long verify dup", mdb_dbi_open(txn, "dup", 0, &dup));
				die_rc("long verify keydb", mdb_dbi_open(txn, "keydb", 0, &keydb));
				reopened = state_fingerprint(txn, plain, dup, keydb);
				if (abort_round)
					expect_state_fingerprint(&pre_round, &reopened, "top-level abort isolation");
				else
					expect_state_fingerprint(&post_round, &reopened, "commit persistence");
				verify_all(txn, plain, dup, keydb, schema);
				mdb_txn_abort(txn);
			}
		}

		g_phase = "long-final";
		die_rc("long final txn", mdb_txn_begin(env, NULL, MDB_RDONLY, &txn));
		die_rc("long final plain", mdb_dbi_open(txn, "plain", 0, &plain));
		die_rc("long final dup", mdb_dbi_open(txn, "dup", 0, &dup));
		die_rc("long final keydb", mdb_dbi_open(txn, "keydb", 0, &keydb));
		verify_all(txn, plain, dup, keydb, schema);
		mdb_txn_abort(txn);
	}
#endif

#if defined(AGG_COMPARE_STRESS_MUTATIONS) || defined(AGG_COMPARE_STRESS_GROWTH)
	g_phase = "mutate";
	die_rc("write txn", mdb_txn_begin(env, NULL, 0, &txn));
	die_rc("reopen plain w", mdb_dbi_open(txn, "plain", 0, &plain));
	die_rc("reopen dup w", mdb_dbi_open(txn, "dup", 0, &dup));
	die_rc("reopen keydb w", mdb_dbi_open(txn, "keydb", 0, &keydb));
#ifdef AGG_COMPARE_STRESS_MUTATIONS
	mutate_plain(txn, plain, vlen);
	mutate_dups(txn, dup, vlen);
	mutate_keydb(txn, keydb, klen);
#else
	grow_plain(txn, plain, vlen);
	grow_dups(txn, dup, vlen);
	grow_keydb(txn, keydb, klen);
#endif

	g_phase = "mutated-live";
	verify_all(txn, plain, dup, keydb, schema);
	g_phase = "mutate-commit";
	die_rc("commit mutate", mdb_txn_commit(txn));

	g_phase = "mutated-reopen";
	die_rc("read txn 2", mdb_txn_begin(env, NULL, MDB_RDONLY, &txn));
	die_rc("reopen plain 2", mdb_dbi_open(txn, "plain", 0, &plain));
	die_rc("reopen dup 2", mdb_dbi_open(txn, "dup", 0, &dup));
	die_rc("reopen keydb 2", mdb_dbi_open(txn, "keydb", 0, &keydb));
	verify_all(txn, plain, dup, keydb, schema);
	mdb_txn_abort(txn);
#endif

	mdb_env_close(env);
	cleanup_env(path);
	printf("PASS impl=%s mode=%s hash=%d signature=%016llx\n",
		TEST_IMPL_NAME,
#ifdef AGG_COMPARE_STRESS_LONG
		"long",
#elif defined(AGG_COMPARE_STRESS_MUTATIONS)
		"stress",
#elif defined(AGG_COMPARE_STRESS_GROWTH)
		"growth",
#else
		"query",
#endif
		MDB_HASH_SIZE, (unsigned long long)g_signature);
	return 0;
}
