/* large_keys.c - structural long-key stress on the LMDB API.
 *
 * Near-maximum keys give B+trees with very low fanout, so a few thousand
 * records already produce trees of depth 4-5 and every insert/delete wave
 * exercises split, rekey, merge, move and root collapse at every level.
 *
 * This is the stable form of the "large-key" mode of tests/compare: the same
 * key generator and phase sequence, but
 *   - it uses ONLY the LMDB API by default and builds against every stage
 *     (baseline LMDB, 01-base, 02-bottomup, 03-aggformat, AELMDB);
 *   - correctness is checked against an in-memory model of the content
 *     (full ordered scan + per-operation postconditions), not through
 *     aggregate queries;
 *   - aggregate checks are optional and compile-time (LK_AGG): a top-down
 *     integrity check of the whole tree and, optionally, the root totals.
 *     When the integrity oracle validates every branch aggregate against its
 *     subtree, query results are implied by it and are not re-tested here.
 *
 * Build:
 *   cc -I<stage> large_keys.c <stage>/mdb.c <stage>/midl.c -lpthread
 * Optional compile-time switches:
 *   -DLK_AGG=1      create the DBs with ENTRIES|KEYS|HASHSUM (hash offset 0)
 *                   and enable the runtime checks below (aggregate stages only)
 *   -DMDB_DEBUG_AGG_INTEGRITY=1   compile the integrity oracle into the
 *                   aggregate-enabled AELMDB library; needed by LK_AGG_CHECK
 *
 * Runtime parameters (environment):
 *   LK_PROFILE     near-max (default) | half   key length: maxkey-16, or
 *                  variable in [ceil(maxkey/2), maxkey-16]
 *   LK_COUNT       ids per parity (default 768: 1536 keys when full)
 *   LK_ROUNDS      delete/regrow rounds (default 4)
 *   LK_MIN_DEPTH   required peak depth of both DBs (default 4; 0 disables)
 *   LK_SEED        generator seed (default 0x6c617267656b6579)
 *   LK_DUPS        duplicates per key in large_dup (default 1)
 *   LK_DUP_VLEN    duplicate value length (default = value length); values
 *                  near maxkey push dupsets into sub-DBs quickly
 *   LK_CURSOR_DEL  1: delete through cursors (MDB_SET / MDB_GET_BOTH +
 *                  mdb_cursor_del) instead of mdb_del
 *   LK_VERIFY      0: phase-end scan only; N>0: also full scan every N ops
 *   LK_DIR         database directory (default ./testdb_lk)
 *   LK_SNAPSHOT_DIR  copy data.mdb there after every committed phase
 *                  (NN-phase.mdb), for tools/lmdb_fingerprint.py
 *   LK_AGG_CHECK   (LK_AGG builds) 0 none, 1 after every phase (default),
 *                  2 after every mutation - top-down integrity oracle
 *   LK_AGG_TOTALS  (LK_AGG builds) 1: compare mdb_agg_totals with the model
 *
 * Output: LK-CONFIG / STRUCT / STRUCT-SUMMARY lines on stderr, a final
 * "PASS ... content=<hash>" line on stdout.  The content hash depends only on
 * the logical content sequence, so it must be equal on every stage.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <errno.h>
#include <sys/stat.h>
#include "lmdb.h"

#ifndef LK_AGG
#define LK_AGG 0
#endif

#if LK_AGG
# ifndef MDB_AGG_HASHSUM
#  error "LK_AGG requires an aggregate stage header (03-aggformat+, AELMDB)"
# endif
/* top-down integrity oracle: both implementations compile it only with
 * -DMDB_DEBUG_AGG_INTEGRITY=1 */
# if defined(MDB_DEBUG_AGG_INTEGRITY) && MDB_DEBUG_AGG_INTEGRITY
#  define LK_INTEGRITY(txn, dbi) mdb_agg_check_integrity((txn), (dbi))
# endif
/* mdb_agg_totals exists with the query API (AELMDB, AELMDB); 03 only
 * stores the aggregate format */
# ifdef MDB_AGG_PREFIX_INCL
#  define LK_HAVE_TOTALS 1
# endif
# define LK_SCHEMA (MDB_AGG_ENTRIES | MDB_AGG_KEYS | MDB_AGG_HASHSUM)
#else
# define LK_SCHEMA 0
#endif

#define DIE(msg) do { fprintf(stderr, "FAIL [%s] %s:%d: %s\n", g_phase, \
	__FILE__, __LINE__, msg); exit(1); } while (0)
#define C(x) do { int r_ = (x); if (r_) { fprintf(stderr, "FAIL [%s] %s:%d: %s: %s\n", \
	g_phase, __FILE__, __LINE__, #x, mdb_strerror(r_)); exit(1); } } while (0)
#define CHECK(c, msg) do { if (!(c)) DIE(msg); } while (0)

static char g_phase[96] = "init";

/* ------------------------------------------------------------------ params */

static unsigned p_count = 768, p_rounds = 4, p_min_depth = 4, p_dups = 1;
static unsigned p_cursor_del, p_verify, p_variable;
static unsigned p_agg_check = 1, p_agg_totals;
static uint64_t p_seed = UINT64_C(0x6c617267656b6579);
static size_t p_vlen, p_dup_vlen, maxkey, key_min, key_max;
static const char *p_dir = "testdb_lk", *p_snapdir;

static unsigned long
envnum(const char *name, unsigned long dflt)
{
	const char *e = getenv(name);
	return e && *e ? strtoul(e, NULL, 0) : dflt;
}

/* --------------------------------------------------------------- generator */

static uint64_t
mix64(uint64_t x)
{
	x ^= x >> 30; x *= UINT64_C(0xbf58476d1ce4e5b9);
	x ^= x >> 27; x *= UINT64_C(0x94d049bb133111eb);
	x ^= x >> 31;
	return x;
}

static size_t
key_size(uint64_t id)
{
	if (!p_variable)
		return key_max;
	return key_min + (size_t)(mix64(id ^ p_seed) % (key_max - key_min + 1));
}

/* 8-byte big-endian id, then pseudo-random padding: key order == id order */
static void
make_key(unsigned char *buf, size_t len, uint64_t id)
{
	uint64_t x = mix64(id ^ p_seed ^ (uint64_t)len);
	size_t i;
	for (i = 0; i < 8; i++)
		buf[i] = (unsigned char)(id >> (56 - 8 * i));
	for (i = 8; i < len; i++) {
		if ((i & 7) == 0)
			x = mix64(x ^ (uint64_t)i);
		buf[i] = (unsigned char)(x >> ((i & 7) * 8));
	}
}

/* tag 1 = plain value, tag 2+j = duplicate j */
static void
make_value(unsigned char *buf, size_t len, uint64_t id, unsigned tag)
{
	uint64_t x = mix64(id ^ ((uint64_t)tag << 48) ^ p_seed);
	size_t i;
	for (i = 0; i < len; i++) {
		if ((i & 7) == 0)
			x = mix64(x ^ (uint64_t)i);
		buf[i] = (unsigned char)(x >> ((i & 7) * 8));
	}
}

static int
delete_selected(uint64_t id, unsigned round)
{
	/* alternating contiguous blocks empty whole runs of leaves; the block
	 * width changes per round so merge boundaries move */
	uint64_t block = 24 + (uint64_t)(round % 4) * 11;
	return (((id / block) + round) & 1) == 0;
}

/* ------------------------------------------------------------------- model */

static uint64_t nids;		/* ids are 0 .. 2*count-1 */
static unsigned char *m_plain;	/* 1 if the plain key is present */
static uint32_t *m_dup;		/* bitmask of present duplicates */
static uint64_t m_ops;
static uint64_t content_hash = UINT64_C(0xcbf29ce484222325);

static void
hash_u64(uint64_t v)
{
	content_hash = mix64(content_hash ^ v) + UINT64_C(0x9e3779b97f4a7c15);
}

static int
popcount32(uint32_t v)
{
	int n = 0;
	for (; v; v &= v - 1)
		n++;
	return n;
}

/* ------------------------------------------------------------------ helpers */

static MDB_env *env;
static unsigned char *kbuf, *vbuf, *vbuf2;

typedef struct {
	unsigned peak_depth, last_depth, depth_changes;
	size_t peak_branch, peak_leaf;
} Track;
static Track t_plain, t_dup;

/* expected duplicates of id, sorted by memcmp (the default dupsort order
 * for equal-length values) */
static int
dup_order(const void *a, const void *b)
{
	unsigned ja = *(const unsigned *)a, jb = *(const unsigned *)b;
	unsigned char *va = vbuf + (size_t)ja * p_dup_vlen;
	unsigned char *vb2 = vbuf + (size_t)jb * p_dup_vlen;
	return memcmp(va, vb2, p_dup_vlen);
}

static void
verify_content(MDB_txn *txn, MDB_dbi plain, MDB_dbi dup, const char *when)
{
	MDB_cursor *c;
	MDB_val k, v;
	MDB_stat st;
	uint64_t id, n_plain = 0, n_dup = 0;
	int rc;
	unsigned order[32];

	/* plain: an ordered scan must return exactly the present ids */
	C(mdb_cursor_open(txn, plain, &c));
	rc = mdb_cursor_get(c, &k, &v, MDB_FIRST);
	for (id = 0; id < nids; id++) {
		size_t kl;
		if (!m_plain[id])
			continue;
		CHECK(rc == MDB_SUCCESS, "plain scan ended early");
		kl = key_size(id);
		make_key(kbuf, kl, id);
		make_value(vbuf2, p_vlen, id, 1);
		CHECK(k.mv_size == kl && !memcmp(k.mv_data, kbuf, kl), "plain scan key");
		CHECK(v.mv_size == p_vlen && !memcmp(v.mv_data, vbuf2, p_vlen), "plain scan value");
		n_plain++;
		rc = mdb_cursor_get(c, &k, &v, MDB_NEXT);
	}
	CHECK(rc == MDB_NOTFOUND, "plain scan has extra records");
	mdb_cursor_close(c);
	C(mdb_stat(txn, plain, &st));
	CHECK(st.ms_entries == n_plain, "plain ms_entries");

	/* dup: every key with its duplicate set, in dupsort order */
	C(mdb_cursor_open(txn, dup, &c));
	rc = mdb_cursor_get(c, &k, &v, MDB_FIRST);
	for (id = 0; id < nids; id++) {
		size_t kl;
		unsigned j, nd = 0;
		mdb_size_t cnt;
		if (!m_dup[id])
			continue;
		CHECK(rc == MDB_SUCCESS, "dup scan ended early");
		kl = key_size(id);
		make_key(kbuf, kl, id);
		CHECK(k.mv_size == kl && !memcmp(k.mv_data, kbuf, kl), "dup scan key");
		C(mdb_cursor_count(c, &cnt));
		CHECK(cnt == (mdb_size_t)popcount32(m_dup[id]), "dup count");
		for (j = 0; j < p_dups; j++) {
			make_value(vbuf + (size_t)j * p_dup_vlen, p_dup_vlen, id, 2 + j);
			if (m_dup[id] & (1u << j))
				order[nd++] = j;
		}
		qsort(order, nd, sizeof(order[0]), dup_order);
		for (j = 0; j < nd; j++) {
			CHECK(rc == MDB_SUCCESS, "dup scan ended inside a dupset");
			CHECK(k.mv_size == kl && !memcmp(k.mv_data, kbuf, kl), "dup scan key in dupset");
			CHECK(v.mv_size == p_dup_vlen &&
				!memcmp(v.mv_data, vbuf + (size_t)order[j] * p_dup_vlen, p_dup_vlen),
				"dup scan value");
			n_dup++;
			rc = mdb_cursor_get(c, &k, &v, MDB_NEXT);
		}
	}
	CHECK(rc == MDB_NOTFOUND, "dup scan has extra records");
	mdb_cursor_close(c);
	C(mdb_stat(txn, dup, &st));
	CHECK(st.ms_entries == n_dup, "dup ms_entries");
	(void)when;
}

#if LK_AGG
/* little-endian modular sum of MDB_HASH_SIZE bytes (hash offset 0) */
static void
hs_add(uint8_t *acc, const uint8_t *x)
{
	unsigned i, carry = 0;
	for (i = 0; i < MDB_HASH_SIZE; i++) {
		unsigned s = (unsigned)acc[i] + x[i] + carry;
		acc[i] = (uint8_t)s;
		carry = s >> 8;
	}
}

static void
agg_check_db(MDB_txn *txn, MDB_dbi dbi, int is_dup)
{
#ifdef LK_INTEGRITY
	if (p_agg_check)
		C(LK_INTEGRITY(txn, dbi));
#endif
#ifdef LK_HAVE_TOTALS
	if (p_agg_totals) {
		MDB_agg got;
		uint8_t hs[MDB_HASH_SIZE];
		uint64_t id, entries = 0, keys = 0;
		unsigned j;
		memset(hs, 0, sizeof(hs));
		for (id = 0; id < nids; id++) {
			if (!is_dup) {
				if (!m_plain[id])
					continue;
				make_value(vbuf2, p_vlen, id, 1);
				hs_add(hs, vbuf2);
				entries++; keys++;
				continue;
			}
			if (!m_dup[id])
				continue;
			keys++;
			for (j = 0; j < p_dups; j++)
				if (m_dup[id] & (1u << j)) {
					make_value(vbuf2, p_dup_vlen, id, 2 + j);
					hs_add(hs, vbuf2);
					entries++;
				}
		}
		C(mdb_agg_totals(txn, dbi, &got));
		CHECK(got.mv_agg_entries == entries, "agg totals entries");
		CHECK(got.mv_agg_keys == keys, "agg totals keys");
		CHECK(!memcmp(got.mv_agg_hashes, hs, MDB_HASH_SIZE), "agg totals hashsum");
	}
#else
	(void)txn; (void)dbi; (void)is_dup; (void)hs_add;
#endif
}
#endif

static void
agg_check(MDB_txn *txn, MDB_dbi plain, MDB_dbi dup)
{
#if LK_AGG
	agg_check_db(txn, plain, 0);
	agg_check_db(txn, dup, 1);
#else
	(void)txn; (void)plain; (void)dup;
#endif
}

/* called after every single mutation */
static void
after_op(MDB_txn *txn, MDB_dbi plain, MDB_dbi dup)
{
	m_ops++;
	if (p_verify && m_ops % p_verify == 0)
		verify_content(txn, plain, dup, "op");
#if LK_AGG
	if (p_agg_check >= 2) {
		unsigned save = p_agg_totals;
		p_agg_totals = 0;
		agg_check(txn, plain, dup);
		p_agg_totals = save;
	}
#endif
}

static void
note(MDB_txn *txn, MDB_dbi dbi, const char *db, Track *t)
{
	MDB_stat st;
	C(mdb_stat(txn, dbi, &st));
	if (t->last_depth && t->last_depth != st.ms_depth)
		t->depth_changes++;
	t->last_depth = st.ms_depth;
	if (st.ms_depth > t->peak_depth) t->peak_depth = st.ms_depth;
	if (st.ms_branch_pages > t->peak_branch) t->peak_branch = st.ms_branch_pages;
	if (st.ms_leaf_pages > t->peak_leaf) t->peak_leaf = st.ms_leaf_pages;
	fprintf(stderr, "STRUCT phase=%s db=%s depth=%u branch=%zu leaf=%zu "
		"overflow=%zu entries=%zu\n", g_phase, db, st.ms_depth,
		(size_t)st.ms_branch_pages, (size_t)st.ms_leaf_pages,
		(size_t)st.ms_overflow_pages, (size_t)st.ms_entries);
}

static void
snapshot(void)
{
	static unsigned seq;
	char src[1024], dst[1024], buf[65536];
	FILE *in, *out;
	size_t n;
	if (!p_snapdir)
		return;
	snprintf(src, sizeof(src), "%s/data.mdb", p_dir);
	snprintf(dst, sizeof(dst), "%s/%02u-%s.mdb", p_snapdir, seq++, g_phase);
	in = fopen(src, "rb");
	out = fopen(dst, "wb");
	CHECK(in && out, "snapshot open");
	while ((n = fread(buf, 1, sizeof(buf), in)) > 0)
		CHECK(fwrite(buf, 1, n, out) == n, "snapshot write");
	fclose(in);
	fclose(out);
}

/* read-only verification of a committed phase */
static void
phase_end(void)
{
	MDB_txn *txn;
	MDB_dbi plain, dup;
	uint64_t id;
	unsigned j;
	C(mdb_txn_begin(env, NULL, MDB_RDONLY, &txn));
	C(mdb_dbi_open(txn, "large_plain", 0, &plain));
	C(mdb_dbi_open(txn, "large_dup", 0, &dup));
	verify_content(txn, plain, dup, "phase");
	agg_check(txn, plain, dup);
	note(txn, plain, "plain", &t_plain);
	note(txn, dup, "dup", &t_dup);
	mdb_txn_abort(txn);
	/* content hash: phase boundary + the whole model */
	hash_u64(UINT64_C(0x5048415345));
	for (id = 0; id < nids; id++) {
		hash_u64(m_plain[id]);
		for (j = 0; j < p_dups; j++)
			hash_u64((m_dup[id] >> j) & 1);
	}
	snapshot();
}

/* ---------------------------------------------------------------- mutations */

static void
put_plain(MDB_txn *txn, MDB_dbi plain, MDB_dbi dup, uint64_t id)
{
	size_t kl = key_size(id);
	MDB_val k = { kl, kbuf }, v = { p_vlen, vbuf2 }, g;
	make_key(kbuf, kl, id);
	make_value(vbuf2, p_vlen, id, 1);
	C(mdb_put(txn, plain, &k, &v, MDB_NOOVERWRITE));
	m_plain[id] = 1;
	CHECK(mdb_get(txn, plain, &k, &g) == 0 && g.mv_size == p_vlen &&
		!memcmp(g.mv_data, vbuf2, p_vlen), "plain put postcondition");
	after_op(txn, plain, dup);
}

static void
put_dup(MDB_txn *txn, MDB_dbi plain, MDB_dbi dup, uint64_t id, unsigned j)
{
	size_t kl = key_size(id);
	MDB_val k = { kl, kbuf }, v = { p_dup_vlen, vbuf2 };
	MDB_cursor *c;
	make_key(kbuf, kl, id);
	make_value(vbuf2, p_dup_vlen, id, 2 + j);
	C(mdb_put(txn, dup, &k, &v, MDB_NODUPDATA));
	m_dup[id] |= 1u << j;
	C(mdb_cursor_open(txn, dup, &c));
	CHECK(mdb_cursor_get(c, &k, &v, MDB_GET_BOTH) == 0, "dup put postcondition");
	mdb_cursor_close(c);
	after_op(txn, plain, dup);
}

static void
del_plain(MDB_txn *txn, MDB_dbi plain, MDB_dbi dup, uint64_t id)
{
	size_t kl = key_size(id);
	MDB_val k = { kl, kbuf }, g;
	make_key(kbuf, kl, id);
	if (p_cursor_del) {
		MDB_cursor *c;
		C(mdb_cursor_open(txn, plain, &c));
		C(mdb_cursor_get(c, &k, &g, MDB_SET));
		C(mdb_cursor_del(c, 0));
		mdb_cursor_close(c);
		k.mv_size = kl; k.mv_data = kbuf;
	} else {
		C(mdb_del(txn, plain, &k, NULL));
	}
	m_plain[id] = 0;
	CHECK(mdb_get(txn, plain, &k, &g) == MDB_NOTFOUND, "plain delete postcondition");
	after_op(txn, plain, dup);
}

static void
del_dup(MDB_txn *txn, MDB_dbi plain, MDB_dbi dup, uint64_t id, unsigned j)
{
	size_t kl = key_size(id);
	MDB_val k = { kl, kbuf }, v = { p_dup_vlen, vbuf2 };
	MDB_cursor *c;
	int rc;
	make_key(kbuf, kl, id);
	make_value(vbuf2, p_dup_vlen, id, 2 + j);
	if (p_cursor_del) {
		C(mdb_cursor_open(txn, dup, &c));
		C(mdb_cursor_get(c, &k, &v, MDB_GET_BOTH));
		C(mdb_cursor_del(c, 0));
		mdb_cursor_close(c);
		k.mv_size = kl; k.mv_data = kbuf;
		v.mv_size = p_dup_vlen; v.mv_data = vbuf2;
	} else {
		C(mdb_del(txn, dup, &k, &v));
	}
	m_dup[id] &= ~(1u << j);
	C(mdb_cursor_open(txn, dup, &c));
	rc = mdb_cursor_get(c, &k, &v, MDB_GET_BOTH);
	CHECK(rc == MDB_NOTFOUND, "dup delete postcondition");
	if (!m_dup[id]) {
		MDB_val g;
		k.mv_size = kl; k.mv_data = kbuf;
		CHECK(mdb_get(txn, dup, &k, &g) == MDB_NOTFOUND, "dup last delete removes key");
	}
	mdb_cursor_close(c);
	after_op(txn, plain, dup);
}

/* ------------------------------------------------------------------ phases */

static void
begin(MDB_txn **txn, MDB_dbi *plain, MDB_dbi *dup, unsigned flags)
{
	C(mdb_txn_begin(env, NULL, 0, txn));
	C(mdb_dbi_open(*txn, "large_plain", flags | LK_SCHEMA, plain));
	C(mdb_dbi_open(*txn, "large_dup", flags | MDB_DUPSORT | LK_SCHEMA, dup));
}

static void
open_env(void)
{
	C(mdb_env_create(&env));
	C(mdb_env_set_mapsize(env, (size_t)1 << 30));
	C(mdb_env_set_maxdbs(env, 8));
	C(mdb_env_open(env, p_dir, MDB_NOSYNC, 0664));
}

int
main(void)
{
	MDB_txn *txn;
	MDB_dbi plain, dup;
	unsigned round, j;
	uint64_t id, n;
	const char *prof = getenv("LK_PROFILE");
	char cmd[1200];

	p_count = (unsigned)envnum("LK_COUNT", p_count);
	p_rounds = (unsigned)envnum("LK_ROUNDS", p_rounds);
	p_min_depth = (unsigned)envnum("LK_MIN_DEPTH", p_min_depth);
	p_seed = (uint64_t)strtoull(getenv("LK_SEED") ? getenv("LK_SEED") : "0x6c617267656b6579", NULL, 0);
	p_dups = (unsigned)envnum("LK_DUPS", 1);
	p_cursor_del = (unsigned)envnum("LK_CURSOR_DEL", 0);
	p_verify = (unsigned)envnum("LK_VERIFY", 0);
	p_agg_check = (unsigned)envnum("LK_AGG_CHECK", 1);
	p_agg_totals = (unsigned)envnum("LK_AGG_TOTALS", 0);
	if (getenv("LK_DIR")) p_dir = getenv("LK_DIR");
	p_snapdir = getenv("LK_SNAPSHOT_DIR");
	p_variable = prof && !strcmp(prof, "half");
	CHECK(!prof || p_variable || !strcmp(prof, "near-max"), "LK_PROFILE: near-max | half");
	CHECK(p_dups >= 1 && p_dups <= 32, "LK_DUPS must be 1..32");
#if LK_AGG
	p_vlen = MDB_HASH_SIZE + 16;	/* hash slice at offset 0 + tail */
#else
	p_vlen = 48;			/* same geometry as the H=32 aggregate runs */
	p_agg_check = p_agg_totals = 0;
#endif
#if LK_AGG && !defined(LK_INTEGRITY)
	if (p_agg_check)
		fprintf(stderr, "LK: no integrity oracle in this build (needs "
			"-DMDB_DEBUG_AGG_INTEGRITY=1); LK_AGG_CHECK ignored\n");
	p_agg_check = 0;
#endif
#if LK_AGG && !defined(LK_HAVE_TOTALS)
	if (p_agg_totals)
		fprintf(stderr, "LK: this stage has no mdb_agg_totals; LK_AGG_TOTALS ignored\n");
	p_agg_totals = 0;
#endif
	p_dup_vlen = (size_t)envnum("LK_DUP_VLEN", p_vlen);

	snprintf(cmd, sizeof(cmd), "rm -rf '%s' && mkdir -p '%s'", p_dir, p_dir);
	CHECK(system(cmd) == 0, "cannot prepare LK_DIR");
	if (p_snapdir) {
		snprintf(cmd, sizeof(cmd), "rm -rf '%s' && mkdir -p '%s'", p_snapdir, p_snapdir);
		CHECK(system(cmd) == 0, "cannot prepare LK_SNAPSHOT_DIR");
	}
	open_env();
	maxkey = (size_t)mdb_env_get_maxkeysize(env);
	CHECK(maxkey >= 64, "maxkey too small");
	key_max = maxkey - 16;
	key_min = p_variable ? (maxkey + 1) / 2 : key_max;
	CHECK(p_dup_vlen >= 8 && p_dup_vlen <= maxkey, "LK_DUP_VLEN must be in [8, maxkey]");
#if LK_AGG
	CHECK(p_dup_vlen >= MDB_HASH_SIZE, "LK_DUP_VLEN must hold a hash slice");
#endif

	nids = (uint64_t)p_count * 2;
	m_plain = calloc(nids, 1);
	m_dup = calloc(nids, sizeof(*m_dup));
	kbuf = malloc(maxkey);
	vbuf = malloc((size_t)p_dups * p_dup_vlen + p_vlen);
	vbuf2 = malloc(p_dup_vlen > p_vlen ? p_dup_vlen : p_vlen);
	CHECK(m_plain && m_dup && kbuf && vbuf && vbuf2, "out of memory");

	fprintf(stderr, "LK-CONFIG profile=%s count=%u rounds=%u maxkey=%zu key_min=%zu "
		"key_max=%zu vlen=%zu dups=%u dup_vlen=%zu cursor_del=%u agg=%d "
		"agg_check=%u agg_totals=%u seed=0x%llx\n",
		p_variable ? "half" : "near-max", p_count, p_rounds, maxkey, key_min,
		key_max, p_vlen, p_dups, p_dup_vlen, p_cursor_del, LK_AGG,
		p_agg_check, p_agg_totals, (unsigned long long)p_seed);
	hash_u64(p_seed); hash_u64(p_count); hash_u64(p_rounds);
	hash_u64(key_min); hash_u64(key_max); hash_u64(p_dups); hash_u64(p_dup_vlen);

	/* 1: sparse ordered tree, even ids only (one insertion gap per pair) */
	strcpy(g_phase, "sparse-grow");
	begin(&txn, &plain, &dup, MDB_CREATE);
#if LK_AGG
	C(mdb_set_hash_offset(txn, plain, 0));
	C(mdb_set_hash_offset(txn, dup, 0));
#endif
	for (id = 0; id < nids; id += 2) {
		put_plain(txn, plain, dup, id);
		for (j = 0; j < p_dups; j++)
			put_dup(txn, plain, dup, id, j);
	}
	C(mdb_txn_commit(txn));
	phase_end();

	/* 2: fill every gap: interior inserts, not right-edge appends */
	strcpy(g_phase, "interior-fill");
	begin(&txn, &plain, &dup, 0);
	for (id = 1; id < nids; id += 2) {
		put_plain(txn, plain, dup, id);
		for (j = 0; j < p_dups; j++)
			put_dup(txn, plain, dup, id, j);
	}
	C(mdb_txn_commit(txn));
	phase_end();

	/* 3: plain churn: empty alternating blocks, regrow them in reverse */
	for (round = 0; round < p_rounds; round++) {
		snprintf(g_phase, sizeof(g_phase), "plain-delete-%u", round);
		begin(&txn, &plain, &dup, 0);
		for (id = 0; id < nids; id++)
			if (delete_selected(id, round))
				del_plain(txn, plain, dup, id);
		C(mdb_txn_commit(txn));
		phase_end();

		snprintf(g_phase, sizeof(g_phase), "plain-regrow-%u", round);
		begin(&txn, &plain, &dup, 0);
		for (n = nids; n; n--)
			if (delete_selected(n - 1, round))
				put_plain(txn, plain, dup, n - 1);
		C(mdb_txn_commit(txn));
		phase_end();
	}

	/* 4: the same churn through DUPSORT exact-pair deletes; the last
	 * duplicate of a key removes the outer key */
	for (round = 0; round < p_rounds; round++) {
		snprintf(g_phase, sizeof(g_phase), "dup-delete-%u", round);
		begin(&txn, &plain, &dup, 0);
		for (id = 0; id < nids; id++)
			if (delete_selected(id, round))
				for (j = 0; j < p_dups; j++)
					del_dup(txn, plain, dup, id, j);
		C(mdb_txn_commit(txn));
		phase_end();

		snprintf(g_phase, sizeof(g_phase), "dup-regrow-%u", round);
		begin(&txn, &plain, &dup, 0);
		for (n = nids; n; n--)
			if (delete_selected(n - 1, round))
				for (j = p_dups; j; j--)
					put_dup(txn, plain, dup, n - 1, j - 1);
		C(mdb_txn_commit(txn));
		phase_end();
	}

	/* 5: persistence across an environment reopen */
	mdb_env_close(env);
	strcpy(g_phase, "reopen-final");
	open_env();
	CHECK((size_t)mdb_env_get_maxkeysize(env) == maxkey, "maxkey changed on reopen");
	phase_end();
	mdb_env_close(env);

	fprintf(stderr, "STRUCT-SUMMARY plain_peak_depth=%u plain_peak_branch=%zu "
		"plain_peak_leaf=%zu plain_depth_changes=%u dup_peak_depth=%u "
		"dup_peak_branch=%zu dup_peak_leaf=%zu dup_depth_changes=%u ops=%llu\n",
		t_plain.peak_depth, t_plain.peak_branch, t_plain.peak_leaf,
		t_plain.depth_changes, t_dup.peak_depth, t_dup.peak_branch,
		t_dup.peak_leaf, t_dup.depth_changes, (unsigned long long)m_ops);
	if (p_min_depth) {
		strcpy(g_phase, "summary");
		CHECK(t_plain.peak_depth >= p_min_depth, "plain DB did not reach LK_MIN_DEPTH");
		CHECK(t_dup.peak_depth >= p_min_depth, "dup DB did not reach LK_MIN_DEPTH");
	}
	printf("PASS large_keys profile=%s count=%u rounds=%u dups=%u agg=%d content=%016llx\n",
		p_variable ? "half" : "near-max", p_count, p_rounds, p_dups, LK_AGG,
		(unsigned long long)content_hash);
	return 0;
}
