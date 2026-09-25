/* F2: opaque transport of aggregate branch prefixes (03-aggformat).
 *
 * In 03 the aggregate bytes of a branch link are opaque: a new link gets
 * zero bytes, an existing link keeps its bytes when a split, a move, a merge
 * or a separator replacement relocates it.  The test stamps every branch
 * link of three aggregate DBs (plain, DUPSORT with sub-DBs, DUPFIXED) with a
 * unique self-checking token, directly in the data file, then runs
 * operations and walks the file again:
 *
 *  - same-size overwrites (copy on write, no structural change): the same
 *    multiset of tokens;
 *  - growth (inserts only: leaf and branch splits, separator replacements,
 *    sub-DB growth): no link is destroyed, so every token survives;
 *  - churn with heavy deletes (merges, moves, root collapses): every prefix
 *    is either zero (a new link) or a valid token, no token appears twice.
 *  In every phase the EVEN padding byte is zero.  After each phase the test
 *  prints a deterministic digest of the tokens in tree order, pinning opaque
 *  transport behavior where no invariant can specify which link a moved token
 *  lands on.
 *
 * White-box: includes mdb.c (MDB_HASH_SIZE may be any value in [1, 256]).
 */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE 1
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "mdb.c"

#define DIR "testdb_f2"
#define SCHEMA (MDB_AGG_ENTRIES|MDB_AGG_KEYS|MDB_AGG_HASHSUM)
#define CK(x) do { int r_ = (x); if (r_) { fprintf(stderr, "%s:%d: %s: %s\n", \
	__FILE__, __LINE__, #x, mdb_strerror(r_)); exit(2); } } while (0)

#include "f_file.h"

/* ------------------------------------------------------------- tokens */

/* token id >= 1 in the first 4 raw bytes, the rest derived from it */
static unsigned char
tok_byte(uint32_t id, size_t i)
{
	return (unsigned char)(id * 131u + (unsigned)i * 7u + 1u);
}

static void
tok_write(unsigned char *p, size_t raw, uint32_t id)
{
	size_t i;
	memcpy(p, &id, 4);
	for (i = 4; i < raw; i++)
		p[i] = tok_byte(id, i);
}

/* 0: all zero; id: a valid token; -1: anything else */
static long
tok_read(const unsigned char *p, size_t raw)
{
	uint32_t id;
	size_t i;
	for (i = 0; i < raw && !p[i]; i++)
		;
	if (i == raw)
		return 0;
	memcpy(&id, p, 4);
	if (!id)
		return -1;
	for (i = 4; i < raw; i++)
		if (p[i] != tok_byte(id, i))
			return -1;
	return id;
}

/* ------------------------------------------------------------- walker */

typedef struct {
	int stamp;		/* 1: write tokens, 0: check */
	uint32_t next;		/* next token id */
	size_t links, zero, tokens, bad, badpad, dup;
	unsigned char *seen;	/* token id -> seen */
	size_t nseen;
	uint32_t *list;		/* tokens found, in walk order */
	size_t nlist;
	uint64_t digest;	/* tokens and zero links in tree order */
} Walk;

static void walk_tree(Walk *w, pgno_t root, int depth);

static void
walk_page(Walk *w, pgno_t n, int level)
{
	MDB_page *mp = pg(n);
	unsigned i, nk = NUMKEYS(mp);

	if (level > CURSOR_STACK) { fail("tree too deep (loop?)"); return; }
	if (IS_BRANCH(mp)) {
		uint16_t agg = PAGE_AGGFLAGS(mp);
		size_t raw = mdb_agg_value_size_flags(agg), pfx = mdb_agg_prefix_len_flags(agg);
		for (i = 0; i < nk; i++) {
			MDB_node *node = NODEPTR(mp, i);
			unsigned char *p = (unsigned char *)node->mn_data;
			if (raw) {
				w->links++;
				if (w->stamp) {
					tok_write(p, raw, w->next++);
				} else {
					long t = tok_read(p, raw);
					w->digest = (w->digest ^ (uint64_t)(t < 0 ? 0xffffffffu : (uint64_t)t)
						^ ((uint64_t)level << 40)) * 0x100000001b3ull;
					if (pfx > raw && p[raw])
						w->badpad++;
					if (t == 0) {
						w->zero++;
					} else if (t < 0) {
						w->bad++;
					} else {
						w->tokens++;
						if ((size_t)t < w->nseen) {
							if (w->seen[t]++)
								w->dup++;
						}
						w->list = realloc(w->list, (w->nlist + 1) * sizeof(*w->list));
						w->list[w->nlist++] = (uint32_t)t;
					}
				}
			}
			walk_page(w, NODEPGNO(node), level + 1);
		}
		return;
	}
	if (IS_LEAF2(mp) || !IS_LEAF(mp))
		return;
	for (i = 0; i < nk; i++) {
		MDB_node *node = NODEPTR(mp, i);
		if (F_ISSET(node->mn_flags, F_SUBDATA) && !(node->mn_flags & F_BIGDATA)) {
			MDB_db db;
			memcpy(&db, NODEDATA(node), sizeof(db));
			walk_tree(w, db.md_root, db.md_depth);
		}
	}
}

static void
walk_tree(Walk *w, pgno_t root, int depth)
{
	if (depth && root != P_INVALID)
		walk_page(w, root, 0);
}

static void
walk_file(Walk *w)
{
	MDB_meta *m;
	load();
	m = meta();
	walk_tree(w, m->mm_dbs[MAIN_DBI].md_root, m->mm_dbs[MAIN_DBI].md_depth);
}

/* ------------------------------------------------------------- workload */

static MDB_env *env;
static MDB_dbi dp, dd, df, db, dr;
#define NBIG 3000
#define NREKEY 4000
static uint64_t rs = 88172645463325252ULL;

static unsigned
rnd(unsigned n)
{
	rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17;
	return (unsigned)(rs % n);
}

static void
mkkey(unsigned char *b, MDB_val *k, unsigned id)
{
	size_t len = 8 + (id * 2654435761u >> 7) % 180, i;
	b[0] = (unsigned char)(id >> 24); b[1] = (unsigned char)(id >> 16);
	b[2] = (unsigned char)(id >> 8); b[3] = (unsigned char)id;
	for (i = 4; i < len; i++)
		b[i] = (unsigned char)(id * 31 + i);
	k->mv_size = len;
	k->mv_data = b;
}

/* long keys (300-511 bytes): few links per branch page, so that branch
 * pages merge, move nodes and replace separators by local splits */
static void
mkbig(unsigned char *b, MDB_val *k, unsigned id)
{
	size_t len = 300 + (id * 2654435761u >> 9) % 212, i;
	if (len > 511) len = 511;
	b[0] = (unsigned char)(id >> 8); b[1] = (unsigned char)id;
	for (i = 2; i < len; i++)
		b[i] = (unsigned char)(id * 13 + i);
	k->mv_size = len;
	k->mv_data = b;
}

/* r: 2-byte id, then a fixed length */
static void
mkr(unsigned char *b, MDB_val *k, unsigned id, size_t len)
{
	size_t i;
	b[0] = (unsigned char)(id >> 8); b[1] = (unsigned char)id;
	for (i = 2; i < len; i++)
		b[i] = (unsigned char)(id + i);
	k->mv_size = len;
	k->mv_data = b;
}

static void
mkval(unsigned char *b, MDB_val *v, unsigned ver, size_t len)
{
	size_t i;
	for (i = 0; i < len; i++)
		b[i] = (unsigned char)(ver >> ((i & 3) * 8)) ^ (unsigned char)i;
	b[0] = (unsigned char)(ver >> 24); b[1] = (unsigned char)(ver >> 16);
	b[2] = (unsigned char)(ver >> 8); b[3] = (unsigned char)ver;
	v->mv_size = len;
	v->mv_data = b;
}

static void
env_open(void)
{
	CK(mdb_env_create(&env));
	CK(mdb_env_set_maxdbs(env, 8));
	CK(mdb_env_set_mapsize(env, (size_t)1 << 30));
	CK(mdb_env_open(env, DIR, MDB_NOSYNC, 0664));
}

static MDB_txn *
begin(void)
{
	MDB_txn *txn;
	CK(mdb_txn_begin(env, NULL, 0, &txn));
	CK(mdb_dbi_open(txn, "p", MDB_CREATE | SCHEMA, &dp));
	CK(mdb_dbi_open(txn, "d", MDB_CREATE | MDB_DUPSORT | SCHEMA, &dd));
	CK(mdb_dbi_open(txn, "f", MDB_CREATE | MDB_DUPSORT | MDB_DUPFIXED | SCHEMA, &df));
	CK(mdb_dbi_open(txn, "b", MDB_CREATE | SCHEMA, &db));
	CK(mdb_dbi_open(txn, "r", MDB_CREATE | SCHEMA, &dr));
	return txn;
}

/* keys 0..nk-1 in p; keys 0..7 in d and f with many duplicates; even
 * long keys in b */
static void
build(unsigned nk)
{
	MDB_txn *txn = begin();
	unsigned char kb[512], vb[512];
	MDB_val k, v;
	unsigned i;
	for (i = 0; i < NBIG; i += 2) {
		mkbig(kb, &k, i);
		mkval(vb, &v, i, 16);
		CK(mdb_put(txn, db, &k, &v, 0));
	}
	for (i = 0; i < NREKEY; i += 4) {
		mkr(kb, &k, i, 100);
		mkval(vb, &v, i, 16);
		CK(mdb_put(txn, dr, &k, &v, 0));
	}
	for (i = 0; i < nk; i++) {
		mkkey(kb, &k, i * 7);
		mkval(vb, &v, i, 40);
		CK(mdb_put(txn, dp, &k, &v, 0));
	}
	for (i = 0; i < 3000; i++) {
		mkkey(kb, &k, i % 8);
		mkval(vb, &v, i, 120);
		CK(mdb_put(txn, dd, &k, &v, 0));
		mkval(vb, &v, i, 64);
		CK(mdb_put(txn, df, &k, &v, 0));
	}
	CK(mdb_txn_commit(txn));
}

/* same-size overwrites: no structural change */
static void
overwrite(unsigned nk)
{
	MDB_txn *txn = begin();
	unsigned char kb[256], vb[512];
	MDB_val k, v;
	unsigned i;
	for (i = 0; i < nk; i += 3) {
		mkkey(kb, &k, i * 7);
		mkval(vb, &v, i + 100000, 40);
		CK(mdb_put(txn, dp, &k, &v, 0));
	}
	CK(mdb_txn_commit(txn));
}

/* Replace the separator above the cursor's leaf (the cursor is at index 0
 * of a leaf that is not the leftmost), as the first-key update of a put
 * does: climb to the lowest ancestor with a non-zero index, replace the key
 * there and publish a replacement split.  Written against the functions
 * both 03 variants share (mdb_branch_rekey_local, mdb_split_publish). */
static int
rekey_separator(MDB_cursor *mc, MDB_val *key)
{
	unsigned short dtop = 1;
	int rc = mdb_cursor_touch(mc);
	if (rc)
		return rc;
	mc->mc_top--;
	while (mc->mc_top && !mc->mc_ki[mc->mc_top]) {
		mc->mc_top--;
		dtop++;
	}
	if (mc->mc_ki[mc->mc_top]) {
		MDB_split_result split;
		rc = mdb_branch_rekey_local(mc, key, &split);
		if (!rc && split.msr_right)
			rc = mdb_split_publish(mc, &split, MDB_SPLIT_REPLACE);
		mdb_split_result_destroy(&split);
	}
	mc->mc_top += dtop;
	return rc;
}

/* separator growth in r: the separator of every leaf but the first becomes
 * a 511-byte key that still sorts between the leaves.  In full branch pages
 * the replacement does not fit and the link is re-added through a local
 * split.  No link is destroyed. */
static void
rekey_grow(void)
{
	MDB_txn *txn = begin();
	MDB_cursor *c;
	MDB_val k, v;
	unsigned char kb[512];
	unsigned *ids = NULL, n = 0, i;
	int rc;

	CK(mdb_cursor_open(txn, dr, &c));
	for (rc = mdb_cursor_get(c, &k, &v, MDB_FIRST); rc == 0;
		rc = mdb_cursor_get(c, &k, &v, MDB_NEXT)) {
		if (c->mc_ki[c->mc_top] == 0 && c->mc_top > 0 && c->mc_ki[c->mc_top - 1] > 0) {
			ids = realloc(ids, (n + 1) * sizeof(*ids));
			ids[n++] = (unsigned)((unsigned char *)k.mv_data)[0] << 8 |
				((unsigned char *)k.mv_data)[1];
		}
	}
	for (i = 0; i < n; i++) {
		MDB_val lk;
		mkr(kb, &k, ids[i], 100);
		CK(mdb_cursor_get(c, &k, &v, MDB_SET));
		if (c->mc_ki[c->mc_top] != 0) { fail("rekey: key not first in its leaf"); continue; }
		/* same 2-byte id, then zeros: below the leaf's first key, above
		 * the previous leaf's keys */
		memset(kb, 0, 511);
		kb[0] = (unsigned char)(ids[i] >> 8); kb[1] = (unsigned char)ids[i];
		lk.mv_size = 511;
		lk.mv_data = kb;
		CK(rekey_separator(c, &lk));
	}
	mdb_cursor_close(c);
	free(ids);
	CK(mdb_txn_commit(txn));
	printf("separator growth: %u separators\n", n);
}

/* growth only: inserts between existing keys (splits, separator
 * replacements) and duplicate growth; no link can be destroyed */
static void
grow(unsigned nk)
{
	MDB_txn *txn = begin();
	unsigned char kb[512], vb[512];
	MDB_val k, v;
	unsigned i;
	for (i = 1; i < NBIG; i += 2) {		/* the odd long keys */
		mkbig(kb, &k, i);
		mkval(vb, &v, i, 16);
		CK(mdb_put(txn, db, &k, &v, 0));
	}
	for (i = 0; i < nk; i++) {
		unsigned id = rnd(nk * 7);
		int rc;
		mkkey(kb, &k, id);
		mkval(vb, &v, id, 20 + rnd(200));
		rc = mdb_put(txn, dp, &k, &v, MDB_NOOVERWRITE);
		if (rc && rc != MDB_KEYEXIST) CK(rc);
	}
	for (i = 0; i < 3000; i++) {
		unsigned ver = 3000 + rnd(6000);
		mkkey(kb, &k, rnd(8));
		mkval(vb, &v, ver, 120);
		CK(mdb_put(txn, dd, &k, &v, 0));
		mkval(vb, &v, ver, 64);
		CK(mdb_put(txn, df, &k, &v, 0));
	}
	CK(mdb_txn_commit(txn));
}

/* heavy deletes through cursors (merges, moves of branch nodes, root
 * collapses): each round removes about half of the records of every DB */
static void
shrink(void)
{
	MDB_dbi dbs[4];
	unsigned r, i;
	for (r = 0; r < 3; r++) {
		MDB_txn *txn = begin();
		dbs[0] = dp; dbs[1] = dd; dbs[2] = df; dbs[3] = db;
		for (i = 0; i < 4; i++) {
			MDB_cursor *c;
			MDB_val k, v;
			int rc;
			CK(mdb_cursor_open(txn, dbs[i], &c));
			rc = mdb_cursor_get(c, &k, &v, MDB_FIRST);
			while (rc == 0) {
				if (rnd(2)) {
					CK(mdb_cursor_del(c, 0));
					rc = mdb_cursor_get(c, &k, &v, MDB_GET_CURRENT);
					if (rc == 0)
						continue;	/* on the next record */
					rc = mdb_cursor_get(c, &k, &v, MDB_NEXT);
				} else {
					rc = mdb_cursor_get(c, &k, &v, MDB_NEXT);
				}
			}
			if (rc != MDB_NOTFOUND) CK(rc);
			mdb_cursor_close(c);
		}
		CK(mdb_txn_commit(txn));
	}
}

/* structural churn: inserts between existing keys (splits, separator
 * changes), deletes (merges, moves), duplicate growth and shrink */
static void
churn(unsigned nk, unsigned rounds)
{
	unsigned char kb[512], vb[512];
	MDB_val k, v;
	unsigned r, i;
	for (r = 0; r < rounds; r++) {
		MDB_txn *txn = begin();
		for (i = 0; i < nk; i++) {
			unsigned id = rnd(nk * 7);
			int rc;
			mkkey(kb, &k, id);
			if (rnd(3)) {
				mkval(vb, &v, id, 20 + rnd(200));
				CK(mdb_put(txn, dp, &k, &v, 0));
			} else {
				rc = mdb_del(txn, dp, &k, NULL);
				if (rc && rc != MDB_NOTFOUND) CK(rc);
			}
		}
		for (i = 0; i < NBIG / 2; i++) {
			unsigned id = rnd(NBIG);
			int rc;
			mkbig(kb, &k, id);
			if (rnd(2)) {
				mkval(vb, &v, id, 16);
				CK(mdb_put(txn, db, &k, &v, 0));
			} else {
				rc = mdb_del(txn, db, &k, NULL);
				if (rc && rc != MDB_NOTFOUND) CK(rc);
			}
		}
		for (i = 0; i < 1500; i++) {
			unsigned ver = rnd(6000);
			int rc;
			mkkey(kb, &k, rnd(8));
			if (r & 1) {
				mkval(vb, &v, ver, 120);
				rc = mdb_del(txn, dd, &k, &v);
				if (rc && rc != MDB_NOTFOUND) CK(rc);
				mkval(vb, &v, ver, 64);
				rc = mdb_del(txn, df, &k, &v);
				if (rc && rc != MDB_NOTFOUND) CK(rc);
			} else {
				mkval(vb, &v, ver, 120);
				CK(mdb_put(txn, dd, &k, &v, 0));
				mkval(vb, &v, ver, 64);
				CK(mdb_put(txn, df, &k, &v, 0));
			}
		}
		CK(mdb_txn_commit(txn));
	}
}

static int
cmp_u32(const void *a, const void *b)
{
	uint32_t x = *(const uint32_t *)a, y = *(const uint32_t *)b;
	return x < y ? -1 : x > y;
}

static void
check(Walk *w, uint32_t ntok, const char *phase)
{
	free(w->list);			/* from an earlier check of w */
	memset(w, 0, sizeof(*w));
	w->nseen = ntok + 1;
	w->seen = calloc(w->nseen, 1);
	w->digest = 0xcbf29ce484222325ull;
	walk_file(w);
	printf("%s: %zu links, %zu tokens, %zu zero, %zu invalid, %zu duplicated, %zu bad padding\n",
		phase, w->links, w->tokens, w->zero, w->bad, w->dup, w->badpad);
	printf("DIGEST %s %016llx\n", phase, (unsigned long long)w->digest);
	if (w->bad) fail("invalid prefix bytes (neither zero nor a token)");
	if (w->dup) fail("a token appears on two links");
	if (w->badpad) fail("non-zero EVEN padding byte");
	free(w->seen);
}

int
main(void)
{
	Walk w = { 0 }, before = { 0 }, after = { 0 };
	const unsigned nk = 6000;
	uint32_t ntok;

	if (system("rm -rf " DIR " && mkdir " DIR)) {}
	env_open();
	build(nk);
	mdb_env_close(env);

	/* stamp every aggregate branch link */
	memset(&w, 0, sizeof(w));
	w.stamp = 1;
	w.next = 1;
	walk_file(&w);
	store();
	ntok = w.next - 1;
	printf("stamped %u links (MDB_HASH_SIZE %d, prefix %zu bytes)\n", ntok, MDB_HASH_SIZE,
		mdb_agg_prefix_len_flags(SCHEMA));
	if (ntok < 50)
		fail("too few branch links to test");

	/* non-structural phase */
	check(&before, ntok, "stamped");
	env_open();
	overwrite(nk);
	mdb_env_close(env);
	check(&after, ntok, "after overwrites");
	qsort(before.list, before.nlist, sizeof(uint32_t), cmp_u32);
	qsort(after.list, after.nlist, sizeof(uint32_t), cmp_u32);
	if (before.nlist != ntok || after.nlist != ntok || after.zero ||
		memcmp(before.list, after.list, ntok * sizeof(uint32_t)))
		fail("same-size overwrites changed branch prefixes");

	/* separator growth: every token survives */
	env_open();
	rekey_grow();
	mdb_env_close(env);
	check(&w, ntok, "after separator growth");
	if (w.tokens != ntok) fail("a token was lost by a separator replacement");

	/* growth: every token survives */
	env_open();
	grow(nk);
	mdb_env_close(env);
	check(&w, ntok, "after growth");
	if (w.tokens != ntok) fail("a token was lost although no link was destroyed");
	if (!w.zero) fail("no new link in the growth phase");

	/* heavy deletes, then churn */
	env_open();
	shrink();
	mdb_env_close(env);
	check(&w, ntok, "after deletes");
	if (!w.tokens) fail("no token survived the delete phase");
	env_open();
	churn(nk, 6);
	mdb_env_close(env);
	check(&w, ntok, "after churn");

	free(w.list);
	free(before.list);
	free(after.list);
	free(fbuf);
	if (failures) {
		fprintf(stderr, "F2 opaque transport: %u failures\n", failures);
		return 1;
	}
	printf("F2 opaque transport passed\n");
	return 0;
}
