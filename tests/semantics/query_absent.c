/* C5 regression: aggregate prefix/range queries with ABSENT boundaries.
 *
 * Before C5, the query layer called mdb_cursor_set(..., MDB_SET_RANGE, &exact).
 * LMDB treats a non-NULL exactp as an exact-match request, so every absent
 * boundary produced MDB_NOTFOUND and the query answered with the totals of the
 * whole DB (key-only prefix) or of the whole dupset (key+data prefix).
 *
 * This test uses only the public API and an independent cursor-scan oracle.
 * It covers: plain and DUPSORT DBs, single-leaf and multi-level trees, inline
 * dupsets and persistent duplicate sub-DBs, present/absent keys, exact and
 * non-exact data boundaries, inclusive/exclusive prefixes, and ranges.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "lmdb.h"

#define C(x) do { int r_ = (x); if (r_) { fprintf(stderr, "%s:%d: %s: %s\n", \
	__FILE__, __LINE__, #x, mdb_strerror(r_)); exit(2); } } while (0)

static MDB_env *env;
static MDB_txn *txn;
static unsigned failures;

/* records strictly before (k,d), or through it with incl; d == NULL means
 * the whole key k is the boundary. */
static uint64_t
oracle_prefix(MDB_dbi dbi, MDB_val *k, MDB_val *d, int incl)
{
	MDB_cursor *c;
	MDB_val kk, vv;
	uint64_t n = 0;
	int rc;
	C(mdb_cursor_open(txn, dbi, &c));
	for (rc = mdb_cursor_get(c, &kk, &vv, MDB_FIRST); rc == 0;
		rc = mdb_cursor_get(c, &kk, &vv, MDB_NEXT)) {
		int ck = mdb_cmp(txn, dbi, &kk, k);
		int cd = d ? mdb_dcmp(txn, dbi, &vv, d) : 0;
		if (ck < 0 || (ck == 0 && d && (cd < 0 || (incl && cd == 0))) ||
			(ck == 0 && !d && incl))
			n++;
	}
	mdb_cursor_close(c);
	return n;
}

static void
check(const char *what, uint64_t got, uint64_t exp, unsigned id)
{
	if (got != exp) {
		if (failures < 10)
			fprintf(stderr, "MISMATCH %s id=%u: got %llu expected %llu\n", what, id,
				(unsigned long long)got, (unsigned long long)exp);
		failures++;
	}
}

static void
mkkey(char *kb, unsigned id)
{
	snprintf(kb, 16, "K%08u", id);
}

static void
mkval(unsigned char *vb, unsigned id, unsigned j)
{
	memset(vb, 0, 48);
	vb[0] = 'D';
	vb[1] = (unsigned char)(id >> 8);
	vb[2] = (unsigned char)id;
	vb[3] = (unsigned char)(2 * j);		/* even: odd values are absent */
}

static void
run(int dupsort, unsigned nkeys, unsigned maxdups)
{
	MDB_dbi dbi;
	char kb[16], kb2[16];
	unsigned char vb[48], vb2[48];
	unsigned id, j;

	C(mdb_txn_begin(env, NULL, 0, &txn));
	C(mdb_dbi_open(txn, dupsort ? "dups" : "plain",
		MDB_CREATE | (dupsort ? MDB_DUPSORT : 0) | MDB_AGG_ENTRIES | MDB_AGG_KEYS, &dbi));
	C(mdb_drop(txn, dbi, 0));
	/* only even ids exist: odd ids are absent keys between present ones */
	for (id = 0; id < 2 * nkeys; id += 2) {
		unsigned nd = dupsort ? 1 + (id / 2) % maxdups : 1;
		for (j = 0; j < nd; j++) {
			MDB_val k = { 9, kb }, v = { 48, vb };
			mkkey(kb, id);
			mkval(vb, id, j);
			C(mdb_put(txn, dbi, &k, &v, 0));
		}
	}

	for (id = 0; id <= 2 * nkeys; id++) {
		int incl;
		MDB_val k = { 9, kb };
		mkkey(kb, id);
		for (incl = 0; incl < 2; incl++) {
			MDB_agg a;
			C(mdb_agg_prefix(txn, dbi, &k, NULL, incl ? MDB_AGG_PREFIX_INCL : 0, &a));
			check(incl ? "prefix key INCL" : "prefix key EXCL", a.mv_agg_entries,
				oracle_prefix(dbi, &k, NULL, incl), id);
			if (dupsort) {
				/* data boundaries: present (even j) and absent (odd j) */
				for (j = 0; j < 2 * maxdups + 1; j++) {
					MDB_val d = { 48, vb };
					mkval(vb, id, 0);
					vb[3] = (unsigned char)j;
					C(mdb_agg_prefix(txn, dbi, &k, &d, incl ? MDB_AGG_PREFIX_INCL : 0, &a));
					check(incl ? "prefix key+data INCL" : "prefix key+data EXCL",
						a.mv_agg_entries, oracle_prefix(dbi, &k, &d, incl), id);
				}
			}
		}
		/* range [id, id+3) with every inclusivity combination */
		{
			unsigned f;
			MDB_val lo = { 9, kb }, hi = { 9, kb2 };
			mkkey(kb2, id + 3);
			for (f = 0; f < 4; f++) {
				unsigned flags = (f & 1 ? MDB_RANGE_LOWER_INCL : 0) |
					(f & 2 ? MDB_RANGE_UPPER_INCL : 0);
				MDB_agg a;
				uint64_t exp = oracle_prefix(dbi, &hi, NULL, (f & 2) != 0) -
					oracle_prefix(dbi, &lo, NULL, !(f & 1));
				C(mdb_agg_range(txn, dbi, &lo, NULL, &hi, NULL, flags, &a));
				check("range key", a.mv_agg_entries, exp, id);
				if (dupsort) {
					MDB_val dl = { 48, vb }, dh = { 48, vb2 };
					mkval(vb, id, 0); vb[3] = 1;		/* absent data */
					mkval(vb2, id + 3, 0); vb2[3] = 3;	/* absent data */
					exp = oracle_prefix(dbi, &hi, &dh, (f & 2) != 0) -
						oracle_prefix(dbi, &lo, &dl, !(f & 1));
					C(mdb_agg_range(txn, dbi, &lo, &dl, &hi, &dh, flags, &a));
					check("range key+data", a.mv_agg_entries, exp, id);
				}
			}
		}
	}
	mdb_txn_abort(txn);
}

int
main(void)
{
	if (system("rm -rf testdb_c5 && mkdir testdb_c5")) {}
	C(mdb_env_create(&env));
	C(mdb_env_set_maxdbs(env, 4));
	C(mdb_env_set_mapsize(env, 256u << 20));
	C(mdb_env_open(env, "testdb_c5", MDB_NOSYNC, 0664));

	run(0, 5, 1);		/* plain, single leaf */
	run(0, 3000, 1);	/* plain, multi-level */
	run(1, 3, 3);		/* DUPSORT, inline dupsets */
	run(1, 400, 4);		/* DUPSORT, multi-level primary */
	run(1, 40, 200);	/* DUPSORT, persistent duplicate sub-DBs */

	mdb_env_close(env);
	if (failures) {
		fprintf(stderr, "C5 query regression: %u mismatches\n", failures);
		return 1;
	}
	printf("C5 absent-boundary query tests passed\n");
	return 0;
}
