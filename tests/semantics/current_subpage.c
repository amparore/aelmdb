/* C6 regression: mdb_cursor_put(MDB_CURRENT) on DUPSORT aggregate DBs.
 *
 * Before C6, a same-data MDB_CURRENT put on a duplicate stored in an inline
 * subpage failed with MDB_INCOMPATIBLE (and marked the transaction as
 * failed) when the containing page was clean at positioning time, or after
 * other writes had moved it: the finish phase compared the subpage's
 * synthetic pgno with the sub-cursor's stale md_root.
 *
 * Matrix: plain / DUPSORT inline / DUPSORT sub-DB / DUPFIXED, one leaf or
 * many, positioned in the same or in a fresh transaction, ENTRIES-only and
 * ENTRIES|KEYS|HASHSUM.  After each put the aggregate totals must equal a
 * cursor-scan oracle, and with -DMDB_DEBUG_AGG_INTEGRITY=1 the recursive
 * integrity check must pass.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "lmdb.h"

#define C(x) do { int r_ = (x); if (r_) { fprintf(stderr, "%s:%d: %s: %s\n", \
	__FILE__, __LINE__, #x, mdb_strerror(r_)); return 1; } } while (0)

static int
check_totals(MDB_txn *txn, MDB_dbi dbi)
{
	MDB_cursor *c;
	MDB_val k, v;
	MDB_agg a;
	uint64_t n = 0;
	int rc;
	C(mdb_cursor_open(txn, dbi, &c));
	for (rc = mdb_cursor_get(c, &k, &v, MDB_FIRST); rc == 0;
		rc = mdb_cursor_get(c, &k, &v, MDB_NEXT))
		n++;
	mdb_cursor_close(c);
	C(mdb_agg_totals(txn, dbi, &a));
	if (a.mv_agg_entries != n) {
		fprintf(stderr, "totals: entries %llu, scan %llu\n",
			(unsigned long long)a.mv_agg_entries, (unsigned long long)n);
		return 1;
	}
#if defined(MDB_DEBUG_AGG_INTEGRITY) && MDB_DEBUG_AGG_INTEGRITY
	C(mdb_agg_check_integrity(txn, dbi));
#endif
	return 0;
}

static int
run(unsigned dbflags, unsigned agg, int nkeys, int ndup, size_t vlen, int newtxn)
{
	MDB_env *env;
	MDB_txn *txn;
	MDB_dbi dbi;
	MDB_cursor *c;
	MDB_val k, v;
	char kb[16];
	unsigned char vb[600], kc[16], vc[600];
	int i, j, rc, pass;

	if (system("rm -rf testdb_c6 && mkdir testdb_c6")) {}
	C(mdb_env_create(&env));
	C(mdb_env_set_maxdbs(env, 2));
	C(mdb_env_set_mapsize(env, 1u << 26));
	C(mdb_env_open(env, "testdb_c6", MDB_NOSYNC, 0664));
	C(mdb_txn_begin(env, NULL, 0, &txn));
	C(mdb_dbi_open(txn, "x", MDB_CREATE | dbflags | agg, &dbi));
	for (i = 0; i < nkeys; i++)
		for (j = 0; j < ndup; j++) {
			snprintf(kb, sizeof(kb), "K%06d", i);
			memset(vb, 0, sizeof(vb));
			vb[0] = (unsigned char)j; vb[1] = (unsigned char)i; vb[2] = (unsigned char)(i >> 8);
			k.mv_size = 7; k.mv_data = kb;
			v.mv_size = vlen; v.mv_data = vb;
			C(mdb_put(txn, dbi, &k, &v, 0));
		}
	/* two passes: CURRENT on every other key, then on the rest */
	for (pass = 0; pass < 2; pass++) {
		if (newtxn) {
			C(mdb_txn_commit(txn));
			C(mdb_txn_begin(env, NULL, 0, &txn));
		}
		C(mdb_cursor_open(txn, dbi, &c));
		for (i = pass; i < nkeys; i += 2) {
			snprintf(kb, sizeof(kb), "K%06d", i);
			k.mv_size = 7; k.mv_data = kb;
			C(mdb_cursor_get(c, &k, &v, MDB_SET_KEY));
			if (ndup > 1)
				C(mdb_cursor_get(c, &k, &v, MDB_NEXT_DUP));
			/* caller-owned copies (docs/findings/key_aliasing.md) */
			memcpy(kc, k.mv_data, k.mv_size);
			memcpy(vc, v.mv_data, v.mv_size);
			k.mv_data = kc; v.mv_data = vc;
			if (!(dbflags & MDB_DUPSORT))
				vc[vlen - 1] ^= 0x5a;	/* plain DB: really replace */
			rc = mdb_cursor_put(c, &k, &v, MDB_CURRENT);
			if (rc) {
				fprintf(stderr, "CURRENT put: flags %#x agg %#x keys %d dups %d "
					"vlen %zu newtxn %d key %d: %s\n", dbflags, agg, nkeys, ndup,
					vlen, newtxn, i, mdb_strerror(rc));
				return 1;
			}
			if (i % 97 == 0 && check_totals(txn, dbi))
				return 1;
		}
		mdb_cursor_close(c);
		if (check_totals(txn, dbi))
			return 1;
	}
	C(mdb_txn_commit(txn));
	mdb_env_close(env);
	return 0;
}

int
main(void)
{
	static const unsigned aggs[2] = {
		MDB_AGG_ENTRIES, MDB_AGG_ENTRIES | MDB_AGG_KEYS | MDB_AGG_HASHSUM };
	int a, nt, nk, bad = 0;

	for (a = 0; a < 2; a++)
		for (nt = 0; nt < 2; nt++)
			for (nk = 1; nk <= 2000; nk *= 20) {
				bad += run(0, aggs[a], nk, 1, 40, nt);
				bad += run(MDB_DUPSORT, aggs[a], nk, 3, 40, nt);
				bad += run(MDB_DUPSORT, aggs[a], nk, 60, 400, nt);
				bad += run(MDB_DUPSORT | MDB_DUPFIXED, aggs[a], nk, 3, 40, nt);
			}
	if (bad) {
		fprintf(stderr, "C6 MDB_CURRENT regression: %d failing configurations\n", bad);
		return 1;
	}
	printf("C6 MDB_CURRENT subpage tests passed\n");
	return 0;
}
