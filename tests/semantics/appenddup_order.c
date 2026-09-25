/* C7 regression: a failed MDB_APPENDDUP must not leave unpublished changes.
 *
 * LMDB checks the MDB_APPENDDUP order inside the nested duplicate put.  For a
 * key holding a single value, the primary put first converts the value into a
 * dupset, which can split the primary leaf; then the nested put fails with
 * MDB_KEYEXIST.  Before C7 the aggregate wrapper returned without its finish
 * phase and the split links kept stale aggregates.
 *
 * Every case must return MDB_KEYEXIST (LMDB's result), leave the content
 * unchanged, and keep totals and (with -DMDB_DEBUG_AGG_INTEGRITY=1) the
 * recursive integrity oracle exact.  In-order MDB_APPENDDUP must still work.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "lmdb.h"

#define C(x) do { int r_ = (x); if (r_) { fprintf(stderr, "%s:%d: %s: %s\n", \
	__FILE__, __LINE__, #x, mdb_strerror(r_)); exit(2); } } while (0)

static MDB_env *env;
static unsigned failures;

static uint64_t
scan_count(MDB_txn *txn, MDB_dbi dbi)
{
	MDB_cursor *c;
	MDB_val k, v;
	uint64_t n = 0;
	int rc;
	C(mdb_cursor_open(txn, dbi, &c));
	for (rc = mdb_cursor_get(c, &k, &v, MDB_FIRST); rc == 0;
		rc = mdb_cursor_get(c, &k, &v, MDB_NEXT))
		n++;
	mdb_cursor_close(c);
	return n;
}

static void
verify(MDB_txn *txn, MDB_dbi dbi, uint64_t expect, const char *what)
{
	MDB_agg a;
	uint64_t n = scan_count(txn, dbi);
	int rc;
	C(mdb_agg_totals(txn, dbi, &a));
	if (n != expect || a.mv_agg_entries != expect) {
		fprintf(stderr, "%s: scan %llu totals %llu expected %llu\n", what,
			(unsigned long long)n, (unsigned long long)a.mv_agg_entries,
			(unsigned long long)expect);
		failures++;
	}
#if defined(MDB_DEBUG_AGG_INTEGRITY) && MDB_DEBUG_AGG_INTEGRITY
	rc = mdb_agg_check_integrity(txn, dbi);
	if (rc) {
		fprintf(stderr, "%s: integrity %s\n", what, mdb_strerror(rc));
		failures++;
	}
#else
	(void)rc;
#endif
}

/* 16-bit big-endian value code in the first bytes: memcmp order == code order */
static void
setval(unsigned char *vb, unsigned code)
{
	memset(vb, 0, 600);
	vb[0] = (unsigned char)(code >> 8);
	vb[1] = (unsigned char)code;
}

static void
run(unsigned dbflags, int nkeys, int ndup, size_t vlen)
{
	MDB_txn *txn;
	MDB_dbi dbi;
	MDB_val k, v;
	char kb[16], what[128];
	unsigned char vb[600];
	int i, j, rc;
	uint64_t total = (uint64_t)nkeys * ndup;

	/* each case in its own aborted transaction: a fresh DB every time */
	C(mdb_txn_begin(env, NULL, 0, &txn));
	C(mdb_dbi_open(txn, "x", MDB_CREATE | dbflags |
		MDB_AGG_ENTRIES | MDB_AGG_KEYS | MDB_AGG_HASHSUM, &dbi));
	for (i = 0; i < nkeys; i++)
		for (j = 0; j < ndup; j++) {
			snprintf(kb, sizeof(kb), "K%06d", i);
			setval(vb, 10 + 10 * (unsigned)j);
			k.mv_size = 7; k.mv_data = kb;
			v.mv_size = vlen; v.mv_data = vb;
			C(mdb_put(txn, dbi, &k, &v, 0));
		}
	snprintf(what, sizeof(what), "flags %#x keys %d dups %d vlen %zu", dbflags,
		nkeys, ndup, vlen);
	/* out of order: smaller than every duplicate, and equal to the last */
	for (i = 0; i < 2; i++) {
		snprintf(kb, sizeof(kb), "K%06d", nkeys / 2);
		setval(vb, i ? 10 + 10 * (unsigned)(ndup - 1) : 5);
		k.mv_size = 7; k.mv_data = kb;
		v.mv_size = vlen; v.mv_data = vb;
		rc = mdb_put(txn, dbi, &k, &v, MDB_APPENDDUP);
		if (rc != MDB_KEYEXIST) {
			fprintf(stderr, "%s: out-of-order APPENDDUP returned %s\n", what,
				mdb_strerror(rc));
			failures++;
		}
		verify(txn, dbi, total, what);
	}
	/* in order: larger than the last duplicate */
	setval(vb, 15 + 10 * (unsigned)ndup);
	k.mv_size = 7; k.mv_data = kb;
	v.mv_size = vlen; v.mv_data = vb;
	C(mdb_put(txn, dbi, &k, &v, MDB_APPENDDUP));
	verify(txn, dbi, total + 1, what);
	mdb_txn_abort(txn);
}

int
main(void)
{
	int nk, nd;
	size_t vl;

	if (system("rm -rf testdb_c7 && mkdir testdb_c7")) {}
	C(mdb_env_create(&env));
	C(mdb_env_set_maxdbs(env, 2));
	C(mdb_env_set_mapsize(env, 1u << 28));
	C(mdb_env_open(env, "testdb_c7", MDB_NOSYNC, 0664));
	/* values hold at least one hash slice (value-source HASHSUM) */
#define VMIN (MDB_HASH_SIZE > 32 ? MDB_HASH_SIZE : 32)
	for (nk = 1; nk <= 301; nk += 300) {
		for (nd = 1; nd <= 22; nd += 3)
			for (vl = VMIN; vl <= 480; vl += 16)
				run(MDB_DUPSORT, nk, nd, vl);
		for (nd = 1; nd <= 250; nd += 7)
			run(MDB_DUPSORT | MDB_DUPFIXED, nk, nd, VMIN);
	}
	mdb_env_close(env);
	if (failures) {
		fprintf(stderr, "C7 APPENDDUP regression: %u failures\n", failures);
		return 1;
	}
	printf("C7 APPENDDUP order tests passed\n");
	return 0;
}
