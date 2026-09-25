/* B1 regression: DUPSORT sub-cursor state after writes through other handles,
 * and cursor state after a failed exact positioning.
 *
 * LMDB 0.9.70 (docs/findings/lmdb_cursor_hazards.md, H1 and H2):
 *  - a cursor keeps a copy of its dupset's MDB_db in the sub-cursor; writes
 *    through other handles, or its own delete that moves it to the next key,
 *    leave the copy stale; mdb_cursor_del(MDB_NODUPDATA), the delete of the
 *    last duplicate and mdb_cursor_count then use the stale copy and corrupt
 *    md_entries (or report a wrong count);
 *  - after a failed MDB_SET / MDB_SET_KEY / MDB_GET_BOTH / MDB_GET_BOTH_RANGE
 *    the cursor may sit on a node whose sub-cursor belongs to another key, so
 *    a relative duplicate operation reads stale state.
 *
 * 01-base (B1) re-synchronises the sub-cursor from the node before
 * mdb_cursor_del and mdb_cursor_count, and leaves the cursor unpositioned
 * after a failed exact positioning: it then behaves exactly like a freshly
 * opened cursor (EINVAL for MDB_GET_CURRENT and FIRST/LAST_DUP, MDB_NEXT*
 * starts from the first record, MDB_PREV* from the last).
 * On LMDB 0.9.70 this test fails; it documents the corrected contract.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include "lmdb.h"

#define C(x) do { int r_ = (x); if (r_) { fprintf(stderr, "%s:%d: %s: %s\n", \
	__FILE__, __LINE__, #x, mdb_strerror(r_)); exit(2); } } while (0)

static MDB_env *env;
static unsigned failures;

static void
fail(const char *what, const char *detail)
{
	fprintf(stderr, "FAIL %s: %s\n", what, detail);
	failures++;
}

/* ms_entries must equal an ordered scan, and every key's mdb_cursor_count
 * must equal its duplicates */
static void
verify(MDB_txn *txn, MDB_dbi dbi, const char *what)
{
	MDB_cursor *c;
	MDB_val k, v;
	MDB_stat st;
	size_t n = 0, perkey = 0;
	mdb_size_t cnt;
	int rc;
	char buf[128];

	C(mdb_cursor_open(txn, dbi, &c));
	for (rc = mdb_cursor_get(c, &k, &v, MDB_FIRST); rc == 0;
		rc = mdb_cursor_get(c, &k, &v, MDB_NEXT_NODUP)) {
		C(mdb_cursor_count(c, &cnt));
		perkey += cnt;
	}
	mdb_cursor_close(c);
	C(mdb_cursor_open(txn, dbi, &c));
	n = 0;
	for (rc = mdb_cursor_get(c, &k, &v, MDB_FIRST); rc == 0;
		rc = mdb_cursor_get(c, &k, &v, MDB_NEXT))
		n++;
	mdb_cursor_close(c);
	C(mdb_stat(txn, dbi, &st));
	if (st.ms_entries != n || perkey != n) {
		snprintf(buf, sizeof(buf), "ms_entries %zu, scan %zu, sum of counts %zu",
			(size_t)st.ms_entries, n, perkey);
		fail(what, buf);
	}
}

static void
mkval(unsigned char *vb, size_t len, unsigned code)
{
	memset(vb, 0, len);
	vb[0] = (unsigned char)(code >> 8);
	vb[1] = (unsigned char)code;
}

/* put key k with dups codes [from, from+n) of length len */
static void
fill(MDB_txn *txn, MDB_dbi dbi, const char *k, unsigned from, unsigned n, size_t len)
{
	unsigned char vb[512];
	unsigned i;
	for (i = 0; i < n; i++) {
		MDB_val kk = { strlen(k), (void *)k }, vv = { len, vb };
		mkval(vb, len, from + i);
		C(mdb_put(txn, dbi, &kk, &vv, 0));
	}
}

static MDB_dbi
fresh(MDB_txn *txn, unsigned flags)
{
	MDB_dbi dbi;
	C(mdb_dbi_open(txn, flags & MDB_DUPFIXED ? "f" : "d", MDB_CREATE | flags, &dbi));
	C(mdb_drop(txn, dbi, 0));
	return dbi;
}

/* H2a: own delete moves the cursor to the next key, then NODUPDATA delete */
static void
h2_own_delete(unsigned flags, size_t len, unsigned nb)
{
	MDB_txn *txn;
	MDB_dbi dbi;
	MDB_cursor *c;
	MDB_val k = { 1, "A" }, v;
	char what[96];
	snprintf(what, sizeof(what), "H2a own delete (flags %#x, len %zu, %u dups)", flags, len, nb);
	C(mdb_txn_begin(env, NULL, 0, &txn));
	dbi = fresh(txn, flags);
	fill(txn, dbi, "A", 0, 1, len);
	fill(txn, dbi, "B", 0, nb, len);
	fill(txn, dbi, "C", 0, 2, len);
	C(mdb_cursor_open(txn, dbi, &c));
	C(mdb_cursor_get(c, &k, &v, MDB_SET_KEY));
	C(mdb_cursor_del(c, 0));		/* A gone: cursor now on B */
	C(mdb_cursor_del(c, MDB_NODUPDATA));	/* all of B */
	mdb_cursor_close(c);
	verify(txn, dbi, what);
	mdb_txn_abort(txn);
}

/* H2b: another handle adds duplicates to the dupset the cursor is on */
static void
h2_other_writer(unsigned flags, size_t len, unsigned nb, int multiple, int nodup)
{
	MDB_txn *txn;
	MDB_dbi dbi;
	MDB_cursor *c, *w;
	MDB_val k = { 1, "B" }, v;
	unsigned char vb[16 * 512];
	char what[128];
	mdb_size_t cnt;
	unsigned i;
	snprintf(what, sizeof(what), "H2b other writer (flags %#x, len %zu, %u dups, %s, %s)",
		flags, len, nb, multiple ? "MULTIPLE" : "mdb_put",
		nodup ? "NODUPDATA delete" : "count");
	C(mdb_txn_begin(env, NULL, 0, &txn));
	dbi = fresh(txn, flags);
	fill(txn, dbi, "A", 0, 2, len);
	fill(txn, dbi, "B", 0, nb, len);
	C(mdb_cursor_open(txn, dbi, &c));
	C(mdb_cursor_get(c, &k, &v, MDB_SET_KEY));
	C(mdb_cursor_get(c, &k, &v, MDB_NEXT));		/* inside B's dupset */
	if (multiple) {
		MDB_val mv[2];
		for (i = 0; i < 10; i++)
			mkval(vb + i * len, len, 1000 + i);
		mv[0].mv_size = len; mv[0].mv_data = vb;
		mv[1].mv_size = 10; mv[1].mv_data = NULL;
		k.mv_size = 1; k.mv_data = "B";
		C(mdb_cursor_open(txn, dbi, &w));
		C(mdb_cursor_put(w, &k, mv, MDB_MULTIPLE));
		mdb_cursor_close(w);
	} else {
		fill(txn, dbi, "B", 1000, 10, len);
	}
	if (nodup) {
		C(mdb_cursor_del(c, MDB_NODUPDATA));
	} else {
		C(mdb_cursor_count(c, &cnt));
		if (cnt != nb + 10) {
			char d[64];
			snprintf(d, sizeof(d), "count %zu, expected %u", (size_t)cnt, nb + 10);
			fail(what, d);
		}
	}
	mdb_cursor_close(c);
	verify(txn, dbi, what);
	mdb_txn_abort(txn);
}

/* H1: relative operations after a failed exact positioning */
static void
h1_failed_set(unsigned op)
{
	MDB_txn *txn;
	MDB_dbi dbi;
	MDB_cursor *c;
	MDB_val k, v;
	unsigned char vb[64];
	unsigned rel[] = { MDB_GET_CURRENT, MDB_PREV_DUP, MDB_NEXT_DUP,
		MDB_FIRST_DUP, MDB_LAST_DUP };
	unsigned i;
	char what[96];
	int rc;
	C(mdb_txn_begin(env, NULL, 0, &txn));
	dbi = fresh(txn, MDB_DUPSORT);
	fill(txn, dbi, "A", 0, 5, 20);
	fill(txn, dbi, "C", 0, 5, 20);
	for (i = 0; i < sizeof(rel) / sizeof(rel[0]); i++) {
		C(mdb_cursor_open(txn, dbi, &c));
		k.mv_size = 1; k.mv_data = "A";
		C(mdb_cursor_get(c, &k, &v, MDB_SET_KEY));	/* sub-cursor on A */
		k.mv_size = 1; k.mv_data = "B";			/* absent key */
		mkval(vb, 20, 3);
		v.mv_size = 20; v.mv_data = vb;
		rc = mdb_cursor_get(c, &k, &v, op);
		if (rc != MDB_NOTFOUND) {
			snprintf(what, sizeof(what), "H1 op %u on absent key", op);
			fail(what, mdb_strerror(rc));
		}
		/* the cursor must behave exactly like a freshly opened one */
		{
			MDB_cursor *f;
			MDB_val fk = { 0, NULL }, fv = { 0, NULL };
			int frc;
			C(mdb_cursor_open(txn, dbi, &f));
			frc = mdb_cursor_get(f, &fk, &fv, rel[i]);
			mdb_cursor_close(f);
			k.mv_size = 0; k.mv_data = NULL;
			v.mv_size = 0; v.mv_data = NULL;
			rc = mdb_cursor_get(c, &k, &v, rel[i]);
			if (rc != frc || (rc == 0 && (v.mv_size != fv.mv_size ||
				memcmp(v.mv_data, fv.mv_data, v.mv_size)))) {
				snprintf(what, sizeof(what), "H1 op %u then op %u", op, rel[i]);
				fail(what, rc ? mdb_strerror(rc) : "differs from a fresh cursor");
			}
		}
		/* the cursor is usable again after an absolute positioning */
		C(mdb_cursor_get(c, &k, &v, MDB_FIRST));
		mdb_cursor_close(c);
	}
	mdb_txn_abort(txn);
}

/* B1b: MDB_SET_RANGE beyond the last key from an already positioned cursor
 * must leave a clean past-the-end state: MDB_NEXT -> MDB_NOTFOUND,
 * MDB_PREV -> the last record */
static void
set_range_past_end(void)
{
	MDB_txn *txn;
	MDB_dbi dbi;
	MDB_cursor *c;
	MDB_val k, v;
	int rc;
	C(mdb_txn_begin(env, NULL, 0, &txn));
	dbi = fresh(txn, MDB_DUPSORT);
	fill(txn, dbi, "A", 0, 3, 20);
	fill(txn, dbi, "C", 0, 3, 20);
	C(mdb_cursor_open(txn, dbi, &c));
	C(mdb_cursor_get(c, &k, &v, MDB_LAST));
	k.mv_size = 1; k.mv_data = "D";
	rc = mdb_cursor_get(c, &k, &v, MDB_SET_RANGE);
	if (rc != MDB_NOTFOUND)
		fail("B1b SET_RANGE past end", mdb_strerror(rc));
	rc = mdb_cursor_get(c, &k, &v, MDB_NEXT);
	if (rc != MDB_NOTFOUND)
		fail("B1b NEXT after SET_RANGE past end", rc ? mdb_strerror(rc) : "returned a record");
	k.mv_size = 1; k.mv_data = "D";
	C(mdb_cursor_get(c, &k, &v, MDB_SET_RANGE) == MDB_NOTFOUND ? 0 : 1);
	rc = mdb_cursor_get(c, &k, &v, MDB_PREV);
	if (rc || k.mv_size != 1 || *(char *)k.mv_data != 'C' || ((unsigned char *)v.mv_data)[1] != 2)
		fail("B1b PREV after SET_RANGE past end", rc ? mdb_strerror(rc) : "not the last record");
	mdb_cursor_close(c);
	mdb_txn_abort(txn);
}

int
main(void)
{
	unsigned f, i;
	static const unsigned fl[2] = { MDB_DUPSORT, MDB_DUPSORT | MDB_DUPFIXED };

	if (system("rm -rf testdb_b1 && mkdir testdb_b1")) {}
	C(mdb_env_create(&env));
	C(mdb_env_set_maxdbs(env, 4));
	C(mdb_env_set_mapsize(env, 1u << 28));
	C(mdb_env_open(env, "testdb_b1", MDB_NOSYNC, 0664));
	for (f = 0; f < 2; f++) {
		/* inline subpage (3 dups of 16 bytes) and sub-DB (200 dups) */
		h2_own_delete(fl[f], 16, 3);
		h2_own_delete(fl[f], 16, 200);
		for (i = 0; i < 4; i++) {
			if ((i & 1) && !(fl[f] & MDB_DUPFIXED))
				continue;			/* MULTIPLE needs DUPFIXED */
			h2_other_writer(fl[f], 16, 3, i & 1, i >> 1);
			h2_other_writer(fl[f], 16, 200, i & 1, i >> 1);
		}
	}
	h1_failed_set(MDB_SET);
	h1_failed_set(MDB_SET_KEY);
	h1_failed_set(MDB_GET_BOTH);
	h1_failed_set(MDB_GET_BOTH_RANGE);
	set_range_past_end();
	mdb_env_close(env);
	if (failures) {
		fprintf(stderr, "B1 cursor synchronisation: %u failures\n", failures);
		return 1;
	}
	printf("B1 cursor synchronisation tests passed\n");
	return 0;
}
