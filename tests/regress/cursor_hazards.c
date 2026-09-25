/* B1c-B1f regression: DUPSORT cursor state after writes through other
 * handles and after the cursor's own delete (docs/findings/
 * lmdb_cursor_hazards.md, H2 MDB_CURRENT case, H4, H5, H6).  Found by the
 * coverage-focus op stream (tests/struct/diff_ops.c, DS_FOCUS=1).
 *
 *  H2c  MDB_CURRENT through a cursor whose sub-cursor copy of the dupset
 *       is stale: the nested put writes the stale MDB_db back into the
 *       node (01-base B1c synchronises the sub-cursor first);
 *  H4   the cursor's own delete removes a key without duplicates and moves
 *       the cursor onto a key with a sub-DB: the sub-cursor of an earlier
 *       key survives (B1d clears it; the cursor is on the first duplicate);
 *  H5   a cursor put inserts a key beyond the last one and leaves C_EOF set:
 *       once the key's dupset grows, MDB_NEXT stops (B1e);
 *  H6   FIRST_DUP / LAST_DUP re-search the duplicate tree from the
 *       sub-cursor's copy of its root, which another handle's write moved
 *       (copy on write): the cursor reads an old page (B1f).
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

/* ms_entries must equal an ordered scan and the sum of mdb_cursor_count */
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

static unsigned
code_of(const MDB_val *v)
{
	const unsigned char *b = v->mv_data;
	return v->mv_size >= 2 ? (unsigned)b[0] << 8 | b[1] : ~0u;
}

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

/* a committed DB (clean pages: the next write copies them) */
static MDB_dbi
setup(unsigned flags, const char *keys, char single, unsigned ndup, size_t len)
{
	MDB_txn *txn;
	MDB_dbi dbi;
	const char *p;
	char k[2] = { 0, 0 };
	C(mdb_txn_begin(env, NULL, 0, &txn));
	C(mdb_dbi_open(txn, flags & MDB_DUPFIXED ? "f" : "d", MDB_CREATE | flags, &dbi));
	C(mdb_drop(txn, dbi, 0));
	for (p = keys; *p; p++) {
		k[0] = *p;
		if (*p == single)
			fill(txn, dbi, k, 7, 1, len);
		else
			fill(txn, dbi, k, 0, ndup, len);
	}
	C(mdb_txn_commit(txn));
	return dbi;
}

/* H2c: MDB_CURRENT after another handle added duplicates */
static void
h2c_current(unsigned flags, unsigned nb)
{
	MDB_txn *txn;
	MDB_dbi dbi = setup(flags, "AB", 0, nb, 16);
	MDB_cursor *c;
	MDB_val k = { 1, "B" }, v;
	unsigned char kb[8], vb[16];
	char what[96];
	snprintf(what, sizeof(what), "H2c MDB_CURRENT (flags %#x, %u dups)", flags, nb);
	C(mdb_txn_begin(env, NULL, 0, &txn));
	C(mdb_cursor_open(txn, dbi, &c));
	C(mdb_cursor_get(c, &k, &v, MDB_SET_KEY));
	C(mdb_cursor_get(c, &k, &v, MDB_NEXT));
	fill(txn, dbi, "B", 1000, 10, 16);		/* another handle */
	C(mdb_cursor_get(c, &k, &v, MDB_GET_CURRENT));
	memcpy(kb, k.mv_data, k.mv_size);
	memcpy(vb, v.mv_data, v.mv_size);
	k.mv_data = kb;
	v.mv_data = vb;
	C(mdb_cursor_put(c, &k, &v, MDB_CURRENT));
	mdb_cursor_close(c);
	verify(txn, dbi, what);
	mdb_txn_abort(txn);
}

/* H4: own delete of a single-value key moves onto a key with a sub-DB */
static void
h4_own_delete(unsigned flags, unsigned nb)
{
	MDB_txn *txn;
	MDB_dbi dbi = setup(flags, "ABC", 'B', nb, 16);
	MDB_cursor *c;
	MDB_val k = { 1, "A" }, v;
	char what[96], d[64];
	int i, rc;
	snprintf(what, sizeof(what), "H4 own delete (flags %#x, %u dups)", flags, nb);
	C(mdb_txn_begin(env, NULL, 0, &txn));
	C(mdb_cursor_open(txn, dbi, &c));
	C(mdb_cursor_get(c, &k, &v, MDB_SET_KEY));
	for (i = 0; i < 5; i++)
		C(mdb_cursor_get(c, &k, &v, MDB_NEXT_DUP));	/* inside A */
	C(mdb_cursor_get(c, &k, &v, MDB_NEXT_NODUP));		/* B: no dupset */
	C(mdb_cursor_del(c, 0));				/* now on C */
	rc = mdb_cursor_get(c, &k, &v, MDB_GET_CURRENT);
	if (rc || k.mv_size != 1 || *(char *)k.mv_data != 'C' || code_of(&v) != 0) {
		snprintf(d, sizeof(d), "after the delete: %s, key %.1s, code %u",
			rc ? mdb_strerror(rc) : "ok", rc ? "-" : (char *)k.mv_data,
			rc ? 0 : code_of(&v));
		fail(what, d);
	}
	C(mdb_cursor_del(c, 0));
	C(mdb_cursor_get(c, &k, &v, MDB_LAST_DUP));
	C(mdb_cursor_del(c, 0));
	mdb_cursor_close(c);
	verify(txn, dbi, what);
	mdb_txn_abort(txn);
}

/* H5: a cursor put of a new last key, then its dupset grows */
static void
h5_put_past_end(unsigned flags, int positioned)
{
	MDB_txn *txn;
	MDB_dbi dbi = setup(flags, "AB", 0, 3, 16);
	MDB_cursor *c;
	MDB_val k = { 1, "Z" }, v;
	unsigned char vb[16];
	char what[96];
	int rc;
	snprintf(what, sizeof(what), "H5 put past the end (flags %#x, %s)", flags,
		positioned ? "cursor on the last leaf" : "fresh cursor");
	C(mdb_txn_begin(env, NULL, 0, &txn));
	C(mdb_cursor_open(txn, dbi, &c));
	if (positioned)
		C(mdb_cursor_get(c, &k, &v, MDB_LAST));
	mkval(vb, 16, 5);
	k.mv_size = 1; k.mv_data = "Z";
	v.mv_size = 16; v.mv_data = vb;
	C(mdb_cursor_put(c, &k, &v, 0));
	fill(txn, dbi, "Z", 9, 1, 16);			/* another handle */
	rc = mdb_cursor_get(c, &k, &v, MDB_NEXT);
	if (rc || code_of(&v) != 9)
		fail(what, rc ? mdb_strerror(rc) : "MDB_NEXT: not the new duplicate");
	mdb_cursor_close(c);
	verify(txn, dbi, what);
	mdb_txn_abort(txn);
}

/* H6: FIRST_DUP / LAST_DUP after another handle wrote to the sub-DB */
static void
h6_first_last(unsigned flags, unsigned nb)
{
	MDB_txn *txn;
	MDB_dbi dbi = setup(flags, "AB", 0, nb, 16);
	MDB_cursor *c;
	MDB_val k = { 1, "B" }, v, dk = { 1, "B" }, dv;
	unsigned char vb[16];
	char what[96], d[64];
	int rc;
	snprintf(what, sizeof(what), "H6 FIRST/LAST_DUP (flags %#x, %u dups)", flags, nb);
	C(mdb_txn_begin(env, NULL, 0, &txn));
	C(mdb_cursor_open(txn, dbi, &c));
	C(mdb_cursor_get(c, &k, &v, MDB_SET_KEY));
	fill(txn, dbi, "B", 5000, 1, 16);		/* copies the root path */
	rc = mdb_cursor_get(c, &k, &v, MDB_LAST_DUP);
	if (rc || code_of(&v) != 5000) {
		snprintf(d, sizeof(d), "LAST_DUP: %s, code %u", rc ? mdb_strerror(rc) : "ok",
			rc ? 0 : code_of(&v));
		fail(what, d);
	}
	mkval(vb, 16, 0);
	dv.mv_size = 16; dv.mv_data = vb;
	C(mdb_del(txn, dbi, &dk, &dv));			/* another handle */
	rc = mdb_cursor_get(c, &k, &v, MDB_FIRST_DUP);
	if (rc || code_of(&v) != 1) {
		snprintf(d, sizeof(d), "FIRST_DUP: %s, code %u", rc ? mdb_strerror(rc) : "ok",
			rc ? 0 : code_of(&v));
		fail(what, d);
	}
	mdb_cursor_close(c);
	verify(txn, dbi, what);
	mdb_txn_abort(txn);
}

int
main(void)
{
	unsigned f;
	static const unsigned fl[2] = { MDB_DUPSORT, MDB_DUPSORT | MDB_DUPFIXED };

	if (system("rm -rf testdb_b1c && mkdir testdb_b1c")) {}
	C(mdb_env_create(&env));
	C(mdb_env_set_maxdbs(env, 4));
	C(mdb_env_set_mapsize(env, 1u << 28));
	C(mdb_env_open(env, "testdb_b1c", MDB_NOSYNC, 0664));
	for (f = 0; f < 2; f++) {
		h2c_current(fl[f], 400);
		h4_own_delete(fl[f], 400);
		h5_put_past_end(fl[f], 0);
		h5_put_past_end(fl[f], 1);
		h6_first_last(fl[f], 400);
	}
	mdb_env_close(env);
	if (failures) {
		fprintf(stderr, "B1c cursor hazards: %u failures\n", failures);
		return 1;
	}
	printf("B1c cursor hazard tests passed\n");
	return 0;
}
