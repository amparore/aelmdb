/* key_aliasing.c - characterization of LMDB key-pointer aliasing on update.
 *
 * LMDB returns keys/values that point into database pages; they are valid
 * only until the next update operation.  If such a pointer is passed back as
 * the key of a put that REPLACES an existing value with one of a DIFFERENT
 * size, LMDB deletes the old node (moving page memory) and then re-reads the
 * key through the stale pointer: the record is re-inserted under a
 * neighbouring key.  The original key disappears and a duplicate appears.
 *
 * This file does not assert "correct" behaviour: it prints what happens for
 * each variant.  `make test-lmdb` requires every stage to print exactly the
 * same report as baselines/lmdb, so that the bottom-up refactoring preserves
 * LMDB semantics, including this one.  See docs/findings/key_aliasing.md.
 *
 * Correct usage (what applications, tests and the aggregate layers assume):
 * copy the key into caller-owned memory before an update that may resize.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "lmdb.h"

#define C(x) do { int r_ = (x); if (r_) { \
	fprintf(stderr, "%s:%d: %s: %s\n", __FILE__, __LINE__, #x, mdb_strerror(r_)); \
	exit(2); } } while (0)

enum { V_CUR_GROW, V_CUR_OVERFLOW, V_CUR_SAME, V_CUR_SHRINK, V_PUT_FROM_CURSOR,
	V_PUT_FROM_SETKEY, V_CUR_GROW_COPIED, NVARIANTS };

static const char *vname[NVARIANTS] = {
	"cursor_put(CURRENT), aliased key, value grows",
	"cursor_put(CURRENT), aliased key, value -> overflow",
	"cursor_put(CURRENT), aliased key, same size",
	"cursor_put(CURRENT), aliased key, value shrinks",
	"mdb_put, key pointer from cursor, value grows",
	"mdb_put, key pointer from MDB_SET_KEY, value grows",
	"cursor_put(CURRENT), key COPIED by caller, value grows",
};

static void
run(int variant)
{
	MDB_env *env;
	MDB_txn *txn;
	MDB_dbi dbi;
	MDB_cursor *cur;
	MDB_val k, v, nv;
	static char vb[8192];
	char kb[16], saved[16], prev[16];
	unsigned i, n = 0;
	int rc, dup = 0, present;
	size_t newsz = 100;

	if (system("rm -rf testdb && mkdir testdb")) {}
	C(mdb_env_create(&env));
	C(mdb_env_set_maxdbs(env, 2));
	C(mdb_env_open(env, "testdb", MDB_NOSYNC, 0664));
	C(mdb_txn_begin(env, NULL, 0, &txn));
	C(mdb_dbi_open(txn, "x", MDB_CREATE, &dbi));
	for (i = 0; i < 50; i++) {
		snprintf(kb, sizeof(kb), "K%09u", i);
		memset(vb, (int)i, 40);
		k.mv_size = 10; k.mv_data = kb;
		v.mv_size = 40; v.mv_data = vb;
		C(mdb_put(txn, dbi, &k, &v, 0));
	}
	C(mdb_cursor_open(txn, dbi, &cur));
	C(mdb_cursor_get(cur, &k, &v, MDB_FIRST));
	for (i = 0; i < 3; i++)
		C(mdb_cursor_get(cur, &k, &v, MDB_NEXT));	/* on K000000003 */
	memcpy(saved, k.mv_data, 10);

	if (variant == V_CUR_OVERFLOW) newsz = 3000;
	if (variant == V_CUR_SAME) newsz = 40;
	if (variant == V_CUR_SHRINK) newsz = 10;
	memset(vb, 0x5a, newsz);
	nv.mv_size = newsz; nv.mv_data = vb;

	switch (variant) {
	case V_PUT_FROM_CURSOR:
		C(mdb_put(txn, dbi, &k, &nv, 0));
		break;
	case V_PUT_FROM_SETKEY: {
		MDB_cursor *c2;
		MDB_val sk = { 10, saved }, sv;
		C(mdb_cursor_open(txn, dbi, &c2));
		C(mdb_cursor_get(c2, &sk, &sv, MDB_SET_KEY));	/* sk now points into the page */
		C(mdb_put(txn, dbi, &sk, &nv, 0));
		mdb_cursor_close(c2);
		break;
	}
	case V_CUR_GROW_COPIED: {
		MDB_val ck = { 10, saved };			/* caller-owned copy */
		C(mdb_cursor_put(cur, &ck, &nv, MDB_CURRENT));
		break;
	}
	default:
		C(mdb_cursor_put(cur, &k, &nv, MDB_CURRENT));
	}

	k.mv_size = 10; k.mv_data = saved;
	present = mdb_get(txn, dbi, &k, &v) == MDB_SUCCESS;
	prev[0] = 0;
	for (rc = mdb_cursor_get(cur, &k, &v, MDB_FIRST); rc == 0;
		rc = mdb_cursor_get(cur, &k, &v, MDB_NEXT)) {
		if (n && !memcmp(prev, k.mv_data, 10))
			dup = 1;
		memcpy(prev, k.mv_data, 10);
		n++;
	}
	printf("%-56s : updated key %s, records %u, duplicate keys %s\n",
		vname[variant], present ? "present" : "LOST", n, dup ? "YES" : "no");
	mdb_cursor_close(cur);
	mdb_txn_abort(txn);
	mdb_env_close(env);
}

int
main(void)
{
	int i;
	for (i = 0; i < NVARIANTS; i++)
		run(i);
	return 0;
}
