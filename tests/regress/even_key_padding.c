/* E1 regression: page space accounting with EVEN() key padding.
 *
 * 01-base stores node data at NODEKEY + EVEN(ksize) (DLMDB layout), so a leaf
 * node occupies EVEN(NODESIZE + EVEN(ksize) + dsize) bytes.  Before E1,
 * mdb_node_del released EVEN(NODESIZE + ksize + dsize): 2 bytes less when both
 * the key and the data length are odd.  Every such delete leaked 2 bytes of
 * the page until the page was rewritten, so SIZELEFT drifted from the real
 * free space and splits happened too early; together with the unpadded
 * size estimate in mdb_page_split this ended in MDB_PAGE_FULL on puts that
 * LMDB accepts (found by tests/struct/diff_ops, seed 3).
 *
 * The test fills one leaf page to capacity with (odd key, odd data) records,
 * deletes all but one and inserts them again, in one transaction and across
 * transactions: the tree must stay a single leaf page, as with LMDB.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "lmdb.h"

#define C(x) do { int r_ = (x); if (r_) { fprintf(stderr, "%s:%d: %s: %s\n", \
	__FILE__, __LINE__, #x, mdb_strerror(r_)); exit(2); } } while (0)

#define KLEN 5
#define DLEN 5
/* footprint per record: EVEN(8 + EVEN(5) + 5) + sizeof(indx_t) = 22;
 * (4096 - 16) / 22 = 185 records fit in one leaf page */
#define NREC 185

static void
kv(unsigned i, char *kb, char *db)
{
	snprintf(kb, KLEN + 1, "k%04u", i);
	snprintf(db, DLEN + 1, "d%04u", i);
}

static int
cycle(MDB_env *env, MDB_dbi dbi, int split_txns, const char *what)
{
	MDB_txn *txn;
	MDB_stat st;
	char kb[16], db[16];
	unsigned i;

	C(mdb_txn_begin(env, NULL, 0, &txn));
	for (i = 1; i < NREC; i++) {
		MDB_val k = { KLEN, kb };
		kv(i, kb, db);
		C(mdb_del(txn, dbi, &k, NULL));
	}
	if (split_txns) {
		C(mdb_txn_commit(txn));
		C(mdb_txn_begin(env, NULL, 0, &txn));
	}
	for (i = 1; i < NREC; i++) {
		MDB_val k = { KLEN, kb }, v = { DLEN, db };
		kv(i, kb, db);
		C(mdb_put(txn, dbi, &k, &v, MDB_NOOVERWRITE));
	}
	C(mdb_stat(txn, dbi, &st));
	C(mdb_txn_commit(txn));
	printf("%-32s depth %u leaf pages %zu entries %zu\n", what, st.ms_depth,
		(size_t)st.ms_leaf_pages, (size_t)st.ms_entries);
	return st.ms_depth == 1 && st.ms_leaf_pages == 1 && st.ms_entries == NREC;
}

int
main(void)
{
	MDB_env *env;
	MDB_txn *txn;
	MDB_dbi dbi;
	MDB_stat st;
	char kb[16], db[16];
	unsigned i;
	int ok = 1;

	if (system("rm -rf testdb_e1 && mkdir testdb_e1")) {}
	C(mdb_env_create(&env));
	C(mdb_env_set_maxdbs(env, 2));
	C(mdb_env_open(env, "testdb_e1", MDB_NOSYNC, 0664));
	C(mdb_txn_begin(env, NULL, 0, &txn));
	C(mdb_dbi_open(txn, "e1", MDB_CREATE, &dbi));
	for (i = 0; i < NREC; i++) {
		MDB_val k = { KLEN, kb }, v = { DLEN, db };
		kv(i, kb, db);
		C(mdb_put(txn, dbi, &k, &v, 0));
	}
	C(mdb_stat(txn, dbi, &st));
	C(mdb_txn_commit(txn));
	printf("%-32s depth %u leaf pages %zu entries %zu\n", "initial fill", st.ms_depth,
		(size_t)st.ms_leaf_pages, (size_t)st.ms_entries);
	if (st.ms_leaf_pages != 1) {
		fprintf(stderr, "setup: %d records do not fit one page\n", NREC);
		return 2;
	}
	ok &= cycle(env, dbi, 0, "delete+reinsert, one txn");
	ok &= cycle(env, dbi, 1, "delete+reinsert, two txns");
	mdb_env_close(env);
	if (!ok) {
		fprintf(stderr, "E1: page space leaked by deletes (tree split)\n");
		return 1;
	}
	printf("E1 EVEN-padding page accounting test passed\n");
	return 0;
}
