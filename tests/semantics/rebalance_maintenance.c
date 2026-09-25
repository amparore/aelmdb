#ifndef _GNU_SOURCE
#define _GNU_SOURCE 1
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>

#include "aelmdb_mdb.c"

#define CHECK(cond) do { \
	if (!(cond)) { \
		fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, #cond); \
		return 1; \
	} \
} while (0)

static void
fill_bytes(uint8_t *p, size_t n, unsigned seed)
{
	size_t i;
	for (i = 0; i < n; ++i)
		p[i] = (uint8_t)(seed + 29u * (unsigned)i);
}

static int
agg_equal(uint16_t agg, const MDB_aggval *a, const MDB_aggval *b)
{
	if ((agg & MDB_AGG_ENTRIES) && a->entries != b->entries)
		return 0;
	if ((agg & MDB_AGG_KEYS) && a->keys != b->keys)
		return 0;
	if ((agg & MDB_AGG_HASHSUM) &&
		memcmp(a->hashsum, b->hashsum, MDB_HASH_SIZE) != 0)
		return 0;
	return 1;
}

static int
add_record(MDB_db *db, const MDB_val *key, const MDB_val *data, MDB_aggval *total)
{
	MDB_val source;
	MDB_aggval one;
	uint16_t agg = DB_AGGFLAGS(db);
	int rc;

	if ((agg & MDB_AGG_HASHSUM) &&
		(db->md_flags & MDB_AGG_HASHSOURCE_FROM_KEY))
		source = *key;
	else
		source = *data;
	rc = mdb_agg_record_contribution(agg, db->md_hash_offset, &source, &one);
	if (rc)
		return rc;
	return mdb_aggval_add(agg, total, &one);
}

static int
sub_record(MDB_db *db, const MDB_val *key, const MDB_val *data, MDB_aggval *total)
{
	MDB_val source;
	MDB_aggval one;
	uint16_t agg = DB_AGGFLAGS(db);
	int rc;

	if ((agg & MDB_AGG_HASHSUM) &&
		(db->md_flags & MDB_AGG_HASHSOURCE_FROM_KEY))
		source = *key;
	else
		source = *data;
	rc = mdb_agg_record_contribution(agg, db->md_hash_offset, &source, &one);
	if (rc)
		return rc;
	return mdb_aggval_sub(agg, total, &one);
}

static int
check_db_total(MDB_txn *txn, MDB_dbi dbi, const MDB_aggval *expected)
{
	MDB_db *db = &txn->mt_dbs[dbi];
	MDB_aggval got;
	mdb_db_get_aggval(db, &got);
	return agg_equal(DB_AGGFLAGS(db), &got, expected) ? 0 : 1;
}

static int
verify_subtree(MDB_cursor *mc, pgno_t pgno, MDB_aggval *out)
{
	MDB_page *mp;
	MDB_aggval total, child, stored;
	uint16_t agg = DB_AGGFLAGS(mc->mc_db);
	unsigned int i;
	int rc;

	rc = mdb_page_get(mc, pgno, &mp, NULL);
	if (rc)
		return rc;
	if (IS_LEAF(mp))
		return mdb_page_agg_local(mc, mp, out);
	if (!IS_BRANCH(mp))
		return MDB_CORRUPTED;
	mdb_aggval_zero(&total);
	for (i = 0; i < NUMKEYS(mp); ++i) {
		MDB_node *node = NODEPTR(mp, i);
		rc = verify_subtree(mc, NODEPGNO(node), &child);
		if (rc)
			return rc;
		mdb_node_get_aggval(mp, node, &stored);
		if (!agg_equal(agg, &stored, &child))
			return MDB_CORRUPTED;
		rc = mdb_aggval_add(agg, &total, &child);
		if (rc)
			return rc;
	}
	*out = total;
	return MDB_SUCCESS;
}

static int
verify_tree(MDB_txn *txn, MDB_dbi dbi, const MDB_aggval *expected)
{
	MDB_cursor mc;
	MDB_xcursor mx;
	MDB_aggval got;
	int rc;

	if (txn->mt_dbs[dbi].md_root == P_INVALID) {
		MDB_aggval zero;
		mdb_aggval_zero(&zero);
		return agg_equal(DB_AGGFLAGS(&txn->mt_dbs[dbi]), &zero, expected) ? 0 : MDB_CORRUPTED;
	}
	mdb_cursor_init(&mc, txn, dbi, &mx);
	rc = verify_subtree(&mc, txn->mt_dbs[dbi].md_root, &got);
	if (rc)
		return rc;
	return agg_equal(DB_AGGFLAGS(&txn->mt_dbs[dbi]), &got, expected) ? MDB_SUCCESS : MDB_CORRUPTED;
}

static int
delete_will_rebalance(MDB_txn *txn, MDB_dbi dbi, MDB_val *key, int *will)
{
	MDB_cursor mc;
	MDB_xcursor mx;
	MDB_node *node;
	MDB_page *mp;
	size_t free_after, usable;
	unsigned int nkeys_after;
	long fill_after;
	int exact = 0, rc;

	*will = 0;
	mdb_cursor_init(&mc, txn, dbi, &mx);
	rc = mdb_page_search(&mc, key, 0);
	if (rc)
		return rc;
	node = mdb_node_search(&mc, key, &exact);
	if (!node || !exact)
		return MDB_NOTFOUND;
	mp = mc.mc_pg[mc.mc_top];
	if (!IS_LEAF(mp) || IS_LEAF2(mp))
		return MDB_CORRUPTED;
	if (mc.mc_snum < 2) {
		*will = NUMKEYS(mp) == 1;
		return MDB_SUCCESS;
	}
	free_after = (size_t)SIZELEFT(mp) + mdb_node_footprint(mp, node);
	usable = (size_t)txn->mt_env->me_psize - PAGEHDRSZ;
	if (free_after > usable)
		return MDB_CORRUPTED;
	nkeys_after = NUMKEYS(mp) - 1;
	fill_after = (long)(1000L * (usable - free_after) / usable);
	*will = nkeys_after < 1 || fill_after < FILL_THRESHOLD;
	return MDB_SUCCESS;
}

static void
cleanup_env(MDB_env *env, MDB_txn *txn, const char *dir)
{
	char data_path[256], lock_path[256];
	if (txn)
		mdb_txn_abort(txn);
	if (env)
		mdb_env_close(env);
	snprintf(data_path, sizeof(data_path), "%s/data.mdb", dir);
	snprintf(lock_path, sizeof(lock_path), "%s/lock.mdb", dir);
	unlink(data_path);
	unlink(lock_path);
	rmdir(dir);
}

static int
test_delete_rebalance(void)
{
	const uint16_t agg = MDB_AGG_ENTRIES|MDB_AGG_KEYS|MDB_AGG_HASHSUM;
	char dir[] = "/tmp/aelmdb-rebalance-delete-XXXXXX";
	MDB_env *env = NULL;
	MDB_txn *txn = NULL;
	MDB_dbi dbi;
	MDB_aggval expected;
	uint8_t value[MDB_HASH_SIZE+160];
	char keybuf[32];
	unsigned i;
	int saw_move = 0, saw_leaf_merge = 0, saw_branch_merge = 0, saw_collapse = 0;
	int rc = 1;

	CHECK(mkdtemp(dir) != NULL);
	fill_bytes(value, sizeof(value), 37);
	CHECK(mdb_env_create(&env) == MDB_SUCCESS);
	CHECK(mdb_env_set_mapsize(env, 64u * 1024u * 1024u) == MDB_SUCCESS);
	CHECK(mdb_env_open(env, dir, 0, 0600) == MDB_SUCCESS);
	CHECK(mdb_txn_begin(env, NULL, 0, &txn) == MDB_SUCCESS);
	CHECK(mdb_dbi_open(txn, NULL, agg, &dbi) == MDB_SUCCESS);
	CHECK(mdb_set_hash_offset(txn, dbi, 5) == MDB_SUCCESS);
	mdb_aggval_zero(&expected);

	for (i = 0; i < 3000; ++i) {
		MDB_val key, data;
		snprintf(keybuf, sizeof(keybuf), "d%06u", i);
		key.mv_data = keybuf; key.mv_size = strlen(keybuf);
		data.mv_data = value; data.mv_size = sizeof(value);
		CHECK(mdb_put(txn, dbi, &key, &data, MDB_APPEND) == MDB_SUCCESS);
		CHECK(add_record(&txn->mt_dbs[dbi], &key, &data, &expected) == MDB_SUCCESS);
	}
	CHECK(txn->mt_dbs[dbi].md_depth >= 3);
	CHECK(verify_tree(txn, dbi, &expected) == MDB_SUCCESS);
	CHECK(check_db_total(txn, dbi, &expected) == 0);
	CHECK(mdb_txn_commit(txn) == MDB_SUCCESS);
	txn = NULL;

	/* Reopen as a fresh writer so rebalance also exercises copy-on-write of
	 * clean path, sibling, and ancestor pages. */
	CHECK(mdb_txn_begin(env, NULL, 0, &txn) == MDB_SUCCESS);
	for (i = 3000; i-- > 1;) {
		MDB_val key, data;
		mdb_size_t leaf_before, branch_before;
		uint16_t depth_before;
		int will = 0;

		snprintf(keybuf, sizeof(keybuf), "d%06u", i);
		key.mv_data = keybuf; key.mv_size = strlen(keybuf);
		data.mv_data = value; data.mv_size = sizeof(value);
		CHECK(delete_will_rebalance(txn, dbi, &key, &will) == MDB_SUCCESS);
		leaf_before = txn->mt_dbs[dbi].md_leaf_pages;
		branch_before = txn->mt_dbs[dbi].md_branch_pages;
		depth_before = txn->mt_dbs[dbi].md_depth;
		CHECK(mdb_del(txn, dbi, &key, NULL) == MDB_SUCCESS);
		CHECK(sub_record(&txn->mt_dbs[dbi], &key, &data, &expected) == MDB_SUCCESS);
		CHECK(check_db_total(txn, dbi, &expected) == 0);

		if (will && txn->mt_dbs[dbi].md_leaf_pages == leaf_before)
			saw_move = 1;
		if (txn->mt_dbs[dbi].md_leaf_pages < leaf_before)
			saw_leaf_merge = 1;
		if (txn->mt_dbs[dbi].md_branch_pages < branch_before)
			saw_branch_merge = 1;
		if (txn->mt_dbs[dbi].md_depth < depth_before)
			saw_collapse = 1;

		if (will || txn->mt_dbs[dbi].md_leaf_pages != leaf_before ||
			txn->mt_dbs[dbi].md_branch_pages != branch_before ||
			txn->mt_dbs[dbi].md_depth != depth_before || (i % 97u) == 0)
			CHECK(verify_tree(txn, dbi, &expected) == MDB_SUCCESS);
		CHECK(!(txn->mt_flags & MDB_TXN_ERROR));
	}

	CHECK(saw_move);
	CHECK(saw_leaf_merge);
	CHECK(saw_branch_merge);
	CHECK(saw_collapse);
	CHECK(txn->mt_dbs[dbi].md_entries == 1);
	CHECK(txn->mt_dbs[dbi].md_depth == 1);
	CHECK(verify_tree(txn, dbi, &expected) == MDB_SUCCESS);
	CHECK(mdb_txn_commit(txn) == MDB_SUCCESS);
	txn = NULL;

	CHECK(mdb_txn_begin(env, NULL, 0, &txn) == MDB_SUCCESS);
	{
		MDB_val key, data;
		snprintf(keybuf, sizeof(keybuf), "d%06u", 0u);
		key.mv_data = keybuf; key.mv_size = strlen(keybuf);
		data.mv_data = value; data.mv_size = sizeof(value);
		CHECK(mdb_del(txn, dbi, &key, NULL) == MDB_SUCCESS);
		CHECK(sub_record(&txn->mt_dbs[dbi], &key, &data, &expected) == MDB_SUCCESS);
	}
	CHECK(txn->mt_dbs[dbi].md_root == P_INVALID);
	CHECK(txn->mt_dbs[dbi].md_depth == 0);
	CHECK(txn->mt_dbs[dbi].md_entries == 0);
	CHECK(check_db_total(txn, dbi, &expected) == 0);
	CHECK(verify_tree(txn, dbi, &expected) == MDB_SUCCESS);
	CHECK(mdb_txn_commit(txn) == MDB_SUCCESS);
	txn = NULL;

	CHECK(mdb_txn_begin(env, NULL, MDB_RDONLY, &txn) == MDB_SUCCESS);
	CHECK(check_db_total(txn, dbi, &expected) == 0);
	CHECK(txn->mt_dbs[dbi].md_root == P_INVALID);
	mdb_txn_abort(txn); txn = NULL;
	rc = 0;
	cleanup_env(env, txn, dir);
	return rc;
}


static int
test_left_edge_rebalance(void)
{
	const uint16_t agg = MDB_AGG_ENTRIES|MDB_AGG_KEYS|MDB_AGG_HASHSUM;
	char dir[] = "/tmp/aelmdb-rebalance-left-XXXXXX";
	MDB_env *env = NULL;
	MDB_txn *txn = NULL;
	MDB_dbi dbi;
	MDB_aggval expected;
	uint8_t value[MDB_HASH_SIZE+160];
	char keybuf[32];
	unsigned i;
	int saw_move = 0, saw_merge = 0, rc = 1;

	CHECK(mkdtemp(dir) != NULL);
	fill_bytes(value, sizeof(value), 83);
	CHECK(mdb_env_create(&env) == MDB_SUCCESS);
	CHECK(mdb_env_set_mapsize(env, 24u * 1024u * 1024u) == MDB_SUCCESS);
	CHECK(mdb_env_open(env, dir, 0, 0600) == MDB_SUCCESS);
	CHECK(mdb_txn_begin(env, NULL, 0, &txn) == MDB_SUCCESS);
	CHECK(mdb_dbi_open(txn, NULL, agg, &dbi) == MDB_SUCCESS);
	CHECK(mdb_set_hash_offset(txn, dbi, 1) == MDB_SUCCESS);
	mdb_aggval_zero(&expected);

	for (i = 0; i < 700; ++i) {
		MDB_val key, data;
		snprintf(keybuf, sizeof(keybuf), "l%06u", i);
		key.mv_data = keybuf; key.mv_size = strlen(keybuf);
		data.mv_data = value; data.mv_size = sizeof(value);
		CHECK(mdb_put(txn, dbi, &key, &data, MDB_APPEND) == MDB_SUCCESS);
		CHECK(add_record(&txn->mt_dbs[dbi], &key, &data, &expected) == MDB_SUCCESS);
	}
	CHECK(txn->mt_dbs[dbi].md_depth >= 2);
	CHECK(mdb_txn_commit(txn) == MDB_SUCCESS);
	txn = NULL;
	CHECK(mdb_txn_begin(env, NULL, 0, &txn) == MDB_SUCCESS);

	/* Delete from the left edge so rebalance must use the right sibling
	 * (fromleft == 0). This complements the descending/right-edge stress. */
	for (i = 0; i < 699; ++i) {
		MDB_val key, data;
		mdb_size_t leaf_before;
		int will = 0;
		snprintf(keybuf, sizeof(keybuf), "l%06u", i);
		key.mv_data = keybuf; key.mv_size = strlen(keybuf);
		data.mv_data = value; data.mv_size = sizeof(value);
		CHECK(delete_will_rebalance(txn, dbi, &key, &will) == MDB_SUCCESS);
		leaf_before = txn->mt_dbs[dbi].md_leaf_pages;
		CHECK(mdb_del(txn, dbi, &key, NULL) == MDB_SUCCESS);
		CHECK(sub_record(&txn->mt_dbs[dbi], &key, &data, &expected) == MDB_SUCCESS);
		if (will && txn->mt_dbs[dbi].md_leaf_pages == leaf_before)
			saw_move = 1;
		if (txn->mt_dbs[dbi].md_leaf_pages < leaf_before)
			saw_merge = 1;
		if (will || txn->mt_dbs[dbi].md_leaf_pages != leaf_before || (i % 83u) == 0)
			CHECK(verify_tree(txn, dbi, &expected) == MDB_SUCCESS);
		CHECK(check_db_total(txn, dbi, &expected) == 0);
	}
	CHECK(saw_move);
	CHECK(saw_merge);
	CHECK(txn->mt_dbs[dbi].md_entries == 1);
	CHECK(verify_tree(txn, dbi, &expected) == MDB_SUCCESS);
	rc = 0;
	cleanup_env(env, txn, dir);
	return rc;
}

int
main(void)
{
	if (test_delete_rebalance()) return 1;
	if (test_left_edge_rebalance()) return 1;
	printf("aggregate delete/rebalance tests passed (MDB_HASH_SIZE=%d)\n", MDB_HASH_SIZE);
	return 0;
}
