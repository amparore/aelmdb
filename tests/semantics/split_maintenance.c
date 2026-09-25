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
test_root_leaf_maintenance(void)
{
	const uint16_t agg = MDB_AGG_ENTRIES|MDB_AGG_KEYS|MDB_AGG_HASHSUM;
	char dir[] = "/tmp/aelmdb-split-root-XXXXXX";
	MDB_env *env = NULL;
	MDB_txn *txn = NULL;
	MDB_dbi dbi;
	MDB_val k1, k2, d1, d2, d2b;
	MDB_aggval expected, rootagg;
	MDB_cursor mc;
	MDB_xcursor mx;
	uint8_t v1[MDB_HASH_SIZE+8], v2[MDB_HASH_SIZE+8], v2b[MDB_HASH_SIZE+8];
	const char s1[] = "alpha", s2[] = "beta";
	int rc = 1;

	CHECK(mkdtemp(dir) != NULL);
	fill_bytes(v1, sizeof(v1), 3);
	fill_bytes(v2, sizeof(v2), 41);
	fill_bytes(v2b, sizeof(v2b), 113);
	CHECK(mdb_env_create(&env) == MDB_SUCCESS);
	CHECK(mdb_env_set_mapsize(env, 4u * 1024u * 1024u) == MDB_SUCCESS);
	CHECK(mdb_env_open(env, dir, 0, 0600) == MDB_SUCCESS);
	CHECK(mdb_txn_begin(env, NULL, 0, &txn) == MDB_SUCCESS);
	CHECK(mdb_dbi_open(txn, NULL, agg, &dbi) == MDB_SUCCESS);
	CHECK(mdb_set_hash_offset(txn, dbi, 2) == MDB_SUCCESS);
	k1.mv_data = (void *)s1; k1.mv_size = sizeof(s1)-1;
	k2.mv_data = (void *)s2; k2.mv_size = sizeof(s2)-1;
	d1.mv_data = v1; d1.mv_size = sizeof(v1);
	d2.mv_data = v2; d2.mv_size = sizeof(v2);
	d2b.mv_data = v2b; d2b.mv_size = sizeof(v2b);
	mdb_aggval_zero(&expected);

	CHECK(mdb_put(txn, dbi, &k1, &d1, 0) == MDB_SUCCESS);
	CHECK(add_record(&txn->mt_dbs[dbi], &k1, &d1, &expected) == MDB_SUCCESS);
	CHECK(check_db_total(txn, dbi, &expected) == 0);
	CHECK(mdb_put(txn, dbi, &k2, &d2, 0) == MDB_SUCCESS);
	CHECK(add_record(&txn->mt_dbs[dbi], &k2, &d2, &expected) == MDB_SUCCESS);
	CHECK(check_db_total(txn, dbi, &expected) == 0);

	CHECK(mdb_put(txn, dbi, &k2, &d2b, 0) == MDB_SUCCESS);
	CHECK(sub_record(&txn->mt_dbs[dbi], &k2, &d2, &expected) == MDB_SUCCESS);
	CHECK(add_record(&txn->mt_dbs[dbi], &k2, &d2b, &expected) == MDB_SUCCESS);
	CHECK(check_db_total(txn, dbi, &expected) == 0);

	CHECK(mdb_del(txn, dbi, &k1, NULL) == MDB_SUCCESS);
	CHECK(sub_record(&txn->mt_dbs[dbi], &k1, &d1, &expected) == MDB_SUCCESS);
	CHECK(check_db_total(txn, dbi, &expected) == 0);
	mdb_cursor_init(&mc, txn, dbi, &mx);
	CHECK(mdb_page_get(&mc, txn->mt_dbs[dbi].md_root, &mc.mc_pg[0], NULL) == MDB_SUCCESS);
	CHECK(mdb_page_agg_local(&mc, mc.mc_pg[0], &rootagg) == MDB_SUCCESS);
	CHECK(agg_equal(agg, &rootagg, &expected));

	CHECK(mdb_txn_commit(txn) == MDB_SUCCESS);
	txn = NULL;
	CHECK(mdb_txn_begin(env, NULL, MDB_RDONLY, &txn) == MDB_SUCCESS);
	CHECK(check_db_total(txn, dbi, &expected) == 0);
	mdb_txn_abort(txn); txn = NULL;

	CHECK(mdb_txn_begin(env, NULL, 0, &txn) == MDB_SUCCESS);
	CHECK(mdb_del(txn, dbi, &k2, NULL) == MDB_SUCCESS);
	CHECK(sub_record(&txn->mt_dbs[dbi], &k2, &d2b, &expected) == MDB_SUCCESS);
	CHECK(check_db_total(txn, dbi, &expected) == 0);
	CHECK(txn->mt_dbs[dbi].md_root == P_INVALID);
	CHECK(txn->mt_dbs[dbi].md_depth == 0);
	CHECK(mdb_txn_commit(txn) == MDB_SUCCESS);
	txn = NULL;
	rc = 0;
	cleanup_env(env, txn, dir);
	return rc;
}

static int
test_split_publication(void)
{
	const uint16_t agg = MDB_AGG_ENTRIES|MDB_AGG_KEYS|MDB_AGG_HASHSUM;
	char dir[] = "/tmp/aelmdb-split-split-XXXXXX";
	MDB_env *env = NULL;
	MDB_txn *txn = NULL;
	MDB_dbi dbi;
	MDB_cursor mc;
	MDB_xcursor mx;
	MDB_aggval expected, verified;
	uint8_t value[MDB_HASH_SIZE+160];
	char keybuf[32];
	unsigned i;
	int rc = 1;

	CHECK(mkdtemp(dir) != NULL);
	fill_bytes(value, sizeof(value), 17);
	CHECK(mdb_env_create(&env) == MDB_SUCCESS);
	CHECK(mdb_env_set_mapsize(env, 32u * 1024u * 1024u) == MDB_SUCCESS);
	CHECK(mdb_env_open(env, dir, 0, 0600) == MDB_SUCCESS);
	CHECK(mdb_txn_begin(env, NULL, 0, &txn) == MDB_SUCCESS);
	CHECK(mdb_dbi_open(txn, NULL, agg, &dbi) == MDB_SUCCESS);
	CHECK(mdb_set_hash_offset(txn, dbi, 0) == MDB_SUCCESS);
	mdb_aggval_zero(&expected);

	for (i = 0; i < 3000; ++i) {
		MDB_val key, data;
		snprintf(keybuf, sizeof(keybuf), "k%06u", i);
		key.mv_data = keybuf; key.mv_size = strlen(keybuf);
		data.mv_data = value; data.mv_size = sizeof(value);
		CHECK(mdb_put(txn, dbi, &key, &data, MDB_APPEND) == MDB_SUCCESS);
		CHECK(add_record(&txn->mt_dbs[dbi], &key, &data, &expected) == MDB_SUCCESS);
		if ((i & 127u) == 127u) {
			mdb_cursor_init(&mc, txn, dbi, &mx);
			CHECK(verify_subtree(&mc, txn->mt_dbs[dbi].md_root, &verified) == MDB_SUCCESS);
			CHECK(agg_equal(agg, &verified, &expected));
			CHECK(check_db_total(txn, dbi, &expected) == 0);
		}
	}
	CHECK(txn->mt_dbs[dbi].md_depth >= 3);
	mdb_cursor_init(&mc, txn, dbi, &mx);
	CHECK(verify_subtree(&mc, txn->mt_dbs[dbi].md_root, &verified) == MDB_SUCCESS);
	CHECK(agg_equal(agg, &verified, &expected));
	CHECK(check_db_total(txn, dbi, &expected) == 0);
	CHECK(!(txn->mt_flags & MDB_TXN_ERROR));

	CHECK(mdb_txn_commit(txn) == MDB_SUCCESS);
	txn = NULL;
	CHECK(mdb_txn_begin(env, NULL, MDB_RDONLY, &txn) == MDB_SUCCESS);
	CHECK(check_db_total(txn, dbi, &expected) == 0);
	mdb_cursor_init(&mc, txn, dbi, &mx);
	CHECK(verify_subtree(&mc, txn->mt_dbs[dbi].md_root, &verified) == MDB_SUCCESS);
	CHECK(agg_equal(agg, &verified, &expected));

	/* Start a fresh writer so every path page is clean and must be touched.
	 * Copy-on-write changes physical page numbers but not B+tree structure;
	 * aggregate maintenance must classify this as an ordinary logical update, not a split. */
	mdb_txn_abort(txn); txn = NULL;
	CHECK(mdb_txn_begin(env, NULL, 0, &txn) == MDB_SUCCESS);
	{
		MDB_val key, olddata, newdata;
		uint8_t replv[MDB_HASH_SIZE+160];
		fill_bytes(replv, sizeof(replv), 201);
		snprintf(keybuf, sizeof(keybuf), "k%06u", 1500u);
		key.mv_data = keybuf; key.mv_size = strlen(keybuf);
		olddata.mv_data = value; olddata.mv_size = sizeof(value);
		newdata.mv_data = replv; newdata.mv_size = sizeof(replv);
		CHECK(mdb_put(txn, dbi, &key, &newdata, 0) == MDB_SUCCESS);
		CHECK(sub_record(&txn->mt_dbs[dbi], &key, &olddata, &expected) == MDB_SUCCESS);
		CHECK(add_record(&txn->mt_dbs[dbi], &key, &newdata, &expected) == MDB_SUCCESS);
	}
	mdb_cursor_init(&mc, txn, dbi, &mx);
	CHECK(verify_subtree(&mc, txn->mt_dbs[dbi].md_root, &verified) == MDB_SUCCESS);
	CHECK(agg_equal(agg, &verified, &expected));
	CHECK(check_db_total(txn, dbi, &expected) == 0);
	rc = 0;
	cleanup_env(env, txn, dir);
	return rc;
}

static int
test_replacement_split(void)
{
	const uint16_t agg = MDB_AGG_ENTRIES|MDB_AGG_KEYS|MDB_AGG_HASHSUM;
	char dir[] = "/tmp/aelmdb-split-repl-XXXXXX";
	MDB_env *env = NULL;
	MDB_txn *txn = NULL;
	MDB_dbi dbi;
	MDB_cursor mc;
	MDB_xcursor mx;
	MDB_aggval expected, verified;
	uint8_t small[MDB_HASH_SIZE+24];
	uint8_t *big = NULL;
	char keybuf[32], chosen[32];
	size_t bigsz;
	unsigned i, chosen_i = 0;
	uint16_t depth_before;
	int found = 0, rc = 1;

	CHECK(mkdtemp(dir) != NULL);
	fill_bytes(small, sizeof(small), 31);
	CHECK(mdb_env_create(&env) == MDB_SUCCESS);
	CHECK(mdb_env_set_mapsize(env, 16u * 1024u * 1024u) == MDB_SUCCESS);
	CHECK(mdb_env_open(env, dir, 0, 0600) == MDB_SUCCESS);
	CHECK(mdb_txn_begin(env, NULL, 0, &txn) == MDB_SUCCESS);
	CHECK(mdb_dbi_open(txn, NULL, agg, &dbi) == MDB_SUCCESS);
	CHECK(mdb_set_hash_offset(txn, dbi, 0) == MDB_SUCCESS);
	mdb_aggval_zero(&expected);

	for (i = 0; i < 500; ++i) {
		MDB_val key, data;
		snprintf(keybuf, sizeof(keybuf), "r%06u", i);
		key.mv_data = keybuf; key.mv_size = strlen(keybuf);
		data.mv_data = small; data.mv_size = sizeof(small);
		CHECK(mdb_put(txn, dbi, &key, &data, 0) == MDB_SUCCESS);
		CHECK(add_record(&txn->mt_dbs[dbi], &key, &data, &expected) == MDB_SUCCESS);
	}
	CHECK(txn->mt_dbs[dbi].md_depth >= 2);
	depth_before = txn->mt_dbs[dbi].md_depth;

	/* Pick an inline value size that is large enough to force replacement of
	 * at least one existing leaf through the split path, but remains below the
	 * overflow threshold so the leaf footprint actually grows. */
	bigsz = txn->mt_env->me_nodemax > 256 ? txn->mt_env->me_nodemax - 128 : 128;
	if (bigsz < MDB_HASH_SIZE)
		bigsz = MDB_HASH_SIZE;
	big = malloc(bigsz);
	CHECK(big != NULL);
	fill_bytes(big, bigsz, 109);

	for (i = 20; i < 480; ++i) {
		MDB_val key, probe;
		MDB_page *mp;
		MDB_node *node;
		size_t avail, need;
		int exact = 0;

		snprintf(keybuf, sizeof(keybuf), "r%06u", i);
		key.mv_data = keybuf; key.mv_size = strlen(keybuf);
		mdb_cursor_init(&mc, txn, dbi, &mx);
		CHECK(mdb_page_search(&mc, &key, 0) == MDB_SUCCESS);
		node = mdb_node_search(&mc, &key, &exact);
		CHECK(node && exact);
		mp = mc.mc_pg[mc.mc_top];
		probe.mv_data = big; probe.mv_size = bigsz;
		avail = (size_t)SIZELEFT(mp) + mdb_node_footprint(mp, node);
		need = mdb_leaf_size(txn->mt_env, &key, &probe);
		if (avail < need) {
			chosen_i = i;
			strcpy(chosen, keybuf);
			found = 1;
			break;
		}
	}
	CHECK(found);
	{
		MDB_val key, olddata, newdata;
		key.mv_data = chosen; key.mv_size = strlen(chosen);
		olddata.mv_data = small; olddata.mv_size = sizeof(small);
		newdata.mv_data = big; newdata.mv_size = bigsz;
		CHECK(mdb_put(txn, dbi, &key, &newdata, 0) == MDB_SUCCESS);
		CHECK(sub_record(&txn->mt_dbs[dbi], &key, &olddata, &expected) == MDB_SUCCESS);
		CHECK(add_record(&txn->mt_dbs[dbi], &key, &newdata, &expected) == MDB_SUCCESS);
	}
	(void)chosen_i;
	CHECK(txn->mt_dbs[dbi].md_depth >= depth_before);
	mdb_cursor_init(&mc, txn, dbi, &mx);
	CHECK(verify_subtree(&mc, txn->mt_dbs[dbi].md_root, &verified) == MDB_SUCCESS);
	CHECK(agg_equal(agg, &verified, &expected));
	CHECK(check_db_total(txn, dbi, &expected) == 0);
	CHECK(!(txn->mt_flags & MDB_TXN_ERROR));

	free(big); big = NULL;
	rc = 0;
	cleanup_env(env, txn, dir);
	return rc;
}

static int
test_key_source_reserve(void)
{
	const uint16_t agg = MDB_AGG_ENTRIES|MDB_AGG_KEYS|MDB_AGG_HASHSUM;
	char dir[] = "/tmp/aelmdb-split-reserve-XXXXXX";
	MDB_env *env = NULL;
	MDB_txn *txn = NULL;
	MDB_dbi dbi;
	MDB_val key, data;
	MDB_aggval expected;
	uint8_t keybytes[MDB_HASH_SIZE+4];
	int rc = 1;

	CHECK(mkdtemp(dir) != NULL);
	fill_bytes(keybytes, sizeof(keybytes), 23);
	CHECK(mdb_env_create(&env) == MDB_SUCCESS);
	CHECK(mdb_env_set_mapsize(env, 4u * 1024u * 1024u) == MDB_SUCCESS);
	CHECK(mdb_env_open(env, dir, 0, 0600) == MDB_SUCCESS);
	CHECK(mdb_txn_begin(env, NULL, 0, &txn) == MDB_SUCCESS);
	CHECK(mdb_dbi_open(txn, NULL, agg|MDB_AGG_HASHSOURCE_FROM_KEY, &dbi) == MDB_SUCCESS);
	CHECK(mdb_set_hash_offset(txn, dbi, 1) == MDB_SUCCESS);
	key.mv_data = keybytes; key.mv_size = sizeof(keybytes);
	data.mv_data = NULL; data.mv_size = 64;
	mdb_aggval_zero(&expected);
	CHECK(mdb_put(txn, dbi, &key, &data, MDB_RESERVE) == MDB_SUCCESS);
	CHECK(data.mv_data != NULL);
	memset(data.mv_data, 0xa5, data.mv_size);
	CHECK(add_record(&txn->mt_dbs[dbi], &key, &data, &expected) == MDB_SUCCESS);
	CHECK(check_db_total(txn, dbi, &expected) == 0);
	rc = 0;
	cleanup_env(env, txn, dir);
	return rc;
}

static int
test_branch_path_maintenance(void)
{
	const uint16_t agg = MDB_AGG_ENTRIES|MDB_AGG_KEYS|MDB_AGG_HASHSUM;
	char dir[] = "/tmp/aelmdb-split-branch-XXXXXX";
	MDB_env *env = NULL;
	MDB_txn *txn = NULL;
	MDB_dbi dbi;
	MDB_cursor mc;
	MDB_xcursor mx;
	MDB_aggval expected, verified;
	uint8_t value[MDB_HASH_SIZE+48], repl[MDB_HASH_SIZE+48], rein[MDB_HASH_SIZE+48];
	char keybuf[32];
	unsigned i, del_index = 0;
	pgno_t root_before;
	uint16_t depth_before;
	int deleted = 0, rc = 1;

	CHECK(mkdtemp(dir) != NULL);
	fill_bytes(value, sizeof(value), 5);
	fill_bytes(repl, sizeof(repl), 101);
	fill_bytes(rein, sizeof(rein), 177);
	CHECK(mdb_env_create(&env) == MDB_SUCCESS);
	CHECK(mdb_env_set_mapsize(env, 16u * 1024u * 1024u) == MDB_SUCCESS);
	CHECK(mdb_env_open(env, dir, 0, 0600) == MDB_SUCCESS);
	CHECK(mdb_txn_begin(env, NULL, 0, &txn) == MDB_SUCCESS);
	CHECK(mdb_dbi_open(txn, NULL, agg, &dbi) == MDB_SUCCESS);
	CHECK(mdb_set_hash_offset(txn, dbi, 3) == MDB_SUCCESS);
	mdb_aggval_zero(&expected);

	/* Build a valid branch tree entirely through aggregate-aware public puts. */
	for (i = 0; i < 600; ++i) {
		MDB_val key, data;
		snprintf(keybuf, sizeof(keybuf), "k%06u", i);
		key.mv_data = keybuf; key.mv_size = strlen(keybuf);
		data.mv_data = value; data.mv_size = sizeof(value);
		CHECK(mdb_put(txn, dbi, &key, &data, 0) == MDB_SUCCESS);
		CHECK(add_record(&txn->mt_dbs[dbi], &key, &data, &expected) == MDB_SUCCESS);
	}
	CHECK(txn->mt_dbs[dbi].md_depth >= 2);
	mdb_cursor_init(&mc, txn, dbi, &mx);
	root_before = txn->mt_dbs[dbi].md_root;
	depth_before = txn->mt_dbs[dbi].md_depth;
	CHECK(verify_subtree(&mc, root_before, &verified) == MDB_SUCCESS);
	CHECK(agg_equal(agg, &verified, &expected));

	/* Same-size replacement exercises the complete O(height) path. */
	{
		MDB_val key, olddata, newdata;
		snprintf(keybuf, sizeof(keybuf), "k%06u", 300u);
		key.mv_data = keybuf; key.mv_size = strlen(keybuf);
		olddata.mv_data = value; olddata.mv_size = sizeof(value);
		newdata.mv_data = repl; newdata.mv_size = sizeof(repl);
		CHECK(mdb_put(txn, dbi, &key, &newdata, 0) == MDB_SUCCESS);
		CHECK(sub_record(&txn->mt_dbs[dbi], &key, &olddata, &expected) == MDB_SUCCESS);
		CHECK(add_record(&txn->mt_dbs[dbi], &key, &newdata, &expected) == MDB_SUCCESS);
	}
	CHECK(txn->mt_dbs[dbi].md_root == root_before);
	CHECK(txn->mt_dbs[dbi].md_depth == depth_before);
	mdb_cursor_init(&mc, txn, dbi, &mx);
	CHECK(verify_subtree(&mc, root_before, &verified) == MDB_SUCCESS);
	CHECK(agg_equal(agg, &verified, &expected));
	CHECK(check_db_total(txn, dbi, &expected) == 0);

	/* Find a leaf where one deletion stays above the rebalance threshold. */
	for (i = 50; i < 550; ++i) {
		MDB_val key;
		int drc;
		if (i == 300)
			continue;
		snprintf(keybuf, sizeof(keybuf), "k%06u", i);
		key.mv_data = keybuf; key.mv_size = strlen(keybuf);
		drc = mdb_del(txn, dbi, &key, NULL);
		if (drc == MDB_INCOMPATIBLE)
			continue;
		CHECK(drc == MDB_SUCCESS);
		del_index = i;
		deleted = 1;
		break;
	}
	CHECK(deleted);
	{
		MDB_val key, olddata;
		snprintf(keybuf, sizeof(keybuf), "k%06u", del_index);
		key.mv_data = keybuf; key.mv_size = strlen(keybuf);
		olddata.mv_data = value; olddata.mv_size = sizeof(value);
		CHECK(sub_record(&txn->mt_dbs[dbi], &key, &olddata, &expected) == MDB_SUCCESS);
	}
	mdb_cursor_init(&mc, txn, dbi, &mx);
	CHECK(verify_subtree(&mc, root_before, &verified) == MDB_SUCCESS);
	CHECK(agg_equal(agg, &verified, &expected));
	CHECK(check_db_total(txn, dbi, &expected) == 0);

	/* Reinsertion into the just-freed leaf tests non-structural insert with
	 * ancestor propagation. */
	{
		MDB_val key, data;
		snprintf(keybuf, sizeof(keybuf), "k%06u", del_index);
		key.mv_data = keybuf; key.mv_size = strlen(keybuf);
		data.mv_data = rein; data.mv_size = sizeof(rein);
		CHECK(mdb_put(txn, dbi, &key, &data, 0) == MDB_SUCCESS);
		CHECK(add_record(&txn->mt_dbs[dbi], &key, &data, &expected) == MDB_SUCCESS);
	}
	CHECK(txn->mt_dbs[dbi].md_root == root_before);
	CHECK(txn->mt_dbs[dbi].md_depth == depth_before);
	mdb_cursor_init(&mc, txn, dbi, &mx);
	CHECK(verify_subtree(&mc, root_before, &verified) == MDB_SUCCESS);
	CHECK(agg_equal(agg, &verified, &expected));
	CHECK(check_db_total(txn, dbi, &expected) == 0);

	CHECK(mdb_txn_commit(txn) == MDB_SUCCESS);
	txn = NULL;
	CHECK(mdb_txn_begin(env, NULL, MDB_RDONLY, &txn) == MDB_SUCCESS);
	CHECK(check_db_total(txn, dbi, &expected) == 0);
	mdb_cursor_init(&mc, txn, dbi, &mx);
	CHECK(verify_subtree(&mc, txn->mt_dbs[dbi].md_root, &verified) == MDB_SUCCESS);
	CHECK(agg_equal(agg, &verified, &expected));
	rc = 0;
	cleanup_env(env, txn, dir);
	return rc;
}

int
main(void)
{
	if (test_root_leaf_maintenance()) return 1;
	if (test_split_publication()) return 1;
	if (test_replacement_split()) return 1;
	if (test_key_source_reserve()) return 1;
	if (test_branch_path_maintenance()) return 1;
	printf("aggregate split-maintenance tests passed (MDB_HASH_SIZE=%d)\n", MDB_HASH_SIZE);
	return 0;
}
