#ifndef _GNU_SOURCE
#define _GNU_SOURCE 1
#endif

/* Reuse the independent DUPSORT maintenance oracle and regression body. */
#define main dupsort_regression_main
#include "test_dupsort_maintenance.c"
#undef main

#define INTEGRATION_VALUE_SIZE ((MDB_HASH_SIZE + 8 > 160) ? MDB_HASH_SIZE + 8 : 160)

static int
is_zero_hash(const uint8_t h[MDB_HASH_SIZE])
{
    size_t i;
    for (i = 0; i < MDB_HASH_SIZE; ++i) if (h[i]) return 0;
    return 1;
}


static void
c1_make_key(unsigned char key[8], unsigned int x)
{
    unsigned int i;
    for (i = 0; i < 8; ++i)
        key[7-i] = (unsigned char)(((uint64_t)x) >> (8*i));
}

/* C1 regression: deleting the last item of a sufficiently full leaf does not
 * rebalance the tree, but LMDB legitimately moves the cursor to the next leaf.
 * Aggregate publication must still update the original branch path. */
static int
test_delete_cursor_relocation(void)
{
    const uint16_t schema = MDB_AGG_ENTRIES|MDB_AGG_KEYS|MDB_AGG_HASHSUM;
    char dir[] = "/tmp/aelmdb-c1-relocate-XXXXXX";
    MDB_env *env = NULL;
    MDB_txn *txn = NULL;
    MDB_cursor *cur = NULL;
    MDB_dbi dbi;
    MDB_aggval expected;
    unsigned char kb[8], value[INTEGRATION_VALUE_SIZE];
    MDB_val key = { sizeof(kb), kb }, data = { sizeof(value), value };
    unsigned int i;
    int rc, found = 0;

    CHECK(mkdtemp(dir) != NULL);
    CHECK(mdb_env_create(&env) == MDB_SUCCESS);
    CHECK(mdb_env_set_mapsize(env, 64u*1024u*1024u) == MDB_SUCCESS);
    CHECK(mdb_env_open(env, dir, 0, 0600) == MDB_SUCCESS);
    CHECK(mdb_txn_begin(env, NULL, 0, &txn) == MDB_SUCCESS);
    CHECK(mdb_dbi_open(txn, NULL, schema, &dbi) == MDB_SUCCESS);
    CHECK(mdb_set_hash_offset(txn, dbi, 0) == MDB_SUCCESS);
    mdb_aggval_zero(&expected);
    for (i = 0; i < 900; ++i) {
        c1_make_key(kb, i);
        make_value(value, sizeof(value), i + 1, 73);
        CHECK(mdb_put(txn, dbi, &key, &data, 0) == MDB_SUCCESS);
        CHECK(expected_add_value(&txn->mt_dbs[dbi], &data, 1, &expected) == MDB_SUCCESS);
    }
    CHECK(txn->mt_dbs[dbi].md_depth >= 2);
    CHECK(mdb_txn_commit(txn) == MDB_SUCCESS); txn = NULL;

    /* Use a fresh writer so the path is copy-on-write touched before delete. */
    CHECK(mdb_txn_begin(env, NULL, 0, &txn) == MDB_SUCCESS);
    CHECK(mdb_cursor_open(txn, dbi, &cur) == MDB_SUCCESS);
    rc = mdb_cursor_get(cur, &key, &data, MDB_FIRST);
    CHECK(rc == MDB_SUCCESS);
    while (rc == MDB_SUCCESS && !found) {
        MDB_page *mp = cur->mc_pg[cur->mc_top];
        indx_t last = NUMKEYS(mp) ? NUMKEYS(mp) - 1 : 0;

        if (cur->mc_snum >= 2 && IS_LEAF(mp) && NUMKEYS(mp) >= 2) {
            MDB_node *node = NODEPTR(mp, last);
            size_t usable = txn->mt_env->me_psize - PAGEHDRSZ;
            size_t free_after = SIZELEFT(mp) + mdb_node_footprint(mp, node);
            unsigned int nkeys_after = NUMKEYS(mp) - 1;
            long fill_after = 1000L * (long)(usable - free_after) / (long)usable;
            MDB_cursor probe;
            MDB_val pk = {0}, pv = {0};

            mdb_cursor_copy(cur, &probe);
            probe.mc_xcursor = NULL;
            probe.mc_ki[probe.mc_top] = last;
            if (nkeys_after >= 1 && fill_after >= FILL_THRESHOLD &&
                mdb_cursor_get(&probe, &pk, &pv, MDB_NEXT) == MDB_SUCCESS) {
                pgno_t old_leaf = MP_PGNO(mp);
                unsigned int old_snum = cur->mc_snum;

                cur->mc_ki[cur->mc_top] = last;
                CHECK(mdb_cursor_get(cur, &key, &data, MDB_GET_CURRENT) == MDB_SUCCESS);
                CHECK(expected_sub_value(&txn->mt_dbs[dbi], &data, 1, &expected) == MDB_SUCCESS);
                CHECK(mdb_cursor_del(cur, 0) == MDB_SUCCESS);
                CHECK(cur->mc_snum == old_snum);
                CHECK(MP_PGNO(cur->mc_pg[cur->mc_top]) != old_leaf);
                CHECK(check_all(txn, dbi, &expected) == MDB_SUCCESS);
                found = 1;
                break;
            }
        }

        if (NUMKEYS(mp))
            cur->mc_ki[cur->mc_top] = NUMKEYS(mp) - 1;
        rc = mdb_cursor_get(cur, &key, &data, MDB_NEXT);
    }
    CHECK(found);
    mdb_cursor_close(cur); cur = NULL;
    CHECK(mdb_txn_commit(txn) == MDB_SUCCESS); txn = NULL;
    CHECK(mdb_txn_begin(env, NULL, MDB_RDONLY, &txn) == MDB_SUCCESS);
    CHECK(check_all(txn, dbi, &expected) == MDB_SUCCESS);
    cleanup_env(env, txn, dir);
    return 0;
}

/* C4 regression: a structural DUPSORT delete (leaf merge) whose deleted record
 * was the last one on the merged leaf.  mdb_cursor_del0() then relocates the
 * cursor to the next sibling leaf, which may belong to an untouched (clean)
 * subtree.  the rebalance path has already published exact aggregates on the mutated path; the
 * DUPSORT finish step must not re-fold along the relocated cursor stack. */
#define C4_KEY_SIZE 384

static void
c4_make_key(unsigned char key[C4_KEY_SIZE], unsigned int x)
{
    memset(key, 'k', C4_KEY_SIZE);
    c1_make_key(key, x);
}

static int
c4_delete_one(MDB_env *env, MDB_dbi dbi, unsigned int id,
    const MDB_aggval *base, int commit, int *hit)
{
    MDB_txn *txn = NULL;
    MDB_cursor *cur = NULL;
    MDB_aggval expected = *base;
    unsigned char kb[C4_KEY_SIZE], value[INTEGRATION_VALUE_SIZE];
    MDB_val key = { sizeof(kb), kb }, data = { sizeof(value), value };
    int rebalances;

    c4_make_key(kb, id);
    make_value(value, sizeof(value), id + 1, 41);
    CHECK(mdb_txn_begin(env, NULL, 0, &txn) == MDB_SUCCESS);
    CHECK(mdb_cursor_open(txn, dbi, &cur) == MDB_SUCCESS);
    CHECK(mdb_cursor_get(cur, &key, &data, MDB_GET_BOTH) == MDB_SUCCESS);
    {
        /* Same rebalance predicate as mdb_rebalance() for a leaf. */
        MDB_page *mp = cur->mc_pg[cur->mc_top];
        MDB_node *node = NODEPTR(mp, cur->mc_ki[cur->mc_top]);
        size_t usable = txn->mt_env->me_psize - PAGEHDRSZ;
        size_t free_after = SIZELEFT(mp) + mdb_node_footprint(mp, node);
        long fill_after = 1000L * (long)(usable - free_after) / (long)usable;
        rebalances = cur->mc_snum >= 2 &&
            (fill_after < FILL_THRESHOLD || NUMKEYS(mp) - 1 < 1);
    }
    CHECK(expected_sub_value(&txn->mt_dbs[dbi], &data, 1, &expected) == MDB_SUCCESS);
    CHECK(mdb_cursor_del(cur, 0) == MDB_SUCCESS);
    /* The exact failure precondition: the delete rebalanced (move or merge)
     * and the cursor now sits below a parent page this transaction never
     * touched. */
    if (rebalances &&
        (cur->mc_flags & C_INITIALIZED) && !(cur->mc_flags & C_EOF) &&
        cur->mc_snum >= 3 &&
        !(cur->mc_pg[cur->mc_top - 1]->mp_flags & P_DIRTY))
        *hit = 1;
    CHECK(check_all(txn, dbi, &expected) == MDB_SUCCESS);
#if defined(MDB_DEBUG_AGG_INTEGRITY) && MDB_DEBUG_AGG_INTEGRITY
    CHECK(mdb_agg_check_integrity(txn, dbi) == MDB_SUCCESS);
#endif
    mdb_cursor_close(cur);
    if (!commit) {
        mdb_txn_abort(txn);
        return 0;
    }
    CHECK(mdb_txn_commit(txn) == MDB_SUCCESS);
    CHECK(mdb_txn_begin(env, NULL, MDB_RDONLY, &txn) == MDB_SUCCESS);
    CHECK(check_all(txn, dbi, &expected) == MDB_SUCCESS);
    mdb_txn_abort(txn);
    return 0;
}

static int
test_dupsort_structural_delete_relocation(void)
{
    const uint16_t schema = MDB_AGG_ENTRIES|MDB_AGG_KEYS|MDB_AGG_HASHSUM;
    enum { N = 720 };
    char dir[] = "/tmp/aelmdb-c4-relocate-XXXXXX";
    MDB_env *env = NULL;
    MDB_txn *txn = NULL;
    MDB_dbi dbi;
    MDB_aggval base;
    unsigned char kb[C4_KEY_SIZE], value[INTEGRATION_VALUE_SIZE];
    MDB_val key = { sizeof(kb), kb }, data = { sizeof(value), value };
    unsigned int i, first_hit = N, hits = 0, live = 0;
    static unsigned char present[N];

    CHECK(mkdtemp(dir) != NULL);
    CHECK(mdb_env_create(&env) == MDB_SUCCESS);
    CHECK(mdb_env_set_mapsize(env, 64u*1024u*1024u) == MDB_SUCCESS);
    CHECK(mdb_env_open(env, dir, 0, 0600) == MDB_SUCCESS);
    CHECK((size_t)mdb_env_get_maxkeysize(env) >= C4_KEY_SIZE);
    CHECK(mdb_txn_begin(env, NULL, 0, &txn) == MDB_SUCCESS);
    CHECK(mdb_dbi_open(txn, NULL, MDB_DUPSORT|schema, &dbi) == MDB_SUCCESS);
    CHECK(mdb_set_hash_offset(txn, dbi, 0) == MDB_SUCCESS);
    mdb_aggval_zero(&base);
    for (i = 0; i < N; ++i) {
        c4_make_key(kb, i);
        make_value(value, sizeof(value), i + 1, 41);
        CHECK(mdb_put(txn, dbi, &key, &data, 0) == MDB_SUCCESS);
        CHECK(expected_add_value(&txn->mt_dbs[dbi], &data, 1, &base) == MDB_SUCCESS);
    }
    /* Thin the tree irregularly so many leaves sit just above the rebalance
     * threshold, while keeping it at least three levels deep. */
    for (i = 0; i < N; ++i) {
        present[i] = 1;
        if (i % 5 == 1 || i % 5 == 3 || i % 7 == 2) {
            c4_make_key(kb, i);
            make_value(value, sizeof(value), i + 1, 41);
            CHECK(expected_sub_value(&txn->mt_dbs[dbi], &data, 1, &base) == MDB_SUCCESS);
            CHECK(mdb_del(txn, dbi, &key, &data) == MDB_SUCCESS);
            present[i] = 0;
        } else {
            ++live;
        }
    }
    CHECK(txn->mt_dbs[dbi].md_depth >= 3);
    CHECK(txn->mt_dbs[dbi].md_entries == live);
    CHECK(check_all(txn, dbi, &base) == MDB_SUCCESS);
    CHECK(mdb_txn_commit(txn) == MDB_SUCCESS); txn = NULL;

    /* Each delete runs in a fresh writer against the same committed snapshot,
     * so every sibling outside the copy-on-write path is clean. */
    for (i = 0; i < N; ++i) {
        int hit = 0;
        if (!present[i])
            continue;
        CHECK(c4_delete_one(env, dbi, i, &base, 0, &hit) == 0);
        if (hit) {
            if (first_hit == N)
                first_hit = i;
            ++hits;
        }
    }
    CHECK(hits > 0);
    /* Commit one representative case and verify persistence. */
    {
        int hit = 0;
        CHECK(c4_delete_one(env, dbi, first_hit, &base, 1, &hit) == 0);
        CHECK(hit);
    }
    cleanup_env(env, txn, dir);
    return 0;
}

static int
test_reserve_contract(void)
{
    const uint16_t value_agg = MDB_AGG_ENTRIES|MDB_AGG_KEYS|MDB_AGG_HASHSUM;
    const uint16_t key_agg = value_agg|MDB_AGG_HASHSOURCE_FROM_KEY;
    char dir[] = "/tmp/aelmdb-m7-reserve-XXXXXX";
    MDB_env *env = NULL; MDB_txn *txn = NULL; MDB_dbi dbi;
    MDB_val key, data;
    MDB_aggval expected, one;
    unsigned char *kbuf = NULL;
    unsigned char value[INTEGRATION_VALUE_SIZE];
    size_t klen = MDB_HASH_SIZE + 8;

    CHECK(mkdtemp(dir) != NULL);
    CHECK(mdb_env_create(&env) == MDB_SUCCESS);
    CHECK(mdb_env_set_mapsize(env, 32u*1024u*1024u) == MDB_SUCCESS);
    CHECK(mdb_env_open(env, dir, 0, 0600) == MDB_SUCCESS);
    CHECK(mdb_txn_begin(env, NULL, 0, &txn) == MDB_SUCCESS);

    /* Value-source HASHSUM cannot observe bytes written after MDB_RESERVE returns. */
    CHECK(mdb_dbi_open(txn, NULL, value_agg, &dbi) == MDB_SUCCESS);
    CHECK(mdb_set_hash_offset(txn, dbi, 0) == MDB_SUCCESS);
    key.mv_data = (void *)"reserve-value"; key.mv_size = strlen((char *)key.mv_data);
    data.mv_data = NULL; data.mv_size = sizeof(value);
    CHECK(mdb_put(txn, dbi, &key, &data, MDB_RESERVE) == MDB_INCOMPATIBLE);
    CHECK(txn->mt_dbs[dbi].md_entries == 0);
    CHECK(txn->mt_dbs[dbi].md_keys == 0);
    CHECK(is_zero_hash(txn->mt_dbs[dbi].md_hashsum));
    make_value(value, sizeof(value), 1, 1);
    data.mv_data = value; data.mv_size = sizeof(value);
    CHECK(mdb_put(txn, dbi, &key, &data, 0) == MDB_SUCCESS);
    mdb_aggval_zero(&expected);
    CHECK(expected_add_value(&txn->mt_dbs[dbi], &data, 1, &expected) == MDB_SUCCESS);
    CHECK(check_all(txn, dbi, &expected) == MDB_SUCCESS);

    /* Empty the main DB and switch schema: key-source HASHSUM is compatible with RESERVE. */
    CHECK(mdb_drop(txn, dbi, 0) == MDB_SUCCESS);
    CHECK(txn->mt_dbs[dbi].md_entries == 0 && txn->mt_dbs[dbi].md_keys == 0);
    CHECK(is_zero_hash(txn->mt_dbs[dbi].md_hashsum));
    CHECK(mdb_dbi_open(txn, NULL, key_agg, &dbi) == MDB_SUCCESS);
    CHECK(mdb_set_hash_offset(txn, dbi, 0) == MDB_SUCCESS);
    kbuf = malloc(klen); CHECK(kbuf != NULL);
    make_value(kbuf, klen, 2, 3);
    key.mv_data = kbuf; key.mv_size = klen;
    data.mv_data = NULL; data.mv_size = sizeof(value);
    CHECK(mdb_put(txn, dbi, &key, &data, MDB_RESERVE) == MDB_SUCCESS);
    memset(data.mv_data, 0xa5, data.mv_size);
    mdb_aggval_zero(&expected);
    CHECK(mdb_agg_record_contribution(DB_AGGFLAGS(&txn->mt_dbs[dbi]),
        txn->mt_dbs[dbi].md_hash_offset, &key, &one) == MDB_SUCCESS);
    CHECK(mdb_aggval_add(DB_AGGFLAGS(&txn->mt_dbs[dbi]), &expected, &one) == MDB_SUCCESS);
    { MDB_aggval got; mdb_db_get_aggval(&txn->mt_dbs[dbi], &got);
      CHECK(agg_equal(DB_AGGFLAGS(&txn->mt_dbs[dbi]), &got, &expected)); }

    /* ENTRIES/KEYS-only maintenance has no byte-finalization issue. */
    CHECK(mdb_drop(txn, dbi, 0) == MDB_SUCCESS);
    CHECK(mdb_dbi_open(txn, NULL, MDB_AGG_ENTRIES|MDB_AGG_KEYS, &dbi) == MDB_SUCCESS);
    key.mv_data = (void *)"reserve-counts"; key.mv_size = strlen((char *)key.mv_data);
    data.mv_data = NULL; data.mv_size = sizeof(value);
    CHECK(mdb_put(txn, dbi, &key, &data, MDB_RESERVE) == MDB_SUCCESS);
    memset(data.mv_data, 0x5a, data.mv_size);
    mdb_aggval_zero(&expected); expected.entries = 1; expected.keys = 1;
    CHECK(check_all(txn, dbi, &expected) == MDB_SUCCESS);

    /* MDB_RESERVE is an LMDB-level DUPSORT incompatibility even when no
     * hashsum is enabled. */
    CHECK(mdb_drop(txn, dbi, 0) == MDB_SUCCESS);
    CHECK(mdb_dbi_open(txn, NULL, MDB_DUPSORT|MDB_AGG_ENTRIES|MDB_AGG_KEYS, &dbi) == MDB_SUCCESS);
    key.mv_data = (void *)"dup-reserve"; key.mv_size = strlen((char *)key.mv_data);
    data.mv_data = NULL; data.mv_size = sizeof(value);
    CHECK(mdb_put(txn, dbi, &key, &data, MDB_RESERVE) == MDB_INCOMPATIBLE);
    CHECK(txn->mt_dbs[dbi].md_entries == 0 && txn->mt_dbs[dbi].md_keys == 0);

    free(kbuf);
    cleanup_env(env, txn, dir);
    return 0;
}

static int
test_writemap_value_hash(void)
{
    const uint16_t agg = MDB_AGG_ENTRIES|MDB_AGG_KEYS|MDB_AGG_HASHSUM;
    char dir[] = "/tmp/aelmdb-m7-writemap-XXXXXX";
    MDB_env *env = NULL; MDB_txn *txn = NULL; MDB_dbi dbi;
    MDB_aggval expected;
    MDB_val key, val;
    unsigned char value[INTEGRATION_VALUE_SIZE];
    char keybuf[40];
    unsigned i;

    CHECK(mkdtemp(dir) != NULL);
    CHECK(mdb_env_create(&env) == MDB_SUCCESS);
    CHECK(mdb_env_set_mapsize(env, 64u*1024u*1024u) == MDB_SUCCESS);
    CHECK(mdb_env_open(env, dir, MDB_WRITEMAP, 0600) == MDB_SUCCESS);
    CHECK(mdb_txn_begin(env, NULL, 0, &txn) == MDB_SUCCESS);
    CHECK(mdb_dbi_open(txn, NULL, agg, &dbi) == MDB_SUCCESS);
    CHECK(mdb_set_hash_offset(txn, dbi, 3) == MDB_SUCCESS);
    mdb_aggval_zero(&expected);
    for (i = 0; i < 900; ++i) {
        snprintf(keybuf, sizeof(keybuf), "w%06u", i);
        key.mv_data = keybuf; key.mv_size = strlen(keybuf);
        make_value(value, sizeof(value), i, 9);
        val.mv_data = value; val.mv_size = sizeof(value);
        CHECK(mdb_put(txn, dbi, &key, &val, MDB_APPEND) == MDB_SUCCESS);
        CHECK(expected_add_value(&txn->mt_dbs[dbi], &val, 1, &expected) == MDB_SUCCESS);
    }
    CHECK(txn->mt_dbs[dbi].md_depth >= 2);
    CHECK(check_all(txn, dbi, &expected) == MDB_SUCCESS);
#if defined(MDB_DEBUG_AGG_INTEGRITY) && MDB_DEBUG_AGG_INTEGRITY
    CHECK(mdb_agg_check_integrity(txn, dbi) == MDB_SUCCESS);
#endif
    CHECK(mdb_txn_commit(txn) == MDB_SUCCESS); txn = NULL;
    CHECK(mdb_txn_begin(env, NULL, MDB_RDONLY, &txn) == MDB_SUCCESS);
    CHECK(check_all(txn, dbi, &expected) == MDB_SUCCESS);
    cleanup_env(env, txn, dir);
    return 0;
}

static int
test_writemap_dupsort(void)
{
    const uint16_t agg = MDB_AGG_ENTRIES|MDB_AGG_KEYS|MDB_AGG_HASHSUM;
    char dir[] = "/tmp/aelmdb-m7-wmdup-XXXXXX";
    MDB_env *env = NULL; MDB_txn *txn = NULL; MDB_dbi dbi;
    MDB_aggval expected;
    MDB_val key, val;
    unsigned char value[INTEGRATION_VALUE_SIZE];
    char keybuf[] = "dup";
    unsigned i;

    CHECK(mkdtemp(dir) != NULL);
    CHECK(mdb_env_create(&env) == MDB_SUCCESS);
    CHECK(mdb_env_set_mapsize(env, 64u*1024u*1024u) == MDB_SUCCESS);
    CHECK(mdb_env_open(env, dir, MDB_WRITEMAP, 0600) == MDB_SUCCESS);
    CHECK(mdb_txn_begin(env, NULL, 0, &txn) == MDB_SUCCESS);
    CHECK(mdb_dbi_open(txn, NULL, MDB_DUPSORT|agg, &dbi) == MDB_SUCCESS);
    CHECK(mdb_set_hash_offset(txn, dbi, 2) == MDB_SUCCESS);
    key.mv_data = keybuf; key.mv_size = strlen(keybuf);
    mdb_aggval_zero(&expected);
    for (i = 0; i < 180; ++i) {
        make_value(value, sizeof(value), i, 17);
        val.mv_data = value; val.mv_size = sizeof(value);
        CHECK(mdb_put(txn, dbi, &key, &val, 0) == MDB_SUCCESS);
        CHECK(expected_add_value(&txn->mt_dbs[dbi], &val, i == 0, &expected) == MDB_SUCCESS);
    }
    CHECK(check_all(txn, dbi, &expected) == MDB_SUCCESS);
#if defined(MDB_DEBUG_AGG_INTEGRITY) && MDB_DEBUG_AGG_INTEGRITY
    CHECK(mdb_agg_check_integrity(txn, dbi) == MDB_SUCCESS);
#endif
    val.mv_data = NULL; val.mv_size = sizeof(value);
    CHECK(mdb_put(txn, dbi, &key, &val, MDB_RESERVE) == MDB_INCOMPATIBLE);
    CHECK(check_all(txn, dbi, &expected) == MDB_SUCCESS);
    CHECK(mdb_txn_commit(txn) == MDB_SUCCESS); txn = NULL;
    CHECK(mdb_txn_begin(env, NULL, MDB_RDONLY, &txn) == MDB_SUCCESS);
    { MDB_dbi reopened;
      CHECK(mdb_dbi_open(txn, NULL, MDB_DUPSORT|agg, &reopened) == MDB_SUCCESS);
      CHECK(reopened == MAIN_DBI);
      dbi = reopened; }
    CHECK(check_all(txn, dbi, &expected) == MDB_SUCCESS);
#if defined(MDB_DEBUG_AGG_INTEGRITY) && MDB_DEBUG_AGG_INTEGRITY
    CHECK(mdb_agg_check_integrity(txn, dbi) == MDB_SUCCESS);
#endif
    cleanup_env(env, txn, dir);
    return 0;
}

static int
test_drop_and_catalog(void)
{
    const uint16_t agg = MDB_AGG_ENTRIES|MDB_AGG_KEYS|MDB_AGG_HASHSUM;
    char dir[] = "/tmp/aelmdb-m7-drop-XXXXXX";
    MDB_env *env = NULL; MDB_txn *txn = NULL; MDB_dbi main_dbi, named;
    MDB_val key, val;
    unsigned char value[INTEGRATION_VALUE_SIZE];
    unsigned af;
    int off;

    CHECK(mkdtemp(dir) != NULL);
    CHECK(mdb_env_create(&env) == MDB_SUCCESS);
    CHECK(mdb_env_set_mapsize(env, 64u*1024u*1024u) == MDB_SUCCESS);
    CHECK(mdb_env_set_maxdbs(env, 8) == MDB_SUCCESS);
    CHECK(mdb_env_open(env, dir, 0, 0600) == MDB_SUCCESS);
    CHECK(mdb_txn_begin(env, NULL, 0, &txn) == MDB_SUCCESS);
    CHECK(mdb_dbi_open(txn, NULL, agg, &main_dbi) == MDB_SUCCESS);
    CHECK(mdb_set_hash_offset(txn, main_dbi, 0) == MDB_SUCCESS);
    CHECK(mdb_dbi_open(txn, "named", MDB_CREATE|agg, &named) == MDB_SUCCESS);
    CHECK(mdb_set_hash_offset(txn, named, 4) == MDB_SUCCESS);
    key.mv_data = (void *)"k"; key.mv_size = 1;
    make_value(value, sizeof(value), 7, 2); val.mv_data = value; val.mv_size = sizeof(value);
    CHECK(mdb_put(txn, named, &key, &val, 0) == MDB_SUCCESS);
    CHECK(mdb_txn_commit(txn) == MDB_SUCCESS); txn = NULL;

    CHECK(mdb_txn_begin(env, NULL, 0, &txn) == MDB_SUCCESS);
    CHECK(txn->mt_dbs[main_dbi].md_entries == 1);
    CHECK(txn->mt_dbs[main_dbi].md_keys == 1);
    {
        MDB_val nk, catalog;
        MDB_aggval expect_main, got_main;
        nk.mv_data = (void *)"named"; nk.mv_size = 5;
        CHECK(mdb_get(txn, main_dbi, &nk, &catalog) == MDB_SUCCESS);
        CHECK(mdb_agg_record_contribution(DB_AGGFLAGS(&txn->mt_dbs[main_dbi]),
            txn->mt_dbs[main_dbi].md_hash_offset, &catalog, &expect_main) == MDB_SUCCESS);
        mdb_db_get_aggval(&txn->mt_dbs[main_dbi], &got_main);
        CHECK(agg_equal(DB_AGGFLAGS(&txn->mt_dbs[main_dbi]), &got_main, &expect_main));
    }
    CHECK(mdb_drop(txn, named, 0) == MDB_SUCCESS);
    CHECK(txn->mt_dbs[named].md_entries == 0);
    CHECK(txn->mt_dbs[named].md_keys == 0);
    CHECK(is_zero_hash(txn->mt_dbs[named].md_hashsum));
    CHECK(mdb_agg_info(txn, named, &af) == MDB_SUCCESS && af == agg);
    CHECK(mdb_get_hash_offset(txn, named, &off) == MDB_SUCCESS && off == 4);
    make_value(value, sizeof(value), 8, 3);
    CHECK(mdb_put(txn, named, &key, &val, 0) == MDB_SUCCESS);
    CHECK(mdb_drop(txn, named, 1) == MDB_SUCCESS);
    CHECK(txn->mt_dbs[main_dbi].md_entries == 0);
    CHECK(txn->mt_dbs[main_dbi].md_keys == 0);
    CHECK(is_zero_hash(txn->mt_dbs[main_dbi].md_hashsum));
    CHECK(mdb_txn_commit(txn) == MDB_SUCCESS); txn = NULL;

    CHECK(mdb_txn_begin(env, NULL, MDB_RDONLY, &txn) == MDB_SUCCESS);
    CHECK(mdb_dbi_open(txn, "named", 0, &named) == MDB_NOTFOUND);
    cleanup_env(env, txn, dir);
    return 0;
}

static int
test_nested_transactions(void)
{
    const uint16_t agg = MDB_AGG_ENTRIES|MDB_AGG_KEYS|MDB_AGG_HASHSUM;
    char dir[] = "/tmp/aelmdb-m7-nested-XXXXXX";
    MDB_env *env = NULL; MDB_txn *txn = NULL, *child = NULL; MDB_dbi dbi;
    MDB_aggval expected;
    MDB_val key, val;
    unsigned char a[INTEGRATION_VALUE_SIZE], b[INTEGRATION_VALUE_SIZE], repl[INTEGRATION_VALUE_SIZE];

    CHECK(mkdtemp(dir) != NULL);
    CHECK(mdb_env_create(&env) == MDB_SUCCESS);
    CHECK(mdb_env_set_mapsize(env, 32u*1024u*1024u) == MDB_SUCCESS);
    CHECK(mdb_env_open(env, dir, 0, 0600) == MDB_SUCCESS);
    CHECK(mdb_txn_begin(env, NULL, 0, &txn) == MDB_SUCCESS);
    CHECK(mdb_dbi_open(txn, NULL, agg, &dbi) == MDB_SUCCESS);
    CHECK(mdb_set_hash_offset(txn, dbi, 0) == MDB_SUCCESS);
    mdb_aggval_zero(&expected);
    make_value(a, sizeof(a), 1, 1); key.mv_data=(void *)"a"; key.mv_size=1; val.mv_data=a; val.mv_size=sizeof(a);
    CHECK(mdb_put(txn, dbi, &key, &val, 0) == MDB_SUCCESS);
    CHECK(expected_add_value(&txn->mt_dbs[dbi], &val, 1, &expected) == MDB_SUCCESS);

    CHECK(mdb_txn_begin(env, txn, 0, &child) == MDB_SUCCESS);
    make_value(b, sizeof(b), 2, 2); key.mv_data=(void *)"b"; val.mv_data=b; val.mv_size=sizeof(b);
    CHECK(mdb_put(child, dbi, &key, &val, 0) == MDB_SUCCESS);
    CHECK(mdb_txn_commit(child) == MDB_SUCCESS); child = NULL;
    CHECK(expected_add_value(&txn->mt_dbs[dbi], &val, 1, &expected) == MDB_SUCCESS);
    CHECK(check_all(txn, dbi, &expected) == MDB_SUCCESS);

    CHECK(mdb_txn_begin(env, txn, 0, &child) == MDB_SUCCESS);
    make_value(repl, sizeof(repl), 1, 9); key.mv_data=(void *)"a"; val.mv_data=repl; val.mv_size=sizeof(repl);
    CHECK(mdb_put(child, dbi, &key, &val, 0) == MDB_SUCCESS);
    mdb_txn_abort(child); child = NULL;
    CHECK(check_all(txn, dbi, &expected) == MDB_SUCCESS);
    CHECK(mdb_txn_commit(txn) == MDB_SUCCESS); txn = NULL;
    CHECK(mdb_txn_begin(env, NULL, MDB_RDONLY, &txn) == MDB_SUCCESS);
    CHECK(check_all(txn, dbi, &expected) == MDB_SUCCESS);
    cleanup_env(env, txn, dir);
    return 0;
}

static int
test_compact_copy(void)
{
    const uint16_t agg = MDB_AGG_ENTRIES|MDB_AGG_KEYS|MDB_AGG_HASHSUM;
    char src[] = "/tmp/aelmdb-m7-copy-src-XXXXXX";
    char dst[] = "/tmp/aelmdb-m7-copy-dst-XXXXXX";
    MDB_env *env = NULL, *copyenv = NULL; MDB_txn *txn = NULL; MDB_dbi dbi;
    MDB_aggval expected;
    MDB_val key, val;
    unsigned char value[INTEGRATION_VALUE_SIZE];
    char keybuf[32];
    unsigned i;

    CHECK(mkdtemp(src) != NULL);
    CHECK(mkdtemp(dst) != NULL);
    CHECK(mdb_env_create(&env) == MDB_SUCCESS);
    CHECK(mdb_env_set_mapsize(env, 64u*1024u*1024u) == MDB_SUCCESS);
    CHECK(mdb_env_open(env, src, 0, 0600) == MDB_SUCCESS);
    CHECK(mdb_txn_begin(env, NULL, 0, &txn) == MDB_SUCCESS);
    CHECK(mdb_dbi_open(txn, NULL, agg, &dbi) == MDB_SUCCESS);
    CHECK(mdb_set_hash_offset(txn, dbi, 5) == MDB_SUCCESS);
    mdb_aggval_zero(&expected);
    for (i = 0; i < 1400; ++i) {
        snprintf(keybuf, sizeof(keybuf), "c%06u", i);
        key.mv_data=keybuf; key.mv_size=strlen(keybuf);
        make_value(value, sizeof(value), i, 4); val.mv_data=value; val.mv_size=sizeof(value);
        CHECK(mdb_put(txn, dbi, &key, &val, MDB_APPEND) == MDB_SUCCESS);
        CHECK(expected_add_value(&txn->mt_dbs[dbi], &val, 1, &expected) == MDB_SUCCESS);
    }
    CHECK(check_all(txn, dbi, &expected) == MDB_SUCCESS);
    CHECK(mdb_txn_commit(txn) == MDB_SUCCESS); txn = NULL;
    CHECK(mdb_env_copy2(env, dst, MDB_CP_COMPACT) == MDB_SUCCESS);
    mdb_env_close(env); env = NULL;

    CHECK(mdb_env_create(&copyenv) == MDB_SUCCESS);
    CHECK(mdb_env_set_mapsize(copyenv, 64u*1024u*1024u) == MDB_SUCCESS);
    CHECK(mdb_env_open(copyenv, dst, 0, 0600) == MDB_SUCCESS);
    CHECK(mdb_txn_begin(copyenv, NULL, MDB_RDONLY, &txn) == MDB_SUCCESS);
    dbi = MAIN_DBI;
    CHECK(check_all(txn, dbi, &expected) == MDB_SUCCESS);
#if defined(MDB_DEBUG_AGG_INTEGRITY) && MDB_DEBUG_AGG_INTEGRITY
    CHECK(mdb_agg_check_integrity(txn, dbi) == MDB_SUCCESS);
#endif
    cleanup_env(copyenv, txn, dst); copyenv = NULL; txn = NULL;
    /* source was already closed */
    { char a[256], b[256]; snprintf(a,sizeof(a),"%s/data.mdb",src); snprintf(b,sizeof(b),"%s/lock.mdb",src); unlink(a); unlink(b); rmdir(src); }
    return 0;
}

static int
test_schema_and_noops(void)
{
    const uint16_t agg = MDB_AGG_ENTRIES|MDB_AGG_KEYS|MDB_AGG_HASHSUM;
    char dir[] = "/tmp/aelmdb-m7-schema-XXXXXX";
    MDB_env *env = NULL; MDB_txn *txn = NULL; MDB_dbi dbi;
    MDB_aggval expected, before;
    MDB_val key, val;
    unsigned char value[INTEGRATION_VALUE_SIZE];
    unsigned flags;
    int off;

    CHECK(mkdtemp(dir) != NULL);
    CHECK(mdb_env_create(&env) == MDB_SUCCESS);
    CHECK(mdb_env_set_mapsize(env, 32u*1024u*1024u) == MDB_SUCCESS);
    CHECK(mdb_env_open(env, dir, 0, 0600) == MDB_SUCCESS);
    CHECK(mdb_txn_begin(env, NULL, 0, &txn) == MDB_SUCCESS);
    CHECK(mdb_dbi_open(txn, NULL, MDB_AGG_HASHSOURCE_FROM_KEY, &dbi) == MDB_INCOMPATIBLE);
    CHECK(mdb_dbi_open(txn, NULL, MDB_DUPSORT|MDB_AGG_HASHSUM|MDB_AGG_HASHSOURCE_FROM_KEY, &dbi) == MDB_INCOMPATIBLE);
    CHECK(mdb_dbi_open(txn, NULL, agg, &dbi) == MDB_SUCCESS);
    CHECK(mdb_set_hash_offset(txn, dbi, -1) == MDB_SUCCESS);
    CHECK(mdb_get_hash_offset(txn, dbi, &off) == MDB_SUCCESS && off == -1);
    CHECK(mdb_agg_info(txn, dbi, &flags) == MDB_SUCCESS && flags == agg);
    make_value(value, sizeof(value), 10, 10); key.mv_data=(void *)"noop"; key.mv_size=4; val.mv_data=value; val.mv_size=sizeof(value);
    mdb_aggval_zero(&expected);
    CHECK(mdb_put(txn, dbi, &key, &val, 0) == MDB_SUCCESS);
    CHECK(expected_add_value(&txn->mt_dbs[dbi], &val, 1, &expected) == MDB_SUCCESS);
    before = expected;
    CHECK(mdb_put(txn, dbi, &key, &val, MDB_NOOVERWRITE) == MDB_KEYEXIST);
    CHECK(agg_equal(agg, &before, &expected));
    CHECK(check_all(txn, dbi, &expected) == MDB_SUCCESS);
    CHECK(mdb_set_hash_offset(txn, dbi, 0) == MDB_INCOMPATIBLE);
    CHECK(mdb_dbi_open(txn, NULL, MDB_AGG_ENTRIES, &dbi) == MDB_INCOMPATIBLE);
    cleanup_env(env, txn, dir);
    return 0;
}


#if defined(MDB_DEBUG_AGG_INTEGRITY) && MDB_DEBUG_AGG_INTEGRITY
static int
test_debug_oracle_detects_corruption(void)
{
    const uint16_t agg = MDB_AGG_ENTRIES|MDB_AGG_KEYS|MDB_AGG_HASHSUM;
    char dir[] = "/tmp/aelmdb-m7-oracle-XXXXXX";
    MDB_env *env = NULL; MDB_txn *txn = NULL; MDB_dbi dbi;
    MDB_val key, val;
    unsigned char value[INTEGRATION_VALUE_SIZE];
    char keybuf[32];
    MDB_cursor mc; MDB_xcursor mx; MDB_page *root; MDB_node *node; MDB_aggval bad;
    unsigned i; int rc;

    CHECK(mkdtemp(dir) != NULL);
    CHECK(mdb_env_create(&env) == MDB_SUCCESS);
    CHECK(mdb_env_set_mapsize(env, 64u*1024u*1024u) == MDB_SUCCESS);
    CHECK(mdb_env_open(env, dir, 0, 0600) == MDB_SUCCESS);
    CHECK(mdb_txn_begin(env, NULL, 0, &txn) == MDB_SUCCESS);
    CHECK(mdb_dbi_open(txn, NULL, agg, &dbi) == MDB_SUCCESS);
    CHECK(mdb_set_hash_offset(txn, dbi, 0) == MDB_SUCCESS);
    for (i = 0; i < 900; ++i) {
        snprintf(keybuf, sizeof(keybuf), "o%06u", i);
        key.mv_data = keybuf; key.mv_size = strlen(keybuf);
        make_value(value, sizeof(value), i, 29);
        val.mv_data = value; val.mv_size = sizeof(value);
        CHECK(mdb_put(txn, dbi, &key, &val, MDB_APPEND) == MDB_SUCCESS);
    }
    CHECK(txn->mt_dbs[dbi].md_depth >= 2);
    CHECK(mdb_agg_check_integrity(txn, dbi) == MDB_SUCCESS);
    mdb_cursor_init(&mc, txn, dbi, &mx);
    rc = mdb_page_get(&mc, txn->mt_dbs[dbi].md_root, &root, NULL);
    CHECK(rc == MDB_SUCCESS && IS_BRANCH(root) && (root->mp_flags & P_DIRTY));
    node = NODEPTR(root, 0);
    mdb_node_get_aggval(root, node, &bad);
    bad.entries++;
    mdb_node_set_aggval(root, node, &bad);
    CHECK(mdb_agg_check_integrity(txn, dbi) == MDB_CORRUPTED);
    cleanup_env(env, txn, dir);
    return 0;
}
#endif

int
main(void)
{
    if (dupsort_regression_main()) return 1;
    if (test_reserve_contract()) return 1;
    if (test_writemap_value_hash()) return 1;
    if (test_writemap_dupsort()) return 1;
    if (test_drop_and_catalog()) return 1;
    if (test_nested_transactions()) return 1;
    if (test_compact_copy()) return 1;
    if (test_schema_and_noops()) return 1;
    if (test_delete_cursor_relocation()) return 1;
    if (test_dupsort_structural_delete_relocation()) return 1;
#if defined(MDB_DEBUG_AGG_INTEGRITY) && MDB_DEBUG_AGG_INTEGRITY
    if (test_debug_oracle_detects_corruption()) return 1;
#endif
    printf("aggregate integration tests passed (MDB_HASH_SIZE=%d)\n", MDB_HASH_SIZE);
    return 0;
}
