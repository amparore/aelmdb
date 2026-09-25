#ifndef _GNU_SOURCE
#define _GNU_SOURCE 1
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>

#include "aelmdb_mdb.c"

#define NESTED_VALUE_SIZE ((MDB_HASH_SIZE + 8 > 112) ? MDB_HASH_SIZE + 8 : 112)
#define PRIMARY_VALUE_SIZE ((MDB_HASH_SIZE + 5 > 220) ? MDB_HASH_SIZE + 5 : 220)

#define CHECK(cond) do { \
    if (!(cond)) { \
        fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        return 1; \
    } \
} while (0)

static int
agg_equal(uint16_t agg, const MDB_aggval *a, const MDB_aggval *b)
{
    if ((agg & MDB_AGG_ENTRIES) && a->entries != b->entries) return 0;
    if ((agg & MDB_AGG_KEYS) && a->keys != b->keys) return 0;
    if ((agg & MDB_AGG_HASHSUM) && memcmp(a->hashsum, b->hashsum, MDB_HASH_SIZE)) return 0;
    return 1;
}

static int
expected_add_value(MDB_db *db, const MDB_val *value, int new_primary_key,
    MDB_aggval *expected)
{
    MDB_aggval one;
    uint16_t agg = DB_AGGFLAGS(db);
    int rc = mdb_agg_record_contribution(agg, db->md_hash_offset, value, &one);
    if (rc) return rc;
    if ((agg & MDB_AGG_KEYS) && !new_primary_key) one.keys = 0;
    return mdb_aggval_add(agg, expected, &one);
}

static int
expected_sub_value(MDB_db *db, const MDB_val *value, int last_primary_value,
    MDB_aggval *expected)
{
    MDB_aggval one;
    uint16_t agg = DB_AGGFLAGS(db);
    int rc = mdb_agg_record_contribution(agg, db->md_hash_offset, value, &one);
    if (rc) return rc;
    if ((agg & MDB_AGG_KEYS) && !last_primary_value) one.keys = 0;
    return mdb_aggval_sub(agg, expected, &one);
}

static int
oracle_dup_leaf(MDB_cursor *dc, MDB_page *mp, MDB_aggval *out)
{
    MDB_aggval total, one;
    uint16_t agg = DB_AGGFLAGS(dc->mc_db);
    unsigned i;
    int rc;
    mdb_aggval_zero(&total);
    if (!IS_LEAF(mp)) return MDB_CORRUPTED;
    if (IS_LEAF2(mp)) {
        for (i = 0; i < NUMKEYS(mp); ++i) {
            MDB_val v;
            v.mv_size = dc->mc_db->md_pad;
            v.mv_data = LEAF2KEY(mp, i, v.mv_size);
            rc = mdb_agg_record_contribution(agg, dc->mc_db->md_hash_offset, &v, &one);
            if (rc) return rc;
            rc = mdb_aggval_add(agg, &total, &one);
            if (rc) return rc;
        }
    } else {
        for (i = 0; i < NUMKEYS(mp); ++i) {
            MDB_node *n = NODEPTR(mp, i);
            MDB_val v;
            if (n->mn_flags & (F_BIGDATA|F_DUPDATA|F_SUBDATA)) return MDB_CORRUPTED;
            v.mv_size = NODEKSZ(n);
            v.mv_data = NODEKEY(mp, n);
            rc = mdb_agg_record_contribution(agg, dc->mc_db->md_hash_offset, &v, &one);
            if (rc) return rc;
            rc = mdb_aggval_add(agg, &total, &one);
            if (rc) return rc;
        }
    }
    *out = total;
    return MDB_SUCCESS;
}

static int
oracle_dup_subtree(MDB_cursor *dc, pgno_t pgno, MDB_aggval *out)
{
    MDB_page *mp;
    MDB_aggval total, child, stored;
    uint16_t agg = DB_AGGFLAGS(dc->mc_db);
    unsigned i;
    int rc = mdb_page_get(dc, pgno, &mp, NULL);
    if (rc) return rc;
    if (IS_LEAF(mp)) return oracle_dup_leaf(dc, mp, out);
    if (!IS_BRANCH(mp)) return MDB_CORRUPTED;
    mdb_aggval_zero(&total);
    for (i = 0; i < NUMKEYS(mp); ++i) {
        MDB_node *n = NODEPTR(mp, i);
        rc = oracle_dup_subtree(dc, NODEPGNO(n), &child);
        if (rc) return rc;
        mdb_node_get_aggval(mp, n, &stored);
        if (!agg_equal(agg, &stored, &child)) return MDB_CORRUPTED;
        rc = mdb_aggval_add(agg, &total, &child);
        if (rc) return rc;
    }
    *out = total;
    return MDB_SUCCESS;
}

static int
oracle_primary_node(MDB_cursor *mc, MDB_page *mp, MDB_node *node,
    MDB_aggval *out)
{
    uint16_t agg = DB_AGGFLAGS(mc->mc_db);
    MDB_aggval val;
    int rc;
    if (node->mn_flags & F_DUPDATA) {
        if (node->mn_flags & F_SUBDATA) {
            MDB_db dupdb;
            MDB_aggval dbval;
            if (NODEDSZ(node) != sizeof(MDB_db)) return MDB_CORRUPTED;
            memcpy(&dupdb, NODEDATA(node), sizeof(dupdb));
            rc = mdb_xcursor_init1(mc, node);
            if (rc) return rc;
            rc = oracle_dup_subtree(&mc->mc_xcursor->mx_cursor, dupdb.md_root, &val);
            if (rc) return rc;
            mdb_db_get_aggval(&mc->mc_xcursor->mx_db, &dbval);
            if (!agg_equal(agg, &val, &dbval)) return MDB_CORRUPTED;
        } else {
            MDB_page *sp = NODEDATA(node);
            MDB_cursor dc = mc->mc_xcursor->mx_cursor;
            MDB_db ddb;
            memset(&ddb, 0, sizeof(ddb));
            ddb.md_flags = DB_AGGFLAGS(mc->mc_db);
            if (mc->mc_db->md_flags & MDB_DUPFIXED) {
                ddb.md_flags |= MDB_DUPFIXED;
                ddb.md_pad = sp->mp_pad;
            }
            ddb.md_hash_offset = mc->mc_db->md_hash_offset;
            dc.mc_db = &ddb;
            rc = oracle_dup_leaf(&dc, sp, &val);
            if (rc) return rc;
        }
        if ((agg & MDB_AGG_ENTRIES) && val.entries == 0) return MDB_CORRUPTED;
        if (agg & MDB_AGG_KEYS) val.keys = 1;
        *out = val;
        return MDB_SUCCESS;
    } else {
        MDB_val v;
        MDB_cursor scan = *mc;
        MC_SET_OVPG(&scan, NULL);
        rc = mdb_node_read(&scan, node, &v);
        if (rc) return rc;
        rc = mdb_agg_record_contribution(agg, mc->mc_db->md_hash_offset, &v, out);
#ifdef MDB_VL32
        if (MC_OVPG(&scan)) MDB_PAGE_UNREF(scan.mc_txn, MC_OVPG(&scan));
#endif
        return rc;
    }
}

static int
oracle_primary_subtree(MDB_cursor *mc, pgno_t pgno, MDB_aggval *out)
{
    MDB_page *mp;
    MDB_aggval total, child, stored;
    uint16_t agg = DB_AGGFLAGS(mc->mc_db);
    unsigned i;
    int rc = mdb_page_get(mc, pgno, &mp, NULL);
    if (rc) return rc;
    mdb_aggval_zero(&total);
    if (IS_LEAF(mp)) {
        if (IS_LEAF2(mp)) return MDB_CORRUPTED;
        for (i = 0; i < NUMKEYS(mp); ++i) {
            rc = oracle_primary_node(mc, mp, NODEPTR(mp, i), &child);
            if (rc) return rc;
            rc = mdb_aggval_add(agg, &total, &child);
            if (rc) return rc;
        }
        *out = total;
        return MDB_SUCCESS;
    }
    if (!IS_BRANCH(mp)) return MDB_CORRUPTED;
    for (i = 0; i < NUMKEYS(mp); ++i) {
        MDB_node *n = NODEPTR(mp, i);
        rc = oracle_primary_subtree(mc, NODEPGNO(n), &child);
        if (rc) return rc;
        mdb_node_get_aggval(mp, n, &stored);
        if (!agg_equal(agg, &stored, &child)) return MDB_CORRUPTED;
        rc = mdb_aggval_add(agg, &total, &child);
        if (rc) return rc;
    }
    *out = total;
    return MDB_SUCCESS;
}

static int
check_all(MDB_txn *txn, MDB_dbi dbi, const MDB_aggval *expected)
{
    MDB_db *db = &txn->mt_dbs[dbi];
    MDB_aggval got, dbgot;
    MDB_cursor mc;
    MDB_xcursor mx;
    int rc;
    mdb_db_get_aggval(db, &dbgot);
    if (!agg_equal(DB_AGGFLAGS(db), &dbgot, expected)) return MDB_CORRUPTED;
    if (db->md_root == P_INVALID) {
        MDB_aggval zero;
        mdb_aggval_zero(&zero);
        return agg_equal(DB_AGGFLAGS(db), &zero, expected) ? MDB_SUCCESS : MDB_CORRUPTED;
    }
    mdb_cursor_init(&mc, txn, dbi, &mx);
    rc = oracle_primary_subtree(&mc, db->md_root, &got);
    if (rc) return rc;
    return agg_equal(DB_AGGFLAGS(db), &got, expected) ? MDB_SUCCESS : MDB_CORRUPTED;
}

static int
primary_representation(MDB_txn *txn, MDB_dbi dbi, MDB_val *key,
    unsigned *flags, MDB_db *dupdb)
{
    MDB_cursor mc;
    MDB_xcursor mx;
    MDB_node *node;
    int exact = 0, rc;
    mdb_cursor_init(&mc, txn, dbi, &mx);
    rc = mdb_cursor_set(&mc, key, NULL, MDB_SET, &exact);
    if (rc) return rc;
    if (!exact) return MDB_NOTFOUND;
    node = NODEPTR(mc.mc_pg[mc.mc_top], mc.mc_ki[mc.mc_top]);
    *flags = node->mn_flags;
    if ((node->mn_flags & (F_DUPDATA|F_SUBDATA)) == (F_DUPDATA|F_SUBDATA)) {
        if (NODEDSZ(node) != sizeof(MDB_db)) return MDB_CORRUPTED;
        memcpy(dupdb, NODEDATA(node), sizeof(*dupdb));
    } else {
        memset(dupdb, 0, sizeof(*dupdb));
        if (node->mn_flags & F_DUPDATA) {
            MDB_page *sp = NODEDATA(node);
            dupdb->md_entries = NUMKEYS(sp);
            dupdb->md_depth = 1;
        } else {
            dupdb->md_entries = 1;
        }
    }
    return MDB_SUCCESS;
}

static void
make_value(unsigned char *buf, size_t len, unsigned n, unsigned salt)
{
    size_t i;
    memset(buf, 0, len);
    snprintf((char *)buf, len, "%08u-%04u-", n, salt);
    for (i = 14; i < len; ++i) buf[i] = (unsigned char)('a' + (n + salt + i) % 26);
}

static void
cleanup_env(MDB_env *env, MDB_txn *txn, const char *dir)
{
    char data_path[256], lock_path[256];
    if (txn) mdb_txn_abort(txn);
    if (env) mdb_env_close(env);
    snprintf(data_path, sizeof(data_path), "%s/data.mdb", dir);
    snprintf(lock_path, sizeof(lock_path), "%s/lock.mdb", dir);
    unlink(data_path); unlink(lock_path); rmdir(dir);
}

static int
replace_current_dup(MDB_txn *txn, MDB_dbi dbi, MDB_val *key,
    const MDB_val *oldval, MDB_val *newval)
{
    MDB_cursor *cur = NULL;
    MDB_val seek_key = *key, seek_data = *oldval;
    int rc;

    rc = mdb_cursor_open(txn, dbi, &cur);
    if (rc) return rc;
    rc = mdb_cursor_get(cur, &seek_key, &seek_data, MDB_GET_BOTH);
    if (!rc)
        rc = mdb_cursor_put(cur, key, newval, MDB_CURRENT);
    mdb_cursor_close(cur);
    return rc;
}

static int
test_current_replacements(void)
{
    const uint16_t agg = MDB_AGG_ENTRIES|MDB_AGG_KEYS|MDB_AGG_HASHSUM;
    const size_t SZ = MDB_HASH_SIZE + 24 > 96 ? MDB_HASH_SIZE + 24 : 96;
    char dir[] = "/tmp/aelmdb-dupsort-current-XXXXXX";
    MDB_env *env = NULL; MDB_txn *txn = NULL; MDB_dbi dbi;
    MDB_aggval expected;
    MDB_val key, oldv, newv, v;
    unsigned char *oldbuf = NULL, *newbuf = NULL, *tmpbuf = NULL;
    char keybuf[32];
    unsigned i, flags;
    MDB_db dupdb;

    CHECK(mkdtemp(dir) != NULL);
    oldbuf = malloc(SZ); newbuf = malloc(SZ); tmpbuf = malloc(SZ);
    CHECK(oldbuf && newbuf && tmpbuf);
    CHECK(mdb_env_create(&env) == MDB_SUCCESS);
    CHECK(mdb_env_set_mapsize(env, 32u*1024u*1024u) == MDB_SUCCESS);
    CHECK(mdb_env_open(env, dir, 0, 0600) == MDB_SUCCESS);
    CHECK(mdb_txn_begin(env, NULL, 0, &txn) == MDB_SUCCESS);
    CHECK(mdb_dbi_open(txn, NULL, MDB_DUPSORT|agg, &dbi) == MDB_SUCCESS);
    CHECK(mdb_set_hash_offset(txn, dbi, 7) == MDB_SUCCESS);
    mdb_aggval_zero(&expected);

    /* Singleton primary representation. */
    strcpy(keybuf, "single"); key.mv_data = keybuf; key.mv_size = strlen(keybuf);
    make_value(oldbuf, SZ, 100, 10); oldv.mv_data = oldbuf; oldv.mv_size = SZ;
    CHECK(mdb_put(txn, dbi, &key, &oldv, 0) == MDB_SUCCESS);
    CHECK(expected_add_value(&txn->mt_dbs[dbi], &oldv, 1, &expected) == MDB_SUCCESS);
    CHECK(check_all(txn, dbi, &expected) == MDB_SUCCESS);
    make_value(newbuf, SZ, 100, 11); newv.mv_data = newbuf; newv.mv_size = SZ;
    CHECK(replace_current_dup(txn, dbi, &key, &oldv, &newv) == MDB_SUCCESS);
    CHECK(expected_sub_value(&txn->mt_dbs[dbi], &oldv, 0, &expected) == MDB_SUCCESS);
    CHECK(expected_add_value(&txn->mt_dbs[dbi], &newv, 0, &expected) == MDB_SUCCESS);
    CHECK(check_all(txn, dbi, &expected) == MDB_SUCCESS);

    /* Small inline dup-subpage. */
    strcpy(keybuf, "inline"); key.mv_data = keybuf; key.mv_size = strlen(keybuf);
    for (i = 0; i < 4; ++i) {
        make_value(tmpbuf, SZ, 200 + i*10, 20);
        v.mv_data = tmpbuf; v.mv_size = SZ;
        CHECK(mdb_put(txn, dbi, &key, &v, 0) == MDB_SUCCESS);
        CHECK(expected_add_value(&txn->mt_dbs[dbi], &v, i == 0, &expected) == MDB_SUCCESS);
    }
    CHECK(primary_representation(txn, dbi, &key, &flags, &dupdb) == MDB_SUCCESS);
    CHECK((flags & F_DUPDATA) && !(flags & F_SUBDATA));
    make_value(oldbuf, SZ, 220, 20); oldv.mv_data = oldbuf; oldv.mv_size = SZ;
    make_value(newbuf, SZ, 220, 21); newv.mv_data = newbuf; newv.mv_size = SZ;
    CHECK(replace_current_dup(txn, dbi, &key, &oldv, &newv) == MDB_SUCCESS);
    CHECK(expected_sub_value(&txn->mt_dbs[dbi], &oldv, 0, &expected) == MDB_SUCCESS);
    CHECK(expected_add_value(&txn->mt_dbs[dbi], &newv, 0, &expected) == MDB_SUCCESS);
    CHECK(check_all(txn, dbi, &expected) == MDB_SUCCESS);

    CHECK(mdb_txn_commit(txn) == MDB_SUCCESS); txn = NULL;
    CHECK(mdb_txn_begin(env, NULL, 0, &txn) == MDB_SUCCESS);
    CHECK(check_all(txn, dbi, &expected) == MDB_SUCCESS);

    free(oldbuf); free(newbuf); free(tmpbuf);
    cleanup_env(env, txn, dir);
    return 0;
}

static int
test_nested_dups(void)
{
    const unsigned N = 900;
    const uint16_t agg = MDB_AGG_ENTRIES|MDB_AGG_KEYS|MDB_AGG_HASHSUM;
    char dir[] = "/tmp/aelmdb-dupsort-nested-XXXXXX";
    MDB_env *env = NULL; MDB_txn *txn = NULL; MDB_dbi dbi;
    MDB_aggval expected;
    MDB_val key, val;
    unsigned char value[NESTED_VALUE_SIZE];
    char keybuf[] = "hot";
    unsigned i, flags;
    MDB_db dupdb;
    int saw_inline = 0, saw_persistent = 0, saw_deep = 0, saw_collapse = 0;
    unsigned prev_depth = 0;

    CHECK(mkdtemp(dir) != NULL);
    CHECK(mdb_env_create(&env) == MDB_SUCCESS);
    CHECK(mdb_env_set_mapsize(env, 128u*1024u*1024u) == MDB_SUCCESS);
    CHECK(mdb_env_open(env, dir, 0, 0600) == MDB_SUCCESS);
    CHECK(mdb_txn_begin(env, NULL, 0, &txn) == MDB_SUCCESS);
    CHECK(mdb_dbi_open(txn, NULL, MDB_DUPSORT|agg, &dbi) == MDB_SUCCESS);
    CHECK(mdb_set_hash_offset(txn, dbi, 7) == MDB_SUCCESS);
    mdb_aggval_zero(&expected);
    key.mv_data = keybuf; key.mv_size = strlen(keybuf);

    for (i = 0; i < N; ++i) {
        make_value(value, sizeof(value), i, 11);
        val.mv_data = value; val.mv_size = sizeof(value);
        CHECK(mdb_put(txn, dbi, &key, &val, 0) == MDB_SUCCESS);
        CHECK(expected_add_value(&txn->mt_dbs[dbi], &val, i == 0, &expected) == MDB_SUCCESS);
        CHECK(primary_representation(txn, dbi, &key, &flags, &dupdb) == MDB_SUCCESS);
        if ((flags & F_DUPDATA) && !(flags & F_SUBDATA)) saw_inline = 1;
        if ((flags & (F_DUPDATA|F_SUBDATA)) == (F_DUPDATA|F_SUBDATA)) {
            saw_persistent = 1;
            if (dupdb.md_depth >= 2) saw_deep = 1;
        }
        if (i < 4 || (i % 75u) == 0 || (saw_persistent && i < 20))
            CHECK(check_all(txn, dbi, &expected) == MDB_SUCCESS);
    }
    CHECK(saw_inline && saw_persistent && saw_deep);
    CHECK(check_all(txn, dbi, &expected) == MDB_SUCCESS);
    CHECK(mdb_txn_commit(txn) == MDB_SUCCESS); txn = NULL;
    CHECK(mdb_txn_begin(env, NULL, 0, &txn) == MDB_SUCCESS);
    CHECK(primary_representation(txn, dbi, &key, &flags, &dupdb) == MDB_SUCCESS);
    prev_depth = dupdb.md_depth;

    /* MDB_CURRENT inside a persistent, multi-level duplicate sub-DB. */
    {
        const unsigned current_i = N / 2;
        unsigned char oldbuf[NESTED_VALUE_SIZE], newbuf[NESTED_VALUE_SIZE];
        MDB_val oldv, newv;
        make_value(oldbuf, sizeof(oldbuf), current_i, 11);
        make_value(newbuf, sizeof(newbuf), current_i, 12);
        oldv.mv_data = oldbuf; oldv.mv_size = sizeof(oldbuf);
        newv.mv_data = newbuf; newv.mv_size = sizeof(newbuf);
        CHECK(replace_current_dup(txn, dbi, &key, &oldv, &newv) == MDB_SUCCESS);
        CHECK(expected_sub_value(&txn->mt_dbs[dbi], &oldv, 0, &expected) == MDB_SUCCESS);
        CHECK(expected_add_value(&txn->mt_dbs[dbi], &newv, 0, &expected) == MDB_SUCCESS);
        CHECK(check_all(txn, dbi, &expected) == MDB_SUCCESS);
    }

    for (i = 0; i < N-1; ++i) {
        make_value(value, sizeof(value), i, i == N/2 ? 12 : 11);
        val.mv_data = value; val.mv_size = sizeof(value);
        CHECK(mdb_del(txn, dbi, &key, &val) == MDB_SUCCESS);
        CHECK(expected_sub_value(&txn->mt_dbs[dbi], &val, 0, &expected) == MDB_SUCCESS);
        CHECK(primary_representation(txn, dbi, &key, &flags, &dupdb) == MDB_SUCCESS);
        if (dupdb.md_depth < prev_depth) saw_collapse = 1;
        prev_depth = dupdb.md_depth;
        if ((i % 73u) == 0 || i + 8 >= N)
            CHECK(check_all(txn, dbi, &expected) == MDB_SUCCESS);
    }
    CHECK(saw_collapse);
    make_value(value, sizeof(value), N-1, 11);
    val.mv_data = value; val.mv_size = sizeof(value);
    CHECK(mdb_del(txn, dbi, &key, &val) == MDB_SUCCESS);
    CHECK(expected_sub_value(&txn->mt_dbs[dbi], &val, 1, &expected) == MDB_SUCCESS);
    CHECK(check_all(txn, dbi, &expected) == MDB_SUCCESS);
    CHECK(txn->mt_dbs[dbi].md_root == P_INVALID);
    CHECK(mdb_txn_commit(txn) == MDB_SUCCESS); txn = NULL;
    cleanup_env(env, txn, dir);
    return 0;
}

static int
test_primary_split_and_bulk_delete(void)
{
    const unsigned N = 300;
    const uint16_t agg = MDB_AGG_ENTRIES|MDB_AGG_KEYS|MDB_AGG_HASHSUM;
    char dir[] = "/tmp/aelmdb-dupsort-primary-XXXXXX";
    MDB_env *env = NULL; MDB_txn *txn = NULL; MDB_dbi dbi;
    MDB_aggval expected;
    MDB_val key, val;
    unsigned char value[PRIMARY_VALUE_SIZE];
    char keybuf[64];
    unsigned i;
    int saw_dup_split = 0, saw_primary_shrink = 0;

    CHECK(mkdtemp(dir) != NULL);
    CHECK(mdb_env_create(&env) == MDB_SUCCESS);
    CHECK(mdb_env_set_mapsize(env, 128u*1024u*1024u) == MDB_SUCCESS);
    CHECK(mdb_env_open(env, dir, 0, 0600) == MDB_SUCCESS);
    CHECK(mdb_txn_begin(env, NULL, 0, &txn) == MDB_SUCCESS);
    CHECK(mdb_dbi_open(txn, NULL, MDB_DUPSORT|agg, &dbi) == MDB_SUCCESS);
    CHECK(mdb_set_hash_offset(txn, dbi, 5) == MDB_SUCCESS);
    mdb_aggval_zero(&expected);

    for (i = 0; i < N; ++i) {
        snprintf(keybuf, sizeof(keybuf), "k%05u-abcdefghijklmnop", i);
        key.mv_data = keybuf; key.mv_size = strlen(keybuf);
        make_value(value, sizeof(value), i, 21);
        val.mv_data = value; val.mv_size = sizeof(value);
        CHECK(mdb_put(txn, dbi, &key, &val, MDB_APPEND) == MDB_SUCCESS);
        CHECK(expected_add_value(&txn->mt_dbs[dbi], &val, 1, &expected) == MDB_SUCCESS);
    }
    CHECK(txn->mt_dbs[dbi].md_depth >= 2);
    CHECK(check_all(txn, dbi, &expected) == MDB_SUCCESS);

    for (i = 0; i < N; ++i) {
        mdb_size_t leaves = txn->mt_dbs[dbi].md_leaf_pages;
        snprintf(keybuf, sizeof(keybuf), "k%05u-abcdefghijklmnop", i);
        key.mv_data = keybuf; key.mv_size = strlen(keybuf);
        make_value(value, sizeof(value), i, 77);
        val.mv_data = value; val.mv_size = sizeof(value);
        CHECK(mdb_put(txn, dbi, &key, &val, 0) == MDB_SUCCESS);
        CHECK(expected_add_value(&txn->mt_dbs[dbi], &val, 0, &expected) == MDB_SUCCESS);
        if (txn->mt_dbs[dbi].md_leaf_pages > leaves) saw_dup_split = 1;
        if ((i % 29u) == 0 || txn->mt_dbs[dbi].md_leaf_pages > leaves)
            CHECK(check_all(txn, dbi, &expected) == MDB_SUCCESS);
    }
    CHECK(saw_dup_split);
    CHECK(check_all(txn, dbi, &expected) == MDB_SUCCESS);
    CHECK(mdb_txn_commit(txn) == MDB_SUCCESS); txn = NULL;
    CHECK(mdb_txn_begin(env, NULL, 0, &txn) == MDB_SUCCESS);

    for (i = 0; i < N-20; ++i) {
        mdb_size_t leaves = txn->mt_dbs[dbi].md_leaf_pages;
        unsigned char v1[PRIMARY_VALUE_SIZE], v2[PRIMARY_VALUE_SIZE];
        MDB_val a, b;
        snprintf(keybuf, sizeof(keybuf), "k%05u-abcdefghijklmnop", i);
        key.mv_data = keybuf; key.mv_size = strlen(keybuf);
        make_value(v1, sizeof(v1), i, 21); a.mv_data = v1; a.mv_size = sizeof(v1);
        make_value(v2, sizeof(v2), i, 77); b.mv_data = v2; b.mv_size = sizeof(v2);
        CHECK(mdb_del(txn, dbi, &key, NULL) == MDB_SUCCESS);
        CHECK(expected_sub_value(&txn->mt_dbs[dbi], &a, 0, &expected) == MDB_SUCCESS);
        CHECK(expected_sub_value(&txn->mt_dbs[dbi], &b, 1, &expected) == MDB_SUCCESS);
        if (txn->mt_dbs[dbi].md_leaf_pages < leaves) saw_primary_shrink = 1;
        if ((i % 31u) == 0 || txn->mt_dbs[dbi].md_leaf_pages < leaves)
            CHECK(check_all(txn, dbi, &expected) == MDB_SUCCESS);
    }
    CHECK(saw_primary_shrink);
    CHECK(check_all(txn, dbi, &expected) == MDB_SUCCESS);
    cleanup_env(env, txn, dir);
    return 0;
}

static int
test_dupfixed_multiple(void)
{
    const unsigned N = 96;
    const size_t SZ = MDB_HASH_SIZE < 32 ? 32 : MDB_HASH_SIZE;
    const uint16_t agg = MDB_AGG_ENTRIES|MDB_AGG_KEYS|MDB_AGG_HASHSUM;
    char dir[] = "/tmp/aelmdb-dupsort-fixed-XXXXXX";
    MDB_env *env = NULL; MDB_txn *txn = NULL; MDB_dbi dbi; MDB_cursor *cur = NULL;
    MDB_aggval expected;
    MDB_val key, data[2], one;
    char keybuf[] = "fixed";
    unsigned char *block;
    unsigned i;

    CHECK(mkdtemp(dir) != NULL);
    block = malloc(N * SZ); CHECK(block != NULL);
    for (i = 0; i < N; ++i) make_value(block + i*SZ, SZ, i, 3);
    CHECK(mdb_env_create(&env) == MDB_SUCCESS);
    CHECK(mdb_env_set_mapsize(env, 32u*1024u*1024u) == MDB_SUCCESS);
    CHECK(mdb_env_open(env, dir, 0, 0600) == MDB_SUCCESS);
    CHECK(mdb_txn_begin(env, NULL, 0, &txn) == MDB_SUCCESS);
    CHECK(mdb_dbi_open(txn, NULL, MDB_DUPSORT|MDB_DUPFIXED|agg, &dbi) == MDB_SUCCESS);
    CHECK(mdb_set_hash_offset(txn, dbi, 0) == MDB_SUCCESS);
    mdb_aggval_zero(&expected);
    key.mv_data = keybuf; key.mv_size = strlen(keybuf);
    data[0].mv_data = block; data[0].mv_size = SZ;
    data[1].mv_data = NULL; data[1].mv_size = N;
    CHECK(mdb_cursor_open(txn, dbi, &cur) == MDB_SUCCESS);
    CHECK(mdb_cursor_put(cur, &key, data, MDB_MULTIPLE) == MDB_SUCCESS);
    mdb_cursor_close(cur); cur = NULL;
    CHECK(data[1].mv_size == N);
    for (i = 0; i < N; ++i) {
        one.mv_data = block + i*SZ; one.mv_size = SZ;
        CHECK(expected_add_value(&txn->mt_dbs[dbi], &one, i == 0, &expected) == MDB_SUCCESS);
    }
    CHECK(check_all(txn, dbi, &expected) == MDB_SUCCESS);
    for (i = 0; i < N/2; ++i) {
        one.mv_data = block + i*SZ; one.mv_size = SZ;
        CHECK(mdb_del(txn, dbi, &key, &one) == MDB_SUCCESS);
        CHECK(expected_sub_value(&txn->mt_dbs[dbi], &one, 0, &expected) == MDB_SUCCESS);
    }
    CHECK(check_all(txn, dbi, &expected) == MDB_SUCCESS);
    free(block);
    cleanup_env(env, txn, dir);
    return 0;
}

int
main(void)
{
    if (test_current_replacements()) return 1;
    if (test_nested_dups()) return 1;
    if (test_primary_split_and_bulk_delete()) return 1;
    if (test_dupfixed_multiple()) return 1;
    printf("aggregate DUPSORT maintenance tests passed (MDB_HASH_SIZE=%d)\n", MDB_HASH_SIZE);
    return 0;
}
