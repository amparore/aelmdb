/* Aggregate integrity oracle.
 *
 * This file is intentionally textually included by mdb.c when
 * MDB_DEBUG_AGG_INTEGRITY is enabled.  It depends on private MDB_page,
 * MDB_node, MDB_db, MDB_cursor and aggregate helpers and is not a separately
 * compiled translation unit.
 *
 * The oracle is recursive by design and belongs only to diagnostics/tests.  It
 * never participates in production aggregate maintenance.
 */

#if MDB_DEBUG_AGG_INTEGRITY

static int
mdb_agg_debug_subtree(MDB_cursor *mc, pgno_t pgno, MDB_aggval *out);

static int
mdb_agg_debug_primary_leaf(MDB_cursor *mc, MDB_page *mp, MDB_aggval *out)
{
	MDB_aggval total, one;
	uint16_t agg = DB_AGGFLAGS(mc->mc_db);
	unsigned int i;
	int rc;

	if (!IS_LEAF(mp) || IS_LEAF2(mp))
		return MDB_CORRUPTED;
	mdb_aggval_zero(&total);
	for (i = 0; i < NUMKEYS(mp); ++i) {
		MDB_node *node = NODEPTR(mp, i);

		if ((mc->mc_db->md_flags & MDB_DUPSORT) &&
			(node->mn_flags & (F_DUPDATA|F_SUBDATA)) == (F_DUPDATA|F_SUBDATA)) {
			MDB_cursor *dc;
			MDB_aggval dup_tree, dup_db;

			rc = mdb_xcursor_init1(mc, node);
			if (rc)
				return rc;
			dc = &mc->mc_xcursor->mx_cursor;
			if (dc->mc_db->md_root == P_INVALID || !dc->mc_db->md_entries)
				return MDB_CORRUPTED;
			rc = mdb_agg_debug_subtree(dc, dc->mc_db->md_root, &dup_tree);
			if (rc)
				return rc;
			mdb_db_get_aggval(dc->mc_db, &dup_db);
			if (!mdb_aggval_equal(DB_AGGFLAGS(dc->mc_db), &dup_tree, &dup_db))
				return MDB_CORRUPTED;
			one = dup_tree;
			if (agg & MDB_AGG_KEYS)
				one.keys = 1;
		} else {
			rc = mdb_primary_leaf_node_agg(mc, mp, node, &one);
			if (rc)
				return rc;
		}
		rc = mdb_aggval_add(agg, &total, &one);
		if (rc)
			return rc;
	}
	*out = total;
	return MDB_SUCCESS;
}

static int
mdb_agg_debug_subtree(MDB_cursor *mc, pgno_t pgno, MDB_aggval *out)
{
	MDB_page *mp;
	MDB_aggval total, child, stored;
	uint16_t agg;
	unsigned int i;
	int rc;

	rc = mdb_page_get(mc, pgno, &mp, NULL);
	if (rc)
		return rc;
	rc = mdb_page_check_agg_schema(mc, mp);
	if (rc)
		return rc;
	agg = DB_AGGFLAGS(mc->mc_db);

	if (IS_LEAF(mp)) {
		if ((mc->mc_flags & C_SUB) || IS_SUBP(mp))
			return mdb_page_agg_local(mc, mp, out);
		if (mc->mc_db->md_flags & MDB_DUPSORT)
			return mdb_agg_debug_primary_leaf(mc, mp, out);
		return mdb_page_agg_local(mc, mp, out);
	}
	if (!IS_BRANCH(mp))
		return MDB_CORRUPTED;

	mdb_aggval_zero(&total);
	for (i = 0; i < NUMKEYS(mp); ++i) {
		MDB_node *node = NODEPTR(mp, i);
		rc = mdb_agg_debug_subtree(mc, NODEPGNO(node), &child);
		if (rc)
			return rc;
		mdb_node_get_aggval(mp, node, &stored);
		if (!mdb_aggval_equal(agg, &stored, &child))
			return MDB_CORRUPTED;
		rc = mdb_aggval_add(agg, &total, &child);
		if (rc)
			return rc;
	}
	*out = total;
	return MDB_SUCCESS;
}

int ESECT
mdb_agg_check_integrity(MDB_txn *txn, MDB_dbi dbi)
{
	MDB_cursor mc;
	MDB_xcursor mx;
	MDB_db *db;
	MDB_aggval tree, descriptor, zero;
	int rc;

	if (!txn || !TXN_DBI_EXIST(txn, dbi, DB_USRVALID))
		return EINVAL;
	if (txn->mt_flags & MDB_TXN_BLOCKED)
		return MDB_BAD_TXN;
	if (txn->mt_dbflags[dbi] & DB_STALE) {
		rc = mdb_dbi_refresh_stale_snapshot(txn, dbi);
		if (rc)
			return rc;
	}
	db = &txn->mt_dbs[dbi];
	if (!DB_AGGFLAGS(db))
		return MDB_INCOMPATIBLE;
	rc = mdb_dbi_check_tree_layout(txn, db);
	if (rc)
		return rc;

	mdb_db_get_aggval(db, &descriptor);
	if (db->md_root == P_INVALID) {
		mdb_aggval_zero(&zero);
		return mdb_aggval_equal(DB_AGGFLAGS(db), &descriptor, &zero) ?
			MDB_SUCCESS : MDB_CORRUPTED;
	}

	mdb_cursor_init(&mc, txn, dbi, &mx);
	rc = mdb_agg_debug_subtree(&mc, db->md_root, &tree);
	if (rc)
		return rc;
	return mdb_aggval_equal(DB_AGGFLAGS(db), &tree, &descriptor) ?
		MDB_SUCCESS : MDB_CORRUPTED;
}

#endif /* MDB_DEBUG_AGG_INTEGRITY */
