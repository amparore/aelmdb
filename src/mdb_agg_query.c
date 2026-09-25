/* Aggregate query layer.
 *
 * This file is textually included by mdb.c after the aggregate
 * maintenance/configuration helpers have been defined.  It intentionally uses
 * LMDB-private cursor/page internals without exporting them through lmdb.h.
 *
 * Scope: totals, prefix/range, rank/select, and entry-rank cursor positioning.
 * Window-relative range caching is part of this query layer.
 */

#define MDB_RANGE_ALLOWED_FLAGS (MDB_RANGE_LOWER_INCL|MDB_RANGE_UPPER_INCL)
#define MDB_PREFIX_ALLOWED_FLAGS (MDB_AGG_PREFIX_INCL)

static int
mdb_agg_query_db(MDB_txn *txn, MDB_dbi dbi, MDB_db **dbp)
{
	int rc;

	if (!txn || !dbp || !TXN_DBI_EXIST(txn, dbi, DB_USRVALID))
		return EINVAL;
	if (txn->mt_flags & MDB_TXN_BLOCKED)
		return MDB_BAD_TXN;
	if (txn->mt_dbflags[dbi] & DB_STALE) {
		rc = mdb_dbi_refresh_stale_snapshot(txn, dbi);
		if (rc)
			return rc;
	}
	*dbp = &txn->mt_dbs[dbi];
	return MDB_SUCCESS;
}

static void
mdb_agg_query_zero_public(unsigned flags, MDB_agg *out)
{
	if (!out)
		return;
	out->mv_flags = flags;
	out->mv_agg_entries = 0;
	out->mv_agg_keys = 0;
	memset(out->mv_agg_hashes, 0, MDB_HASH_SIZE);
}

static void
mdb_agg_query_from_internal(uint16_t agg, unsigned schema,
	const MDB_aggval *in, MDB_agg *out)
{
	mdb_agg_query_zero_public(schema, out);
	if (!in)
		return;
	if (agg & MDB_AGG_ENTRIES)
		out->mv_agg_entries = in->entries;
	if (agg & MDB_AGG_KEYS)
		out->mv_agg_keys = in->keys;
	if (agg & MDB_AGG_HASHSUM)
		memcpy(out->mv_agg_hashes, in->hashsum, MDB_HASH_SIZE);
}

static int
mdb_agg_query_totals_internal(MDB_txn *txn, MDB_dbi dbi, MDB_aggval *out)
{
	MDB_db *db;
	int rc;

	if (!out)
		return EINVAL;
	rc = mdb_agg_query_db(txn, dbi, &db);
	if (rc)
		return rc;
	if (!DB_AGGFLAGS(db))
		return MDB_INCOMPATIBLE;
	mdb_db_get_aggval(db, out);
	return MDB_SUCCESS;
}

/* The totals of the tree of a cursor.  An inline duplicate sub-page has no
 * persistent totals of its own (the sub-cursor's MDB_db carries only LMDB's
 * md_entries): they are the fold of the sub-page, one page. */
static int
mdb_agg_query_cursor_totals(MDB_cursor *mc, MDB_aggval *out)
{
	if (!mc || !mc->mc_db || !out)
		return EINVAL;
	if (!DB_AGGFLAGS(mc->mc_db))
		return MDB_INCOMPATIBLE;
	if ((mc->mc_flags & C_SUB) && mc->mc_snum && mc->mc_pg[0] &&
		IS_SUBP(mc->mc_pg[0]))
		return mdb_dup_leaf_agg_local(mc, mc->mc_pg[0], 0, out);
	mdb_db_get_aggval(mc->mc_db, out);
	return MDB_SUCCESS;
}

static int
mdb_agg_query_leaf_item(MDB_cursor *mc, MDB_page *mp, indx_t index,
	MDB_aggval *out)
{
	if (!mc || !mp || !out || index >= NUMKEYS(mp))
		return MDB_CORRUPTED;
	if (mc->mc_flags & C_SUB)
		return mdb_flat_leaf_item_agg(mc, mp, index, out);
	if (!IS_LEAF(mp) || IS_LEAF2(mp))
		return MDB_CORRUPTED;
	return mdb_primary_leaf_node_agg(mc, mp, NODEPTR(mp, index), out);
}


static int
mdb_agg_query_branch_weight(const MDB_page *mp, const MDB_node *node,
	MDB_agg_weight weight, uint64_t *out)
{
	uint16_t agg;
	const uint8_t *p;

	if (!mp || !node || !out || !IS_BRANCH(mp))
		return MDB_CORRUPTED;
	agg = PAGE_AGGFLAGS(mp);
	if (!(agg & (uint16_t)weight))
		return MDB_INCOMPATIBLE;
	p = (const uint8_t *)node->mn_data;
	if (weight == MDB_AGG_WEIGHT_KEYS && (agg & MDB_AGG_ENTRIES))
		p += MDB_AGG_U64_SIZE;
	memcpy(out, p, sizeof(*out));
	return MDB_SUCCESS;
}

static int
mdb_agg_query_leaf_weight(MDB_cursor *mc, MDB_page *mp, indx_t index,
	MDB_agg_weight weight, uint64_t *out)
{
	MDB_node *node;

	if (!mc || !mp || !out || !IS_LEAF(mp) || index >= NUMKEYS(mp))
		return MDB_CORRUPTED;
	if (weight == MDB_AGG_WEIGHT_KEYS) {
		*out = 1;
		return MDB_SUCCESS;
	}
	if ((mc->mc_flags & C_SUB) || !(mc->mc_db->md_flags & MDB_DUPSORT)) {
		*out = 1;
		return MDB_SUCCESS;
	}
	if (IS_LEAF2(mp))
		return MDB_CORRUPTED;
	node = NODEPTR(mp, index);
	if (!(node->mn_flags & F_DUPDATA)) {
		*out = 1;
		return MDB_SUCCESS;
	}
	if (node->mn_flags & F_SUBDATA) {
		MDB_db dupdb;
		if (NODEDSZ(node) != sizeof(MDB_db))
			return MDB_CORRUPTED;
		memcpy(&dupdb, NODEDATA(node), sizeof(dupdb));
		if (dupdb.md_entries == 0)
			return MDB_CORRUPTED;
		*out = (uint64_t)dupdb.md_entries;
		return MDB_SUCCESS;
	}
	{
		MDB_page *subp = NODEDATA(node);
		if (NODEDSZ(node) < PAGEHDRSZ || !IS_SUBP(subp) || NUMKEYS(subp) == 0)
			return MDB_CORRUPTED;
		*out = NUMKEYS(subp);
	}
	return MDB_SUCCESS;
}

static int
mdb_agg_query_add_value_inplace(uint16_t agg, MDB_aggval *dst,
	const MDB_aggval *src)
{
	if ((agg & MDB_AGG_ENTRIES) && UINT64_MAX - dst->entries < src->entries)
		return MDB_CORRUPTED;
	if ((agg & MDB_AGG_KEYS) && UINT64_MAX - dst->keys < src->keys)
		return MDB_CORRUPTED;
	if (agg & MDB_AGG_ENTRIES) dst->entries += src->entries;
	if (agg & MDB_AGG_KEYS) dst->keys += src->keys;
	if (agg & MDB_AGG_HASHSUM) mdb_agg_hash_add(dst->hashsum, src->hashsum);
	return MDB_SUCCESS;
}

static int
mdb_agg_query_add_branch_prefix(MDB_cursor *mc, MDB_page *mp, indx_t index,
	MDB_aggval *out)
{
	MDB_node *node;
	const uint8_t *p;
	uint64_t v;
	uint16_t agg;

	if (!mc || !mp || !out || !IS_BRANCH(mp) || index >= NUMKEYS(mp))
		return MDB_CORRUPTED;
	agg = DB_AGGFLAGS(mc->mc_db);
	if (PAGE_AGGFLAGS(mp) != agg)
		return MDB_CORRUPTED;
	node = NODEPTR(mp, index);
	p = (const uint8_t *)node->mn_data;
	if (agg & MDB_AGG_ENTRIES) {
		memcpy(&v, p, sizeof(v)); p += sizeof(v);
		if (UINT64_MAX - out->entries < v) return MDB_CORRUPTED;
		out->entries += v;
	}
	if (agg & MDB_AGG_KEYS) {
		memcpy(&v, p, sizeof(v)); p += sizeof(v);
		if (UINT64_MAX - out->keys < v) return MDB_CORRUPTED;
		out->keys += v;
	}
	if (agg & MDB_AGG_HASHSUM)
		mdb_agg_hash_add(out->hashsum, p);
	return MDB_SUCCESS;
}

/* Position at the first key >= *key (MDB_SET_RANGE semantics) and report
 * whether the key found equals the requested one.
 *
 * mdb_cursor_set() treats a non-NULL exactp as an exact-match request (MDB_SET
 * semantics): for any absent key it returns MDB_NOTFOUND even though the
 * cursor is already positioned on the next greater key.  The public
 * mdb_cursor_get() therefore passes NULL for MDB_SET_RANGE, and so must we;
 * exactness is derived afterwards with the cursor's own comparator, which is
 * the duplicate comparator on a duplicate subcursor (C5). */
static int
mdb_agg_query_set_range(MDB_cursor *mc, MDB_val *key, MDB_val *data,
	int *exactp)
{
	MDB_val requested = *key;
	int rc = mdb_cursor_set(mc, key, data, MDB_SET_RANGE, NULL);

	*exactp = rc == MDB_SUCCESS && mc->mc_dbx->md_cmp(&requested, key) == 0;
	return rc;
}

/* Aggregate all keys before key, or through key when incl is set and key exists.
 * Works both on a primary tree and on an initialized duplicate subcursor. */
static int
mdb_agg_query_prefix_key_cursor(MDB_cursor *cur, const MDB_val *key,
	unsigned incl, MDB_aggval *out, int *any_out)
{
	MDB_db *db;
	MDB_val k, v;
	MDB_aggval total, one;
	uint16_t agg;
	int exact = 0, rc;
	indx_t lvl;

	if (any_out)
		*any_out = 0;
	if (!cur || !key || !out || !cur->mc_db)
		return EINVAL;
	if (cur->mc_txn->mt_flags & MDB_TXN_BLOCKED)
		return MDB_BAD_TXN;

	db = cur->mc_db;
	agg = DB_AGGFLAGS(db);
	if (!agg)
		return MDB_INCOMPATIBLE;
	mdb_aggval_zero(&total);

	k = *key;
	v.mv_size = 0;
	v.mv_data = NULL;
	rc = mdb_agg_query_set_range(cur, &k, &v, &exact);
	if (rc == MDB_NOTFOUND) {
		rc = mdb_agg_query_cursor_totals(cur, &total);
		if (!rc && any_out)
			*any_out = db->md_entries != 0;
		if (!rc)
			*out = total;
		return rc;
	}
	if (rc)
		return rc;

	/* Fully covered subtrees to the left of the search path. */
	for (lvl = 0; lvl < cur->mc_top; ++lvl) {
		MDB_page *pp = cur->mc_pg[lvl];
		indx_t j, stop = cur->mc_ki[lvl];
		if (!pp || !IS_BRANCH(pp) || PAGE_AGGFLAGS(pp) != agg)
			return MDB_CORRUPTED;
		for (j = 0; j < stop; ++j) {
			rc = mdb_agg_query_add_branch_prefix(cur, pp, j, &total);
			if (rc)
				return rc;
		}
	}

	/* The boundary leaf contributes only its local prefix. */
	{
		MDB_page *lp = cur->mc_pg[cur->mc_top];
		indx_t i, stop = cur->mc_ki[cur->mc_top];
		indx_t limit = stop + ((incl && exact && stop < NUMKEYS(lp)) ? 1 : 0);
		int any = limit != 0;

		if (!lp || !IS_LEAF(lp))
			return MDB_CORRUPTED;
		for (i = 0; i < limit; ++i) {
			rc = mdb_agg_query_leaf_item(cur, lp, i, &one);
			if (rc)
				return rc;
			rc = mdb_agg_query_add_value_inplace(agg, &total, &one);
			if (rc)
				return rc;
		}
		if (!any) {
			for (lvl = 0; lvl < cur->mc_top; ++lvl) {
				if (cur->mc_ki[lvl] != 0) {
					any = 1;
					break;
				}
			}
		}
		if (any_out)
			*any_out = any;
	}

	*out = total;
	return MDB_SUCCESS;
}

static int
mdb_agg_query_prefix_key_internal(MDB_txn *txn, MDB_dbi dbi,
	const MDB_val *key, unsigned incl, MDB_aggval *out)
{
	MDB_cursor *cur = NULL;
	int rc;

	if (!key || !out)
		return EINVAL;
	rc = mdb_cursor_open(txn, dbi, &cur);
	if (rc)
		return rc;
	rc = mdb_agg_query_prefix_key_cursor(cur, key, incl, out, NULL);
	mdb_cursor_close(cur);
	return rc;
}

/* Record-order prefix for DUPSORT.  The primary-key prefix is followed by the
 * requested portion of the duplicate set. */
static int
mdb_agg_query_prefix_kd_internal(MDB_txn *txn, MDB_dbi dbi,
	const MDB_val *key, const MDB_val *data, unsigned incl, MDB_aggval *out)
{
	MDB_db *db;
	MDB_cursor *mc = NULL;
	MDB_page *lp;
	MDB_node *node;
	MDB_val ck, cv;
	MDB_aggval part;
	uint16_t agg;
	int rc;

	if (!key || !out)
		return EINVAL;
	rc = mdb_agg_query_db(txn, dbi, &db);
	if (rc)
		return rc;
	agg = DB_AGGFLAGS(db);
	if (!agg)
		return MDB_INCOMPATIBLE;
	if (!data || !(db->md_flags & MDB_DUPSORT))
		return mdb_agg_query_prefix_key_internal(txn, dbi, key, incl, out);

	rc = mdb_cursor_open(txn, dbi, &mc);
	if (rc)
		return rc;

	/* Base contribution: all primary keys strictly before key. */
	rc = mdb_agg_query_prefix_key_cursor(mc, key, 0, out, NULL);
	if (rc)
		goto done;

	ck.mv_size = cv.mv_size = 0;
	ck.mv_data = cv.mv_data = NULL;
	rc = mdb_cursor_get(mc, &ck, &cv, MDB_GET_CURRENT);
	if (rc == MDB_NOTFOUND) {
		rc = MDB_SUCCESS;
		goto done;
	}
	if (rc)
		goto done;
	if (mdb_cmp(txn, dbi, &ck, key) != 0) {
		rc = MDB_SUCCESS;
		goto done;
	}

	lp = mc->mc_pg[mc->mc_top];
	if (!lp || !IS_LEAF(lp) || IS_LEAF2(lp)) {
		rc = MDB_CORRUPTED;
		goto done;
	}
	node = NODEPTR(lp, mc->mc_ki[mc->mc_top]);

	if (node->mn_flags & F_DUPDATA) {
		MDB_val dk = *data;
		int any = 0;

		if (!mc->mc_xcursor) {
			rc = MDB_PROBLEM;
			goto done;
		}
		rc = mdb_xcursor_init1(mc, node);
		if (rc)
			goto done;
		rc = mdb_agg_query_prefix_key_cursor(&mc->mc_xcursor->mx_cursor,
			&dk, incl, &part, &any);
		if (rc)
			goto done;

		/* The duplicate tree counts every value as a key; the primary tree
		 * exposes the touched dupset as exactly one distinct primary key. */
		if (agg & MDB_AGG_KEYS)
			part.keys = any ? 1 : 0;
		rc = mdb_aggval_add(agg, out, &part);
	} else {
		MDB_val actual;
		int cmp;

		rc = mdb_node_read(mc, node, &actual);
		if (rc)
			goto done;
		cmp = mdb_dcmp(txn, dbi, &actual, data);
		if (cmp < 0 || (incl && cmp == 0)) {
			rc = mdb_primary_leaf_node_agg(mc, lp, node, &part);
			if (!rc)
				rc = mdb_aggval_add(agg, out, &part);
		}
	}

done:
	mdb_cursor_close(mc);
	return rc;
}

/* Number of elements preceding the current cursor position in the requested
 * aggregate weight. */
static int
mdb_agg_query_cursor_rank_base(MDB_cursor *mc, MDB_agg_weight weight,
	uint64_t *out)
{
	uint64_t rank = 0, add;
	uint16_t agg, wflag;
	int lvl, rc;
	indx_t j;

	if (!mc || !out || !(mc->mc_flags & C_INITIALIZED))
		return EINVAL;
	agg = DB_AGGFLAGS(mc->mc_db);
	wflag = (uint16_t)weight;
	if (!agg || !(agg & wflag))
		return MDB_INCOMPATIBLE;

	for (lvl = 0; lvl < (int)mc->mc_top; ++lvl) {
		MDB_page *pp = mc->mc_pg[lvl];
		indx_t ix = mc->mc_ki[lvl];
		if (!pp || !IS_BRANCH(pp) || PAGE_AGGFLAGS(pp) != agg)
			return MDB_CORRUPTED;
		for (j = 0; j < ix; ++j) {
			rc = mdb_agg_query_branch_weight(pp, NODEPTR(pp, j), weight, &add);
			if (rc)
				return rc;
			if (UINT64_MAX - rank < add)
				return MDB_CORRUPTED;
			rank += add;
		}
	}

	{
		MDB_page *lp = mc->mc_pg[mc->mc_top];
		indx_t ix = mc->mc_ki[mc->mc_top];
		if (!lp || !IS_LEAF(lp))
			return MDB_CORRUPTED;
		for (j = 0; j < ix; ++j) {
			rc = mdb_agg_query_leaf_weight(mc, lp, j, weight, &add);
			if (rc)
				return rc;
			if (UINT64_MAX - rank < add)
				return MDB_CORRUPTED;
			rank += add;
		}
	}

	*out = rank;
	return MDB_SUCCESS;
}

/* Locate the primary/flat leaf item containing rank.  For an entry-weight
 * DUPSORT primary query, dup_index receives the offset inside that dupset. */
static int
mdb_agg_query_cursor_rank_search(MDB_cursor *mc, MDB_agg_weight weight,
	uint64_t rank, uint64_t *dup_index)
{
	MDB_page *mp;
	uint64_t remaining = rank, total, cnt;
	uint16_t agg, wflag;
	int rc;

	if (!mc || !mc->mc_db)
		return EINVAL;
	if (mc->mc_txn->mt_flags & MDB_TXN_BLOCKED)
		return MDB_BAD_TXN;
	agg = DB_AGGFLAGS(mc->mc_db);
	wflag = (uint16_t)weight;
	if (!agg || !(agg & wflag))
		return MDB_INCOMPATIBLE;
	if (mc->mc_db->md_root == P_INVALID)
		return MDB_NOTFOUND;

	if (weight == MDB_AGG_WEIGHT_ENTRIES) {
		total = (uint64_t)mc->mc_db->md_entries;
	} else {
		MDB_aggval t;
		rc = mdb_agg_query_cursor_totals(mc, &t);
		if (rc)
			return rc;
		total = t.keys;
	}
	if (rank >= total)
		return MDB_NOTFOUND;

	MDB_CURSOR_UNREF(mc, 0);
	mc->mc_flags &= ~(C_INITIALIZED|C_EOF);
	mc->mc_top = 0;
	mc->mc_snum = 0;
	rc = mdb_page_search(mc, NULL, MDB_PS_ROOTONLY);
	if (rc)
		return rc;
	mp = mc->mc_pg[mc->mc_top];

	while (IS_BRANCH(mp)) {
		indx_t i, n = NUMKEYS(mp);
		MDB_node *node = NULL;
		if (PAGE_AGGFLAGS(mp) != agg)
			return MDB_CORRUPTED;
		for (i = 0; i < n; ++i) {
			rc = mdb_agg_query_branch_weight(mp, NODEPTR(mp, i), weight, &cnt);
			if (rc)
				return rc;
			if (remaining < cnt)
				break;
			remaining -= cnt;
		}
		if (i == n)
			return MDB_CORRUPTED;
		mc->mc_ki[mc->mc_top] = i;
		node = NODEPTR(mp, i);
		rc = mdb_page_get(mc, NODEPGNO(node), &mp, NULL);
		if (rc)
			return rc;
		rc = mdb_cursor_push(mc, mp);
		if (rc)
			return rc;
	}

	if (!IS_LEAF(mp))
		return MDB_CORRUPTED;
	{
		indx_t i, n = NUMKEYS(mp);
		for (i = 0; i < n; ++i) {
			rc = mdb_agg_query_leaf_weight(mc, mp, i, weight, &cnt);
			if (rc)
				return rc;
			if (remaining < cnt)
				break;
			remaining -= cnt;
		}
		if (i == n)
			return MDB_CORRUPTED;
		mc->mc_ki[mc->mc_top] = i;
	}
	if (dup_index)
		*dup_index = (weight == MDB_AGG_WEIGHT_ENTRIES &&
			!(mc->mc_flags & C_SUB) && (mc->mc_db->md_flags & MDB_DUPSORT)) ?
			remaining : 0;
	mc->mc_flags |= C_INITIALIZED;
	return MDB_SUCCESS;
}

/* Locate the first and last record of a record-order DUPSORT range, then use
 * key ranks to obtain the number of distinct primary keys touched. */
static int
mdb_agg_query_range_kd_keys(MDB_txn *txn, MDB_dbi dbi,
	const MDB_val *low_key, const MDB_val *low_data, unsigned lower_incl,
	const MDB_val *high_key, const MDB_val *high_data, unsigned upper_incl,
	uint64_t *out_keys)
{
	MDB_cursor *mc = NULL;
	MDB_val ks, ds, kl, dl;
	uint64_t first_rank, last_rank;
	int rc = MDB_SUCCESS, exact = 0;
	int have_start = 0, have_last = 0;

	if (!txn || !out_keys)
		return EINVAL;
	*out_keys = 0;

	if (low_key && high_key) {
		int kc = mdb_cmp(txn, dbi, low_key, high_key);
		if (kc > 0)
			return MDB_SUCCESS;
		if (kc == 0 && low_data && high_data &&
			mdb_dcmp(txn, dbi, low_data, high_data) > 0)
			return MDB_SUCCESS;
	}

	rc = mdb_cursor_open(txn, dbi, &mc);
	if (rc)
		return rc;

	/* First record in range. */
	{
		MDB_val k = {0}, d = {0};
		if (!low_key) {
			rc = mdb_cursor_get(mc, &k, &d, MDB_FIRST);
			if (!rc) { ks = k; ds = d; have_start = 1; }
		} else if (low_data) {
			MDB_val requested = *low_data;
			k = *low_key; d = *low_data;
			rc = mdb_cursor_set(mc, &k, &d, MDB_GET_BOTH_RANGE, &exact);
			if (rc == MDB_NOTFOUND) {
				MDB_val kt = *low_key, dt = {0, NULL};
				int exactk = 0;
				rc = mdb_agg_query_set_range(mc, &kt, &dt, &exactk);
				if (!rc && exactk)
					rc = mdb_cursor_get(mc, &kt, &dt, MDB_NEXT_NODUP);
				if (!rc) { ks = kt; ds = dt; have_start = 1; }
			} else if (!rc) {
				int deq = mdb_dcmp(txn, dbi, &requested, &d) == 0;
				if (!lower_incl && deq)
					rc = mdb_cursor_get(mc, &k, &d, MDB_NEXT);
				if (!rc) { ks = k; ds = d; have_start = 1; }
			}
		} else {
			k = *low_key;
			rc = mdb_agg_query_set_range(mc, &k, &d, &exact);
			if (!rc && !lower_incl && exact)
				rc = mdb_cursor_get(mc, &k, &d, MDB_NEXT_NODUP);
			if (!rc) { ks = k; ds = d; have_start = 1; }
		}
		if (rc == MDB_NOTFOUND)
			rc = MDB_SUCCESS;
	}
	if (rc || !have_start)
		goto done;

	/* Save the distinct-key rank of the first touched key. */
	rc = mdb_agg_query_cursor_rank_base(mc, MDB_AGG_WEIGHT_KEYS, &first_rank);
	if (rc)
		goto done;

	/* First record strictly after the upper bound; PREV gives last in range. */
	{
		MDB_val k = {0}, d = {0};
		int after_last = 0;

		if (!high_key) {
			after_last = 1;
		} else if (high_data) {
			MDB_val requested = *high_data;
			k = *high_key; d = *high_data;
			rc = mdb_cursor_set(mc, &k, &d, MDB_GET_BOTH_RANGE, &exact);
			if (rc == MDB_NOTFOUND) {
				MDB_val kt = *high_key, dt = {0, NULL};
				int exactk = 0;
				rc = mdb_agg_query_set_range(mc, &kt, &dt, &exactk);
				if (rc == MDB_NOTFOUND) { after_last = 1; rc = MDB_SUCCESS; }
				else if (!rc && exactk) {
					rc = mdb_cursor_get(mc, &kt, &dt, MDB_NEXT_NODUP);
					if (rc == MDB_NOTFOUND) { after_last = 1; rc = MDB_SUCCESS; }
				}
			} else if (!rc && upper_incl &&
				mdb_dcmp(txn, dbi, &requested, &d) == 0) {
				rc = mdb_cursor_get(mc, &k, &d, MDB_NEXT);
				if (rc == MDB_NOTFOUND) { after_last = 1; rc = MDB_SUCCESS; }
			}
		} else {
			k = *high_key;
			rc = mdb_agg_query_set_range(mc, &k, &d, &exact);
			if (rc == MDB_NOTFOUND) { after_last = 1; rc = MDB_SUCCESS; }
			else if (!rc && upper_incl && exact) {
				rc = mdb_cursor_get(mc, &k, &d, MDB_NEXT_NODUP);
				if (rc == MDB_NOTFOUND) { after_last = 1; rc = MDB_SUCCESS; }
			}
		}
		if (rc)
			goto done;
		if (after_last)
			rc = mdb_cursor_get(mc, &k, &d, MDB_LAST);
		else
			rc = mdb_cursor_get(mc, &k, &d, MDB_PREV);
		if (!rc) { kl = k; dl = d; have_last = 1; }
		else if (rc == MDB_NOTFOUND) rc = MDB_SUCCESS;
	}
	if (rc || !have_last)
		goto done;

	/* The located record interval must be nonempty in record order. */
	{
		int kc = mdb_cmp(txn, dbi, &ks, &kl);
		if (kc > 0 || (kc == 0 && mdb_dcmp(txn, dbi, &ds, &dl) > 0))
			goto done;
	}

	/* Reposition to the last record and obtain its distinct-key rank. */
	{
		MDB_val k = kl, d = dl;
		rc = mdb_cursor_get(mc, &k, &d, MDB_GET_BOTH);
		if (rc)
			goto done;
	}
	rc = mdb_agg_query_cursor_rank_base(mc, MDB_AGG_WEIGHT_KEYS, &last_rank);
	if (rc)
		goto done;
	if (last_rank < first_rank) {
		rc = MDB_CORRUPTED;
		goto done;
	}
	*out_keys = last_rank - first_rank + 1;

done:
	mdb_cursor_close(mc);
	return rc;
}

static int
mdb_agg_query_select_cursor(MDB_cursor *mc, MDB_agg_weight weight,
	uint64_t rank, MDB_val *key, MDB_val *data, uint64_t *dup_index_out)
{
	MDB_page *mp;
	MDB_node *node;
	MDB_val dk, dd;
	uint64_t dup_index = 0;
	int rc;

	if (!mc || !key || !data)
		return EINVAL;
	rc = mdb_agg_query_cursor_rank_search(mc, weight, rank, &dup_index);
	if (rc)
		return rc;

	mp = mc->mc_pg[mc->mc_top];
	if (!(mc->mc_flags & C_SUB) && !IS_LEAF2(mp) &&
		(mc->mc_db->md_flags & MDB_DUPSORT)) {
		node = NODEPTR(mp, mc->mc_ki[mc->mc_top]);
		if (node->mn_flags & F_DUPDATA) {
			rc = mdb_xcursor_init1(mc, node);
			if (rc)
				return rc;
			dk.mv_size = dd.mv_size = 0;
			dk.mv_data = dd.mv_data = NULL;
			rc = mdb_cursor_get(&mc->mc_xcursor->mx_cursor, &dk, &dd, MDB_FIRST);
			if (rc)
				return rc;
			if (weight == MDB_AGG_WEIGHT_ENTRIES && dup_index) {
				rc = mdb_agg_query_cursor_rank_search(&mc->mc_xcursor->mx_cursor,
					MDB_AGG_WEIGHT_ENTRIES, dup_index, NULL);
				if (rc)
					return rc;
			}
		}
	}
	if (dup_index_out)
		*dup_index_out = weight == MDB_AGG_WEIGHT_ENTRIES ? dup_index : 0;
	return mdb_cursor_get(mc, key, data, MDB_GET_CURRENT);
}

int ESECT
mdb_agg_totals(MDB_txn *txn, MDB_dbi dbi, MDB_agg *out)
{
	MDB_db *db;
	MDB_aggval total;
	uint16_t agg;
	int rc;

	if (!out)
		return EINVAL;
	rc = mdb_agg_query_db(txn, dbi, &db);
	if (rc)
		return rc;
	agg = DB_AGGFLAGS(db);
	if (!agg)
		return MDB_INCOMPATIBLE;
	mdb_db_get_aggval(db, &total);
	mdb_agg_query_from_internal(agg, DB_AGGSCHEMAFLAGS(db), &total, out);
	return MDB_SUCCESS;
}

int ESECT
mdb_agg_prefix(MDB_txn *txn, MDB_dbi dbi,
	const MDB_val *key, const MDB_val *data, unsigned flags, MDB_agg *out)
{
	MDB_db *db;
	MDB_aggval a;
	uint16_t agg;
	int rc;

	if (!key || !out || (flags & ~MDB_PREFIX_ALLOWED_FLAGS))
		return EINVAL;
	rc = mdb_agg_query_db(txn, dbi, &db);
	if (rc)
		return rc;
	agg = DB_AGGFLAGS(db);
	if (!agg)
		return MDB_INCOMPATIBLE;
	rc = mdb_agg_query_prefix_kd_internal(txn, dbi, key, data,
		!!(flags & MDB_AGG_PREFIX_INCL), &a);
	if (rc)
		return rc;
	mdb_agg_query_from_internal(agg, DB_AGGSCHEMAFLAGS(db), &a, out);
	return MDB_SUCCESS;
}

static int
mdb_agg_query_range_key(MDB_txn *txn, MDB_dbi dbi,
	const MDB_val *low, const MDB_val *high, unsigned flags, MDB_agg *out)
{
	MDB_db *db;
	MDB_aggval hi, lo, result;
	uint16_t agg;
	unsigned lower_incl = !!(flags & MDB_RANGE_LOWER_INCL);
	unsigned upper_incl = !!(flags & MDB_RANGE_UPPER_INCL);
	int rc;

	rc = mdb_agg_query_db(txn, dbi, &db);
	if (rc)
		return rc;
	agg = DB_AGGFLAGS(db);
	if (!agg)
		return MDB_INCOMPATIBLE;
	if (low && high) {
		int cmp = mdb_cmp(txn, dbi, low, high);
		if (cmp > 0 || (cmp == 0 && !(lower_incl && upper_incl))) {
			mdb_agg_query_zero_public(DB_AGGSCHEMAFLAGS(db), out);
			return MDB_SUCCESS;
		}
	}

	if (!high)
		rc = mdb_agg_query_totals_internal(txn, dbi, &hi);
	else
		rc = mdb_agg_query_prefix_key_internal(txn, dbi, high, upper_incl, &hi);
	if (rc)
		return rc;
	if (!low)
		mdb_aggval_zero(&lo);
	else {
		rc = mdb_agg_query_prefix_key_internal(txn, dbi, low, !lower_incl, &lo);
		if (rc)
			return rc;
	}
	result = hi;
	rc = mdb_aggval_sub(agg, &result, &lo);
	if (rc)
		return rc;
	mdb_agg_query_from_internal(agg, DB_AGGSCHEMAFLAGS(db), &result, out);
	return MDB_SUCCESS;
}

int ESECT
mdb_agg_range(MDB_txn *txn, MDB_dbi dbi,
	const MDB_val *low_key, const MDB_val *low_data,
	const MDB_val *high_key, const MDB_val *high_data,
	unsigned flags, MDB_agg *out)
{
	MDB_db *db;
	MDB_aggval hi, lo, result;
	uint16_t agg, diffmask;
	unsigned lower_incl, upper_incl;
	int rc;

	if (!out || (flags & ~MDB_RANGE_ALLOWED_FLAGS))
		return EINVAL;
	rc = mdb_agg_query_db(txn, dbi, &db);
	if (rc)
		return rc;
	agg = DB_AGGFLAGS(db);
	if (!agg)
		return MDB_INCOMPATIBLE;
	if (!(db->md_flags & MDB_DUPSORT) || (!low_data && !high_data))
		return mdb_agg_query_range_key(txn, dbi, low_key, high_key, flags, out);

	lower_incl = !!(flags & MDB_RANGE_LOWER_INCL);
	upper_incl = !!(flags & MDB_RANGE_UPPER_INCL);
	if (low_key && high_key) {
		int kc = mdb_cmp(txn, dbi, low_key, high_key);
		if (kc > 0 || (kc == 0 && low_data && high_data &&
			mdb_dcmp(txn, dbi, low_data, high_data) > 0)) {
			mdb_agg_query_zero_public(DB_AGGSCHEMAFLAGS(db), out);
			return MDB_SUCCESS;
		}
	}

	if (!high_key)
		rc = mdb_agg_query_totals_internal(txn, dbi, &hi);
	else if (high_data)
		rc = mdb_agg_query_prefix_kd_internal(txn, dbi, high_key, high_data,
			upper_incl, &hi);
	else
		rc = mdb_agg_query_prefix_key_internal(txn, dbi, high_key, upper_incl, &hi);
	if (rc)
		return rc;

	if (!low_key)
		mdb_aggval_zero(&lo);
	else if (low_data)
		rc = mdb_agg_query_prefix_kd_internal(txn, dbi, low_key, low_data,
			!lower_incl, &lo);
	else
		rc = mdb_agg_query_prefix_key_internal(txn, dbi, low_key, !lower_incl, &lo);
	if (rc)
		return rc;

	/* Prefix-difference is exact for ENTRIES and HASHSUM.  KEYS is computed
	 * separately because a bound may cut through a duplicate set. */
	result = hi;
	diffmask = agg & (MDB_AGG_ENTRIES|MDB_AGG_HASHSUM);
	rc = mdb_aggval_sub(diffmask, &result, &lo);
	if (rc) {
		mdb_agg_query_zero_public(DB_AGGSCHEMAFLAGS(db), out);
		return MDB_SUCCESS;
	}
	if (agg & MDB_AGG_KEYS) {
		rc = mdb_agg_query_range_kd_keys(txn, dbi,
			low_key, low_data, lower_incl,
			high_key, high_data, upper_incl, &result.keys);
		if (rc)
			return rc;
	}
	mdb_agg_query_from_internal(agg, DB_AGGSCHEMAFLAGS(db), &result, out);
	return MDB_SUCCESS;
}

int ESECT
mdb_agg_select(MDB_txn *txn, MDB_dbi dbi, MDB_agg_weight weight,
	uint64_t rank, MDB_val *key, MDB_val *data, uint64_t *dup_index)
{
	MDB_db *db;
	MDB_cursor *mc = NULL;
	uint16_t wflag = (uint16_t)weight;
	int rc;

	if (!key || !data || (weight != MDB_AGG_WEIGHT_ENTRIES &&
		weight != MDB_AGG_WEIGHT_KEYS))
		return EINVAL;
	rc = mdb_agg_query_db(txn, dbi, &db);
	if (rc)
		return rc;
	if (!(DB_AGGFLAGS(db) & wflag))
		return MDB_INCOMPATIBLE;
	rc = mdb_cursor_open(txn, dbi, &mc);
	if (rc)
		return rc;
	rc = mdb_agg_query_select_cursor(mc, weight, rank, key, data, dup_index);
	mdb_cursor_close(mc);
	return rc;
}

int ESECT
mdb_agg_cursor_seek_rank(MDB_cursor *mc, uint64_t rank,
	MDB_val *key, MDB_val *data)
{
	MDB_val kd = {0, NULL}, dd = {0, NULL};
	return mdb_agg_query_select_cursor(mc, MDB_AGG_WEIGHT_ENTRIES, rank,
		key ? key : &kd, data ? data : &dd, NULL);
}

int ESECT
mdb_agg_rank(MDB_txn *txn, MDB_dbi dbi,
	MDB_val *key, MDB_val *data, MDB_agg_weight weight,
	unsigned flags, uint64_t *rank, uint64_t *dup_index)
{
	MDB_db *db;
	MDB_cursor *mc = NULL;
	MDB_val k, d;
	uint64_t base = 0, dup = 0;
	uint16_t wflag = (uint16_t)weight;
	int rc;

	if (!key || !data || !rank ||
		(weight != MDB_AGG_WEIGHT_ENTRIES && weight != MDB_AGG_WEIGHT_KEYS) ||
		(flags != MDB_AGG_RANK_EXACT && flags != MDB_AGG_RANK_SET_RANGE))
		return EINVAL;
	rc = mdb_agg_query_db(txn, dbi, &db);
	if (rc)
		return rc;
	if (!(DB_AGGFLAGS(db) & wflag))
		return MDB_INCOMPATIBLE;
	if (weight == MDB_AGG_WEIGHT_KEYS && (data->mv_size || data->mv_data))
		return EINVAL;
	if (!(db->md_flags & MDB_DUPSORT) && (data->mv_size || data->mv_data))
		return EINVAL;

	rc = mdb_cursor_open(txn, dbi, &mc);
	if (rc)
		return rc;
	k = *key;
	d = *data;

	if ((db->md_flags & MDB_DUPSORT) && weight == MDB_AGG_WEIGHT_ENTRIES) {
		int data_unspecified = d.mv_size == 0 && d.mv_data == NULL;
		if (flags == MDB_AGG_RANK_EXACT) {
			if (data_unspecified) { rc = EINVAL; goto done; }
			rc = mdb_cursor_get(mc, &k, &d, MDB_GET_BOTH);
		} else if (data_unspecified) {
			rc = mdb_cursor_get(mc, &k, &d, MDB_SET_RANGE);
		} else {
			MDB_val kin = k;
			rc = mdb_cursor_get(mc, &k, &d, MDB_GET_BOTH_RANGE);
			if (rc == MDB_NOTFOUND) {
				k = kin; d.mv_size = 0; d.mv_data = NULL;
				rc = mdb_cursor_get(mc, &k, &d, MDB_SET_RANGE);
				if (!rc && mdb_cmp(txn, dbi, &k, &kin) == 0)
					rc = mdb_cursor_get(mc, &k, &d, MDB_NEXT_NODUP);
			}
		}
	} else {
		rc = mdb_cursor_get(mc, &k, &d,
			flags == MDB_AGG_RANK_SET_RANGE ? MDB_SET_RANGE : MDB_SET_KEY);
	}
	if (rc)
		goto done;

	*key = k;
	*data = d;
	rc = mdb_agg_query_cursor_rank_base(mc, weight, &base);
	if (rc)
		goto done;

	if (weight == MDB_AGG_WEIGHT_ENTRIES && (db->md_flags & MDB_DUPSORT)) {
		MDB_page *mp = mc->mc_pg[mc->mc_top];
		if (!IS_LEAF2(mp)) {
			MDB_node *node = NODEPTR(mp, mc->mc_ki[mc->mc_top]);
			if (node->mn_flags & F_DUPDATA) {
				MDB_cursor *xc;
				MDB_val dk = d, dd = {0, NULL};
				rc = mdb_xcursor_init1(mc, node);
				if (rc)
					goto done;
				xc = &mc->mc_xcursor->mx_cursor;
				rc = mdb_cursor_get(xc, &dk, &dd, MDB_SET_KEY);
				if (rc)
					goto done;
				rc = mdb_agg_query_cursor_rank_base(xc,
					MDB_AGG_WEIGHT_ENTRIES, &dup);
				if (rc)
					goto done;
			}
		}
	}
	*rank = base + dup;
	if (dup_index)
		*dup_index = weight == MDB_AGG_WEIGHT_ENTRIES ? dup : 0;

done:
	mdb_cursor_close(mc);
	return rc;
}


/* Prefix aggregate for the first `rank` logical entries.  This helper uses
 * the public rank/select ordering contract but remains entirely read-only. */
static int
mdb_agg_query_prefix_rank_entries_fallback(MDB_txn *txn, MDB_dbi dbi,
	uint64_t rank, MDB_aggval *out)
{
	MDB_db *db;
	MDB_cursor *mc = NULL;
	MDB_val key = {0, NULL}, data = {0, NULL};
	MDB_aggval total;
	int rc;

	if (!out)
		return EINVAL;
	rc = mdb_agg_query_db(txn, dbi, &db);
	if (rc)
		return rc;
	if (!(DB_AGGFLAGS(db) & MDB_AGG_ENTRIES))
		return MDB_INCOMPATIBLE;
	if (rank == 0) {
		mdb_aggval_zero(out);
		return MDB_SUCCESS;
	}
	rc = mdb_agg_query_totals_internal(txn, dbi, &total);
	if (rc)
		return rc;
	if (rank > total.entries)
		return MDB_NOTFOUND;
	if (rank == total.entries) {
		*out = total;
		return MDB_SUCCESS;
	}
	rc = mdb_cursor_open(txn, dbi, &mc);
	if (rc)
		return rc;
	rc = mdb_agg_query_select_cursor(mc, MDB_AGG_WEIGHT_ENTRIES,
		rank, &key, &data, NULL);
	if (!rc) {
		if (db->md_flags & MDB_DUPSORT)
			rc = mdb_agg_query_prefix_kd_internal(txn, dbi, &key, &data, 0, out);
		else
			rc = mdb_agg_query_prefix_key_internal(txn, dbi, &key, 0, out);
	}
	mdb_cursor_close(mc);
	return rc;
}

/* Direct rank-space prefix for plain trees.  DUPSORT keeps the generic
 * fallback because a boundary may end inside a duplicate container. */
static int
mdb_agg_query_prefix_rank_entries_internal(MDB_txn *txn, MDB_dbi dbi,
	uint64_t rank, MDB_aggval *out)
{
	MDB_db *db;
	MDB_cursor *mc = NULL;
	MDB_page *mp;
	uint16_t agg;
	uint64_t remaining;
	int rc;

	if (!out) return EINVAL;
	rc = mdb_agg_query_db(txn, dbi, &db);
	if (rc) return rc;
	agg = DB_AGGFLAGS(db);
	if (!(agg & MDB_AGG_ENTRIES)) return MDB_INCOMPATIBLE;
	if (db->md_flags & MDB_DUPSORT)
		return mdb_agg_query_prefix_rank_entries_fallback(txn, dbi, rank, out);
	if (rank > (uint64_t)db->md_entries) return MDB_NOTFOUND;
	if (rank == 0) { mdb_aggval_zero(out); return MDB_SUCCESS; }
	if (rank == (uint64_t)db->md_entries) { mdb_db_get_aggval(db, out); return MDB_SUCCESS; }
	mdb_aggval_zero(out);
	rc = mdb_cursor_open(txn, dbi, &mc);
	if (rc) return rc;
	rc = mdb_page_search(mc, NULL, MDB_PS_ROOTONLY);
	if (rc) goto done;
	remaining = rank;
	mp = mc->mc_pg[mc->mc_top];
	while (IS_BRANCH(mp)) {
		indx_t i, n = NUMKEYS(mp);
		if (PAGE_AGGFLAGS(mp) != agg) { rc = MDB_CORRUPTED; goto done; }
		for (i = 0; i < n; ++i) {
			MDB_node *node = NODEPTR(mp, i);
			uint64_t cnt;
			rc = mdb_agg_query_branch_weight(mp, node, MDB_AGG_WEIGHT_ENTRIES, &cnt);
			if (rc) goto done;
			if (remaining >= cnt) {
				rc = mdb_agg_query_add_branch_prefix(mc, mp, i, out);
				if (rc) goto done;
				remaining -= cnt;
				if (!remaining) { rc = MDB_SUCCESS; goto done; }
			} else {
				mc->mc_ki[mc->mc_top] = i;
				rc = mdb_page_get(mc, NODEPGNO(node), &mp, NULL);
				if (rc) goto done;
				rc = mdb_cursor_push(mc, mp);
				if (rc) goto done;
				break;
			}
		}
		if (i == n) { rc = MDB_CORRUPTED; goto done; }
	}
	if (!IS_LEAF(mp) || IS_LEAF2(mp)) { rc = MDB_CORRUPTED; goto done; }
	{
		indx_t i, n = NUMKEYS(mp);
		for (i = 0; i < n && remaining; ++i) {
			MDB_aggval one;
			rc = mdb_agg_query_leaf_item(mc, mp, i, &one);
			if (rc) goto done;
			rc = mdb_agg_query_add_value_inplace(agg, out, &one);
			if (rc) goto done;
			--remaining;
		}
		if (remaining) rc = MDB_CORRUPTED;
	}
done:
	mdb_cursor_close(mc);
	return rc;
}

/* Initialize/reuse the absolute entry-rank interval represented by a window.
 * The caller contract requires the same bounds when reusing a populated
 * descriptor; flags/schema changes force reinitialization. */
static int
mdb_agg_window_ensure(MDB_txn *txn, MDB_dbi dbi,
	const MDB_val *low_key, const MDB_val *low_data,
	const MDB_val *high_key, const MDB_val *high_data,
	unsigned range_flags, MDB_agg_window *window,
	uint64_t *total_entries, uint64_t *abs_lo, uint64_t *abs_hi,
	MDB_aggval *a_lo, int *have_lo, MDB_aggval *a_hi, int *have_hi,
	int *did_init)
{
	MDB_db *db;
	uint16_t agg;
	unsigned schema, lower_incl, upper_incl;
	MDB_aggval tot, lo, hi;
	uint64_t tentries, alo, ahi;
	int rc, hlo = 0, hhi = 0;

	if (!window)
		return EINVAL;
	rc = mdb_agg_query_db(txn, dbi, &db);
	if (rc)
		return rc;
	if (range_flags & ~MDB_RANGE_ALLOWED_FLAGS)
		return EINVAL;
	agg = DB_AGGFLAGS(db);
	schema = DB_AGGSCHEMAFLAGS(db);
	if (!agg || !(agg & MDB_AGG_ENTRIES))
		return MDB_INCOMPATIBLE;
	lower_incl = !!(range_flags & MDB_RANGE_LOWER_INCL);
	upper_incl = !!(range_flags & MDB_RANGE_UPPER_INCL);

	if (!(window->mv_flags && window->mv_flags == schema &&
		window->mv_range_flags == range_flags)) {
		rc = mdb_agg_query_totals_internal(txn, dbi, &tot);
		if (rc) return rc;
		tentries = tot.entries;
		if (!high_key) { hi = tot; hhi = 1; }
		else if (high_data) {
			rc = mdb_agg_query_prefix_kd_internal(txn, dbi, high_key, high_data,
				upper_incl, &hi); if (rc) return rc; hhi = 1;
		} else {
			rc = mdb_agg_query_prefix_key_internal(txn, dbi, high_key,
				upper_incl, &hi); if (rc) return rc; hhi = 1;
		}
		if (!low_key) { mdb_aggval_zero(&lo); hlo = 1; }
		else if (low_data) {
			rc = mdb_agg_query_prefix_kd_internal(txn, dbi, low_key, low_data,
				!lower_incl, &lo); if (rc) return rc; hlo = 1;
		} else {
			rc = mdb_agg_query_prefix_key_internal(txn, dbi, low_key,
				!lower_incl, &lo); if (rc) return rc; hlo = 1;
		}
		alo = lo.entries; ahi = hi.entries;
		if (ahi < alo) ahi = alo;
		window->mv_flags = schema;
		window->mv_range_flags = range_flags;
		window->mv_total_entries = tentries;
		window->mv_abs_lo = alo;
		window->mv_abs_hi = ahi;
		if (did_init) *did_init = 1;
	} else {
		tentries = window->mv_total_entries;
		alo = window->mv_abs_lo;
		ahi = window->mv_abs_hi;
		if (did_init) *did_init = 0;
	}
	if (total_entries) *total_entries = tentries;
	if (abs_lo) *abs_lo = alo;
	if (abs_hi) *abs_hi = ahi;
	if (did_init && *did_init) {
		if (a_lo) *a_lo = lo;
		if (a_hi) *a_hi = hi;
		if (have_lo) *have_lo = hlo;
		if (have_hi) *have_hi = hhi;
	} else {
		if (have_lo) *have_lo = 0;
		if (have_hi) *have_hi = 0;
	}
	return MDB_SUCCESS;
}

int ESECT
mdb_agg_window_aggregate(MDB_txn *txn, MDB_dbi dbi,
	const MDB_val *low_key, const MDB_val *low_data,
	const MDB_val *high_key, const MDB_val *high_data,
	unsigned range_flags, MDB_agg_window *window,
	uint64_t rel_begin, uint64_t rel_end, MDB_agg *out)
{
	MDB_db *db;
	uint16_t agg;
	unsigned schema;
	uint64_t abs_lo, abs_hi, total_entries, win_size, abs_begin, abs_end;
	MDB_aggval a_lo, a_hi, p_begin, p_end;
	int have_lo = 0, have_hi = 0, did_init = 0, rc;

	if (!out || !window)
		return EINVAL;
	rc = mdb_agg_query_db(txn, dbi, &db);
	if (rc) return rc;
	if (range_flags & ~MDB_RANGE_ALLOWED_FLAGS) return EINVAL;
	agg = DB_AGGFLAGS(db); schema = DB_AGGSCHEMAFLAGS(db);
	if (!agg || !(agg & MDB_AGG_ENTRIES)) return MDB_INCOMPATIBLE;
	rc = mdb_agg_window_ensure(txn, dbi, low_key, low_data, high_key, high_data,
		range_flags, window, &total_entries, &abs_lo, &abs_hi,
		&a_lo, &have_lo, &a_hi, &have_hi, &did_init);
	if (rc) return rc;
	win_size = abs_hi - abs_lo;
	if (rel_end == MDB_AGG_WINDOW_END) rel_end = win_size;
	if (rel_begin > rel_end || rel_end > win_size) return EINVAL;
	abs_begin = abs_lo + rel_begin; abs_end = abs_lo + rel_end;
	mdb_agg_query_zero_public(schema, out);
	if (abs_begin == abs_end) return MDB_SUCCESS;
	if (did_init && have_lo && abs_begin == abs_lo) p_begin = a_lo;
	else { rc = mdb_agg_query_prefix_rank_entries_internal(txn, dbi, abs_begin, &p_begin); if (rc) return rc; }
	if (did_init && have_hi && abs_end == abs_hi) p_end = a_hi;
	else { rc = mdb_agg_query_prefix_rank_entries_internal(txn, dbi, abs_end, &p_end); if (rc) return rc; }
	out->mv_agg_entries = p_end.entries - p_begin.entries;
	if (agg & MDB_AGG_HASHSUM) {
		memcpy(out->mv_agg_hashes, p_end.hashsum, MDB_HASH_SIZE);
		mdb_agg_hash_sub(out->mv_agg_hashes, p_begin.hashsum);
	}
	if (agg & MDB_AGG_KEYS) {
		if (!(db->md_flags & MDB_DUPSORT)) {
			out->mv_agg_keys = p_end.keys - p_begin.keys;
		} else if (out->mv_agg_entries) {
			MDB_cursor *mc = NULL;
			MDB_val lk = {0,NULL}, ld = {0,NULL}, hk = {0,NULL}, hd = {0,NULL};
			uint64_t keys = 0;
			rc = mdb_cursor_open(txn, dbi, &mc); if (rc) return rc;
			rc = mdb_agg_query_select_cursor(mc, MDB_AGG_WEIGHT_ENTRIES, abs_begin,
				&lk, &ld, NULL);
			if (!rc && abs_end < total_entries) {
				rc = mdb_agg_query_select_cursor(mc, MDB_AGG_WEIGHT_ENTRIES, abs_end,
					&hk, &hd, NULL);
				if (!rc) rc = mdb_agg_query_range_kd_keys(txn, dbi, &lk, &ld, 1,
					&hk, &hd, 0, &keys);
			} else if (!rc) {
				rc = mdb_agg_query_range_kd_keys(txn, dbi, &lk, &ld, 1,
					NULL, NULL, 0, &keys);
			}
			mdb_cursor_close(mc);
			if (rc) return rc;
			out->mv_agg_keys = keys;
		}
	}
	return MDB_SUCCESS;
}

int ESECT
mdb_agg_window_rank(MDB_txn *txn, MDB_dbi dbi,
	const MDB_val *low_key, const MDB_val *low_data,
	const MDB_val *high_key, const MDB_val *high_data,
	unsigned range_flags, MDB_agg_window *window,
	const MDB_val *key, const MDB_val *data, uint64_t *rel_rank)
{
	MDB_db *db;
	uint64_t total_entries, abs_lo, abs_hi, abs_rank = 0, win_size;
	MDB_val k, d = {0,NULL};
	int rc, data_unspecified, key_only_fastpath;
	unsigned upper_incl;

	if (!window || !key || !rel_rank) return EINVAL;
	rc = mdb_agg_query_db(txn, dbi, &db); if (rc) return rc;
	if (!(DB_AGGFLAGS(db) & MDB_AGG_ENTRIES)) return MDB_INCOMPATIBLE;
	if (range_flags & ~MDB_RANGE_ALLOWED_FLAGS) return EINVAL;
	rc = mdb_agg_window_ensure(txn, dbi, low_key, low_data, high_key, high_data,
		range_flags, window, &total_entries, &abs_lo, &abs_hi,
		NULL, NULL, NULL, NULL, NULL);
	if (rc) return rc;
	win_size = abs_hi - abs_lo;
	if (!win_size) { *rel_rank = 0; return MDB_SUCCESS; }
	upper_incl = !!(range_flags & MDB_RANGE_UPPER_INCL);
	data_unspecified = !data || (!data->mv_size && !data->mv_data);
	key_only_fastpath = !low_data && !high_data && data_unspecified;
	if (key_only_fastpath) {
		if (low_key && mdb_cmp(txn, dbi, (MDB_val *)key, (MDB_val *)low_key) <= 0) {
			*rel_rank = 0; return MDB_SUCCESS;
		}
		if (high_key) {
			int c = mdb_cmp(txn, dbi, (MDB_val *)key, (MDB_val *)high_key);
			if (c > 0 || (c == 0 && !upper_incl)) { *rel_rank = win_size; return MDB_SUCCESS; }
		}
	}
	k = *key; if (data) d = *data;
	rc = mdb_agg_rank(txn, dbi, &k, &d, MDB_AGG_WEIGHT_ENTRIES,
		MDB_AGG_RANK_SET_RANGE, &abs_rank, NULL);
	if (rc == MDB_NOTFOUND) abs_rank = total_entries;
	else if (rc) return rc;
	if (abs_rank < abs_lo) abs_rank = abs_lo;
	if (abs_rank > abs_hi) abs_rank = abs_hi;
	*rel_rank = abs_rank - abs_lo;
	return MDB_SUCCESS;
}
