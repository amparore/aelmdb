/* Aggregate maintenance internals.
 *
 * This file is textually included by mdb.c.  It deliberately uses LMDB
 * private page/cursor types while keeping aggregate semantics and settlement
 * out of the mutation-engine body.
 */

/* Build the contribution of one logical record from the selected hash source.
 * Counts are one for each enabled component; the hashsum slice is copied only
 * when requested.  The caller folds this value into a page-local total. */
static int
mdb_agg_record_contribution(uint16_t agg, int hash_offset,
	const MDB_val *hash_source, MDB_aggval *out)
{
	MDB_aggval val;
	int rc;

	mdb_aggval_zero(&val);
	if (agg & MDB_AGG_ENTRIES)
		val.entries = 1;
	if (agg & MDB_AGG_KEYS)
		val.keys = 1;
	if (agg & MDB_AGG_HASHSUM) {
		rc = mdb_agg_hash_slice(hash_source, hash_offset, val.hashsum);
		if (rc)
			return rc;
	}
	*out = val;
	return MDB_SUCCESS;
}

/* Read the aggregate totals stored in an MDB_db descriptor.  md_entries is
 * LMDB's existing authoritative logical-entry count; the remaining fields are
 * the semantic totals introduced by aggregate maintenance. */
static inline void
mdb_db_get_aggval(const MDB_db *db, MDB_aggval *out)
{
	uint16_t agg = DB_AGGFLAGS(db);

	mdb_aggval_zero(out);
	if (agg & MDB_AGG_ENTRIES)
		out->entries = (uint64_t)db->md_entries;
	if (agg & MDB_AGG_KEYS)
		out->keys = (uint64_t)db->md_keys;
	if (agg & MDB_AGG_HASHSUM)
		memcpy(out->hashsum, db->md_hashsum, MDB_HASH_SIZE);
}

/* Fold one duplicate-tree leaf page.  Duplicate values are represented as
 * keys of the duplicate tree, so both ENTRIES and KEYS count every item and
 * the hash source is always the duplicate value itself.  This helper also
 * handles inline duplicate subpages; bounded_bytes is their containing node
 * size and is zero for an ordinary full page. */
static int
mdb_dup_leaf_agg_local(MDB_cursor *mc, const MDB_page *mp,
	size_t bounded_bytes, MDB_aggval *out)
{
	MDB_aggval total, one;
	uint16_t agg;
	unsigned int i, nkeys;
	int rc;

	if (!mc || !mc->mc_db || !mp || !out)
		return EINVAL;
	if (!IS_LEAF(mp) || IS_OVERFLOW(mp))
		return MDB_CORRUPTED;
	if (PAGE_AGGFLAGS(mp) != 0)
		return MDB_CORRUPTED;
	if (bounded_bytes) {
		if (bounded_bytes < PAGEHDRSZ ||
			MP_LOWER(mp) > MP_UPPER(mp) ||
			(size_t)MP_LOWER(mp) + PAGEBASE > bounded_bytes ||
			(size_t)MP_UPPER(mp) + PAGEBASE > bounded_bytes)
			return MDB_CORRUPTED;
	}

	agg = DB_AGGFLAGS(mc->mc_db);
	mdb_aggval_zero(&total);
	nkeys = NUMKEYS(mp);
	if (IS_LEAF2(mp)) {
		size_t ksize = MP_PAD(mp);

		if (nkeys && !ksize)
			return MDB_CORRUPTED;
		if (bounded_bytes && nkeys &&
			(size_t)PAGEHDRSZ + (size_t)nkeys * ksize > bounded_bytes)
			return MDB_CORRUPTED;
		for (i = 0; i < nkeys; ++i) {
			MDB_val value;
			value.mv_size = ksize;
			value.mv_data = LEAF2KEY(mp, i, ksize);
			rc = mdb_agg_record_contribution(agg,
				mc->mc_db->md_hash_offset, &value, &one);
			if (rc)
				return rc;
			rc = mdb_aggval_add(agg, &total, &one);
			if (rc)
				return rc;
		}
	} else {
		for (i = 0; i < nkeys; ++i) {
			MDB_node *node;
			MDB_val value;

			if (bounded_bytes) {
				size_t off = (size_t)MP_PTRS(mp)[i] + PAGEBASE;
				if (off > bounded_bytes || NODESIZE > bounded_bytes - off)
					return MDB_CORRUPTED;
			}
			node = NODEPTR(mp, i);
			/* Duplicate-tree records are flat keys with empty data.  Nested
			 * duplicate/subDB/overflow forms are not valid here. */
			if (node->mn_flags & (F_BIGDATA|F_DUPDATA|F_SUBDATA))
				return MDB_CORRUPTED;
			if (bounded_bytes) {
				size_t off = (size_t)((const char *)node - (const char *)mp);
				size_t need = EVEN(NODESIZE + EVEN((size_t)node->mn_ksize) +
					(size_t)NODEDSZ(node));
				if (need > bounded_bytes - off)
					return MDB_CORRUPTED;
			}
			value.mv_size = node->mn_ksize;
			value.mv_data = NODEKEY(mp, node);
			rc = mdb_agg_record_contribution(agg,
				mc->mc_db->md_hash_offset, &value, &one);
			if (rc)
				return rc;
			rc = mdb_aggval_add(agg, &total, &one);
			if (rc)
				return rc;
		}
	}

	*out = total;
	return MDB_SUCCESS;
}

/* Compute the contribution of one node in a primary-tree leaf. */
static int
mdb_primary_leaf_node_agg(MDB_cursor *mc, const MDB_page *mp,
	MDB_node *node, MDB_aggval *out)
{
	MDB_aggval val;
	uint16_t agg = DB_AGGFLAGS(mc->mc_db);
	MDB_val source;
	int rc;

	if (mc->mc_db->md_flags & MDB_DUPSORT) {
		if (mc->mc_db->md_flags & MDB_AGG_HASHSOURCE_FROM_KEY)
			return MDB_INCOMPATIBLE;
		if (node->mn_flags & F_DUPDATA) {
			if (node->mn_flags & F_SUBDATA) {
				MDB_db dupdb;

				if (NODEDSZ(node) != sizeof(MDB_db))
					return MDB_CORRUPTED;
				memcpy(&dupdb, NODEDATA(node), sizeof(dupdb));
				/* A persistent duplicate DB uses the parent's physical
				 * aggregate schema, but never key-source semantics. */
				if (DB_AGGSCHEMAFLAGS(&dupdb) != agg)
					return MDB_INCOMPATIBLE;
				if ((agg & MDB_AGG_HASHSUM) &&
					dupdb.md_hash_offset != mc->mc_db->md_hash_offset)
					return MDB_INCOMPATIBLE;
				if (dupdb.md_entries == 0)
					return MDB_CORRUPTED;
				mdb_db_get_aggval(&dupdb, &val);
			} else {
				MDB_page *subp = NODEDATA(node);

				if (NODEDSZ(node) < PAGEHDRSZ || !IS_SUBP(subp))
					return MDB_CORRUPTED;
				rc = mdb_dup_leaf_agg_local(mc, subp, NODEDSZ(node), &val);
				if (rc)
					return rc;
				if (NUMKEYS(subp) == 0)
					return MDB_CORRUPTED;
			}

			/* The primary tree represents the whole dupset as one distinct
			 * key, while preserving duplicate multiplicity and hashsum. */
			if (agg & MDB_AGG_KEYS)
				val.keys = 1;
			*out = val;
			return MDB_SUCCESS;
		}
	}

	/* Plain records, and the singleton representation of a DUPSORT key. */
	if (node->mn_flags & F_DUPDATA)
		return MDB_CORRUPTED;
	if ((agg & MDB_AGG_HASHSUM) &&
		(mc->mc_db->md_flags & MDB_AGG_HASHSOURCE_FROM_KEY)) {
		source.mv_size = node->mn_ksize;
		source.mv_data = NODEKEY(mp, node);
	} else if (agg & MDB_AGG_HASHSUM) {
		MDB_cursor scan = *mc;

		MC_SET_OVPG(&scan, NULL);
		rc = mdb_node_read(&scan, node, &source);
		if (rc)
			return rc;
		rc = mdb_agg_record_contribution(agg,
			mc->mc_db->md_hash_offset, &source, &val);
#ifdef MDB_VL32
		if (MC_OVPG(&scan))
			MDB_PAGE_UNREF(scan.mc_txn, MC_OVPG(&scan));
#endif
		if (rc)
			return rc;
		*out = val;
		return MDB_SUCCESS;
	} else {
		source.mv_size = 0;
		source.mv_data = NULL;
	}

	return mdb_agg_record_contribution(agg, mc->mc_db->md_hash_offset,
		&source, out);
}

/* Contribution of one item in a flat aggregate tree.  For C_SUB the duplicate
 * value is physically stored as the key of the duplicate tree, including the
 * LEAF2 representation used by MDB_DUPFIXED; in a primary tree it is the
 * contribution of the leaf node (a whole dupset counts as one key). */
static int
mdb_flat_leaf_item_agg(MDB_cursor *mc, const MDB_page *mp,
	unsigned int index, MDB_aggval *out)
{
	MDB_val source;
	MDB_node *node;
	uint16_t agg;

	if (!mc || !mc->mc_db || !mp || !out || !IS_LEAF(mp) ||
		index >= NUMKEYS(mp))
		return MDB_CORRUPTED;
	agg = DB_AGGFLAGS(mc->mc_db);
	if (mc->mc_flags & C_SUB) {
		if (IS_LEAF2(mp)) {
			source.mv_size = mc->mc_db->md_pad;
			source.mv_data = LEAF2KEY(mp, index, source.mv_size);
		} else {
			node = NODEPTR(mp, index);
			if (node->mn_flags & (F_BIGDATA|F_DUPDATA|F_SUBDATA))
				return MDB_CORRUPTED;
			source.mv_size = NODEKSZ(node);
			source.mv_data = NODEKEY(mp, node);
		}
		return mdb_agg_record_contribution(agg, mc->mc_db->md_hash_offset,
			&source, out);
	}
	if (IS_LEAF2(mp))
		return MDB_CORRUPTED;
	node = NODEPTR(mp, index);
	return mdb_primary_leaf_node_agg(mc, mp, node, out);
}

/* Exact aggregate of one final local page.  This is deliberately non-recursive:
 * branch pages fold already-maintained child prefixes; primary leaf pages fold
 * only their local records; inline duplicate subpages are scanned locally; a
 * persistent duplicate subDB contributes through its embedded MDB_db totals. */
static int
mdb_page_agg_local(MDB_cursor *mc, const MDB_page *mp, MDB_aggval *out)
{
	MDB_aggval total, one;
	uint16_t agg;
	unsigned int i, nkeys;
	int rc;

	if (!mc || !mc->mc_db || !mp || !out)
		return EINVAL;
	rc = mdb_page_check_agg_schema(mc, mp);
	if (rc)
		return rc;
	agg = DB_AGGFLAGS(mc->mc_db);
	mdb_aggval_zero(&total);

	if (IS_BRANCH(mp)) {
		nkeys = NUMKEYS(mp);
		for (i = 0; i < nkeys; ++i) {
			mdb_node_get_aggval(mp, NODEPTR(mp, i), &one);
			rc = mdb_aggval_add(agg, &total, &one);
			if (rc)
				return rc;
		}
	} else if (IS_LEAF(mp)) {
		if (IS_SUBP(mp) || (mc->mc_flags & C_SUB))
			return mdb_dup_leaf_agg_local(mc, mp, 0, out);
		if (IS_LEAF2(mp))
			return MDB_CORRUPTED;
		nkeys = NUMKEYS(mp);
		for (i = 0; i < nkeys; ++i) {
			rc = mdb_primary_leaf_node_agg(mc, mp, NODEPTR(mp, i), &one);
			if (rc)
				return rc;
			rc = mdb_aggval_add(agg, &total, &one);
			if (rc)
				return rc;
		}
	} else {
		return MDB_CORRUPTED;
	}

	*out = total;
	return MDB_SUCCESS;
}

/** @defgroup aggregates Aggregate maintenance
 *
 * A put or delete on an aggregate DB changes one item: its contribution
 * goes from \b before to \b after (#MDB_aggchange; zero for an absent
 * item).  The item is the node at the cursor: a record of a plain DB, a
 * whole dupset in a DUPSORT primary tree (a dupset counts as one key), one
 * duplicate value in a duplicate tree.  The contributions are read from the
 * final state of the node (#mdb_flat_leaf_item_agg()), before and after the
 * operation.
 *
 * The unwind records how far up the tree changed structurally
 * (#MDB_txn.mt_unwind_prefix).  When the operation is complete,
 * #mdb_agg_settle() walks the cursor's path once, bottom-up:
 *
 *	- a link inside a structurally changed page gets the exact aggregate of
 *	  its child, folded from that final page (#mdb_page_agg_local(): one
 *	  page, never a subtree);
 *	- a link in the untouched prefix gets the logical change;
 *	- the DB totals (md_keys, md_hashsum) get the logical change once;
 *	  md_entries stays LMDB's.
 *
 * Links that a structural step creates or moves off the cursor's path (the
 * other half of a split, the sibling of a move) are made exact by that step
 * when its pages are final.  Nothing is recomputed recursively.
 * @{
 */

/** Settle the aggregates of a completed put or delete (see @ref aggregates).
 * @param[in] mc the cursor of the operation, on its final path
 * @param[in] ch the logical change of the item
 */
static int
mdb_agg_settle(MDB_cursor *mc, const MDB_aggchange *ch)
{
	uint16_t agg = DB_AGGFLAGS(mc->mc_db), tot;
	MDB_aggval v;
	int prefix, i, rc;

	if (!agg)
		return MDB_SUCCESS;
	prefix = (mc->mc_flags & C_SUB) ? mc->mc_txn->mt_unwind_sub_prefix :
		mc->mc_txn->mt_unwind_prefix;
	if (mc->mc_snum && (mc->mc_flags & C_INITIALIZED)) {
		int top = mc->mc_top;
		if (prefix > top)
			prefix = top;
		/* structurally changed levels: exact values from final pages */
		for (i = top; i > prefix; i--) {
			MDB_page *pp = mc->mc_pg[i-1];
			if (!IS_BRANCH(pp) || !(pp->mp_flags & P_DIRTY) ||
				mc->mc_ki[i-1] >= NUMKEYS(pp))
				return MDB_PROBLEM;
			rc = mdb_page_agg_local(mc, mc->mc_pg[i], &v);
			if (rc)
				return rc;
			mdb_node_set_aggval(pp, NODEPTR(pp, mc->mc_ki[i-1]), &v);
		}
		/* the untouched prefix: the logical change */
		for (i = prefix; i >= 1; i--) {
			MDB_page *pp = mc->mc_pg[i-1];
			MDB_node *node;
			if (!IS_BRANCH(pp) || !(pp->mp_flags & P_DIRTY) ||
				mc->mc_ki[i-1] >= NUMKEYS(pp))
				return MDB_PROBLEM;
			node = NODEPTR(pp, mc->mc_ki[i-1]);
			mdb_node_get_aggval(pp, node, &v);
			rc = mdb_aggval_apply_change(agg, &v, ch);
			if (rc)
				return rc;
			mdb_node_set_aggval(pp, node, &v);
		}
	}
	/* DB totals: md_entries is maintained by LMDB itself.  An inline
	 * duplicate sub-page has no persistent totals of its own: the primary
	 * leaf reads it directly. */
	tot = agg & (MDB_AGG_KEYS|MDB_AGG_HASHSUM);
	if (!tot)
		return MDB_SUCCESS;
	if ((mc->mc_flags & C_SUB) && mc->mc_snum && IS_SUBP(mc->mc_pg[0]))
		return MDB_SUCCESS;
	mdb_db_get_aggval(mc->mc_db, &v);
	rc = mdb_aggval_apply_change(tot, &v, ch);
	if (rc)
		return rc;
	if (tot & MDB_AGG_KEYS)
		mc->mc_db->md_keys = (mdb_size_t)v.keys;
	if (tot & MDB_AGG_HASHSUM)
		memcpy(mc->mc_db->md_hashsum, v.hashsum, MDB_HASH_SIZE);
	return MDB_SUCCESS;
}

/** The contribution of the item at the cursor (zero past the end) */
static int
mdb_agg_item_at(MDB_cursor *mc, MDB_aggval *out)
{
	MDB_page *mp;

	mdb_aggval_zero(out);
	if (!mc->mc_snum || !(mc->mc_flags & C_INITIALIZED))
		return MDB_SUCCESS;
	mp = mc->mc_pg[mc->mc_top];
	if (mc->mc_ki[mc->mc_top] >= NUMKEYS(mp))
		return MDB_SUCCESS;
	return mdb_flat_leaf_item_agg(mc, mp, mc->mc_ki[mc->mc_top], out);
}

/** Validate the aggregate inputs of a put before anything changes: the hash
 * slice must fit its source, and a value-source hashsum cannot be taken from
 * #MDB_RESERVE storage, which the caller fills after the put.  LMDB does not
 * support #MDB_RESERVE on DUPSORT databases: an aggregate DUPSORT DB refuses
 * it explicitly.  #MDB_WRITEMAP needs no rule: put data are final at call
 * time, and applications may not modify returned values.
 */
static int
mdb_agg_put_check(MDB_cursor *mc, MDB_val *key, MDB_val *data,
	unsigned int flags)
{
	uint16_t agg = DB_AGGFLAGS(mc->mc_db);
	uint8_t slice[MDB_HASH_SIZE];
	MDB_val *src;

	if ((flags & MDB_RESERVE) && !(mc->mc_flags & C_SUB) &&
		(mc->mc_db->md_flags & MDB_DUPSORT))
		return MDB_INCOMPATIBLE;
	if (!(agg & MDB_AGG_HASHSUM))
		return MDB_SUCCESS;
	if (mc->mc_flags & C_SUB) {
		src = key;		/* the duplicate value is the key here */
	} else if (mc->mc_db->md_flags & MDB_AGG_HASHSOURCE_FROM_KEY) {
		src = key;
	} else {
		if (flags & MDB_RESERVE)
			return MDB_INCOMPATIBLE;
		src = data;		/* with MDB_MULTIPLE: the first item */
	}
	return mdb_agg_hash_slice(src, mc->mc_db->md_hash_offset, slice);
}

/** @} */

/** The exact aggregate of a final page, as a branch link blob
 * (see @ref aggregates) */
static int
mdb_agg_page_blob(MDB_cursor *mc, const MDB_page *mp, MDB_aggblob *out)
{
	MDB_aggval v;
	int rc = mdb_page_agg_local(mc, mp, &v);
	if (rc)
		return rc;
	mdb_aggval_to_blob(DB_AGGFLAGS(mc->mc_db), &v, out);
	return MDB_SUCCESS;
}

/** Make link @p ki of branch page @p pp exact for its final child @p child */
static int
mdb_agg_link_exact(MDB_cursor *mc, MDB_page *pp, indx_t ki,
	const MDB_page *child)
{
	MDB_aggblob b;
	int rc;

	if (!IS_BRANCH(pp) || ki >= NUMKEYS(pp) || !(pp->mp_flags & P_DIRTY) ||
		NODEPGNO(NODEPTR(pp, ki)) != child->mp_pgno)
		return MDB_PROBLEM;
	if ((rc = mdb_agg_page_blob(mc, child, &b)))
		return rc;
	mdb_node_set_agg_blob(pp, NODEPTR(pp, ki), &b);
	return MDB_SUCCESS;
}
