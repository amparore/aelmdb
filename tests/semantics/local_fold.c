#ifndef _GNU_SOURCE
#define _GNU_SOURCE 1
#endif
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <limits.h>

#include "aelmdb_mdb.c"

#define CHECK(cond) do { \
	if (!(cond)) { \
		fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, #cond); \
		return 1; \
	} \
} while (0)

#define TEST_PAGE_SIZE 4096u
#define TEST_SUBPAGE_SIZE 1536u

static void
fill_hash(uint8_t h[MDB_HASH_SIZE], unsigned seed)
{
	size_t i;
	for (i = 0; i < MDB_HASH_SIZE; ++i)
		h[i] = (uint8_t)(seed + 37u * (unsigned)i);
}

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

static void
test_page_init(void *buf, size_t bytes, uint16_t flags)
{
	MDB_page *mp = buf;
	memset(buf, 0, bytes);
	MP_FLAGS(mp) = flags;
	MP_LOWER(mp) = PAGEHDRSZ - PAGEBASE;
	MP_UPPER(mp) = (indx_t)(bytes - PAGEBASE);
}

static MDB_node *
test_add_node(MDB_page *mp, size_t bytes, const void *key, size_t ksize,
	const void *data, size_t dsize, unsigned flags, pgno_t pgno,
	const MDB_aggval *branch_agg)
{
	unsigned int indx = NUMKEYS(mp);
	size_t nsize = NODESIZE + mdb_agg_prefix_bytes(mp) + EVEN(ksize);
	MDB_node *node;

	if (IS_LEAF(mp))
		nsize += dsize;
	nsize = EVEN(nsize);
	if ((size_t)MP_UPPER(mp) < (size_t)MP_LOWER(mp) + sizeof(indx_t) + nsize)
		return NULL;
	MP_UPPER(mp) -= (indx_t)nsize;
	MP_PTRS(mp)[indx] = MP_UPPER(mp);
	MP_LOWER(mp) += sizeof(indx_t);
	node = NODEPTR(mp, indx);
	memset(node, 0, nsize);
	node->mn_ksize = (unsigned short)ksize;
	node->mn_flags = (unsigned short)flags;
	if (IS_LEAF(mp))
		SETDSZ(node, dsize);
	else
		SETPGNO(node, pgno);
	if (branch_agg)
		mdb_node_set_aggval(mp, node, branch_agg);
	if (ksize)
		memcpy(NODEKEY(mp, node), key, ksize);
	if (IS_LEAF(mp) && dsize)
		memcpy(NODEDATA(node), data, dsize);
	(void)bytes;
	return node;
}

static int
test_add_leaf2(MDB_page *mp, size_t bytes, const void *key, size_t ksize)
{
	unsigned int indx = NUMKEYS(mp);
	char *dst;

	if (MP_PAD(mp) != ksize || ksize < sizeof(indx_t))
		return 0;
	if ((size_t)PAGEHDRSZ + ((size_t)indx + 1) * ksize > bytes)
		return 0;
	dst = LEAF2KEY(mp, indx, ksize);
	memcpy(dst, key, ksize);
	MP_LOWER(mp) += sizeof(indx_t);
	MP_UPPER(mp) -= (indx_t)(ksize - sizeof(indx_t));
	return 1;
}

static int
add_expected_record(uint16_t agg, int offset, const void *p, size_t n,
	MDB_aggval *total)
{
	MDB_val source;
	MDB_aggval one;
	int rc;

	source.mv_data = (void *)p;
	source.mv_size = n;
	rc = mdb_agg_record_contribution(agg, offset, &source, &one);
	if (rc)
		return rc;
	return mdb_aggval_add(agg, total, &one);
}

static void
init_cursor(MDB_cursor *mc, MDB_db *db, unsigned flags)
{
	memset(mc, 0, sizeof(*mc));
	mc->mc_db = db;
	mc->mc_flags = flags;
}

static int
test_branch_local(void)
{
	const uint16_t agg = MDB_AGG_ENTRIES|MDB_AGG_KEYS|MDB_AGG_HASHSUM;
	unsigned char pagebuf[TEST_PAGE_SIZE];
	MDB_page *mp = (MDB_page *)pagebuf;
	MDB_db db;
	MDB_cursor mc;
	MDB_aggval a, b, c, expected, got, sentinel;
	uint8_t k = 1;

	memset(&db, 0, sizeof(db));
	db.md_flags = agg;
	init_cursor(&mc, &db, 0);
	test_page_init(pagebuf, sizeof(pagebuf), P_BRANCH|agg);
	mdb_aggval_zero(&a); mdb_aggval_zero(&b); mdb_aggval_zero(&c);
	a.entries = 3; a.keys = 2; fill_hash(a.hashsum, 3);
	b.entries = 5; b.keys = 4; fill_hash(b.hashsum, 11);
	c.entries = 7; c.keys = 6; fill_hash(c.hashsum, 29);
	CHECK(test_add_node(mp, sizeof(pagebuf), NULL, 0, NULL, 0, 0, 10, &a));
	CHECK(test_add_node(mp, sizeof(pagebuf), &k, 1, NULL, 0, 0, 11, &b));
	k = 2;
	CHECK(test_add_node(mp, sizeof(pagebuf), &k, 1, NULL, 0, 0, 12, &c));
	expected = a;
	CHECK(mdb_aggval_add(agg, &expected, &b) == MDB_SUCCESS);
	CHECK(mdb_aggval_add(agg, &expected, &c) == MDB_SUCCESS);
	CHECK(mdb_page_agg_local(&mc, mp, &got) == MDB_SUCCESS);
	CHECK(agg_equal(agg, &got, &expected));

	/* Count overflow is detected without publishing a partial output. */
	test_page_init(pagebuf, sizeof(pagebuf), P_BRANCH|agg);
	mdb_aggval_zero(&a); mdb_aggval_zero(&b);
	a.entries = UINT64_MAX; b.entries = 1;
	CHECK(test_add_node(mp, sizeof(pagebuf), NULL, 0, NULL, 0, 0, 10, &a));
	CHECK(test_add_node(mp, sizeof(pagebuf), &k, 1, NULL, 0, 0, 11, &b));
	memset(&sentinel, 0x5a, sizeof(sentinel));
	got = sentinel;
	CHECK(mdb_page_agg_local(&mc, mp, &got) == MDB_CORRUPTED);
	CHECK(memcmp(&got, &sentinel, sizeof(got)) == 0);
	return 0;
}

static int
test_plain_leaf_value_source(void)
{
	const uint16_t agg = MDB_AGG_ENTRIES|MDB_AGG_KEYS|MDB_AGG_HASHSUM;
	unsigned char pagebuf[TEST_PAGE_SIZE];
	MDB_page *mp = (MDB_page *)pagebuf;
	MDB_db db;
	MDB_cursor mc;
	MDB_aggval expected, got;
	uint8_t v1[MDB_HASH_SIZE+8], v2[MDB_HASH_SIZE+8];
	const char k1[] = "alpha", k2[] = "beta";

	fill_bytes(v1, sizeof(v1), 7);
	fill_bytes(v2, sizeof(v2), 61);
	memset(&db, 0, sizeof(db));
	db.md_flags = agg;
	db.md_hash_offset = 3;
	init_cursor(&mc, &db, 0);
	test_page_init(pagebuf, sizeof(pagebuf), P_LEAF);
	CHECK(test_add_node(mp, sizeof(pagebuf), k1, sizeof(k1)-1,
		v1, sizeof(v1), 0, 0, NULL));
	CHECK(test_add_node(mp, sizeof(pagebuf), k2, sizeof(k2)-1,
		v2, sizeof(v2), 0, 0, NULL));
	mdb_aggval_zero(&expected);
	CHECK(add_expected_record(agg, db.md_hash_offset, v1, sizeof(v1), &expected) == 0);
	CHECK(add_expected_record(agg, db.md_hash_offset, v2, sizeof(v2), &expected) == 0);
	CHECK(mdb_page_agg_local(&mc, mp, &got) == MDB_SUCCESS);
	CHECK(agg_equal(agg, &got, &expected));
	return 0;
}

static int
test_overflow_value_source(void)
{
	const uint16_t agg = MDB_AGG_ENTRIES|MDB_AGG_KEYS|MDB_AGG_HASHSUM;
	char dir[] = "/tmp/aelmdb-local-fold-XXXXXX";
	char data_path[256], lock_path[256];
	MDB_env *env = NULL;
	MDB_txn *txn = NULL;
	MDB_dbi dbi;
	MDB_cursor mc;
	MDB_xcursor mx;
	MDB_val key, data;
	MDB_aggval expected, got;
	uint8_t *value;
	size_t value_size = (size_t)MDB_HASH_SIZE + 8192u;
	const char k[] = "overflow";
	int rc = 1;

	CHECK(mkdtemp(dir) != NULL);
	value = malloc(value_size);
	CHECK(value != NULL);
	fill_bytes(value, value_size, 31);
	CHECK(mdb_env_create(&env) == MDB_SUCCESS);
	CHECK(mdb_env_set_mapsize(env, 4u * 1024u * 1024u) == MDB_SUCCESS);
	CHECK(mdb_env_open(env, dir, 0, 0600) == MDB_SUCCESS);
	CHECK(mdb_txn_begin(env, NULL, 0, &txn) == MDB_SUCCESS);
	CHECK(mdb_dbi_open(txn, NULL, agg, &dbi) == MDB_SUCCESS);
	CHECK(mdb_set_hash_offset(txn, dbi, 7) == MDB_SUCCESS);
	key.mv_data = (void *)k;
	key.mv_size = sizeof(k)-1;
	data.mv_data = value;
	data.mv_size = value_size;
	CHECK(mdb_put(txn, dbi, &key, &data, 0) == MDB_SUCCESS);
	mdb_cursor_init(&mc, txn, dbi, &mx);
	CHECK(mdb_page_search(&mc, &key, 0) == MDB_SUCCESS);
	CHECK(IS_LEAF(mc.mc_pg[mc.mc_top]));
	mdb_aggval_zero(&expected);
	CHECK(add_expected_record(agg, 7, value, value_size, &expected) == MDB_SUCCESS);
	CHECK(mdb_page_agg_local(&mc, mc.mc_pg[mc.mc_top], &got) == MDB_SUCCESS);
	CHECK(agg_equal(agg, &got, &expected));
	rc = 0;

	mdb_txn_abort(txn);
	mdb_env_close(env);
	free(value);
	snprintf(data_path, sizeof(data_path), "%s/data.mdb", dir);
	snprintf(lock_path, sizeof(lock_path), "%s/lock.mdb", dir);
	unlink(data_path);
	unlink(lock_path);
	rmdir(dir);
	return rc;
}

static int
test_plain_leaf_key_source(void)
{
	const uint16_t agg = MDB_AGG_ENTRIES|MDB_AGG_KEYS|MDB_AGG_HASHSUM;
	unsigned char pagebuf[TEST_PAGE_SIZE];
	MDB_page *mp = (MDB_page *)pagebuf;
	MDB_db db;
	MDB_cursor mc;
	MDB_aggval expected, got;
	uint8_t k1[MDB_HASH_SIZE+5], k2[MDB_HASH_SIZE+5];
	const uint8_t v1[] = {1,2,3}, v2[] = {4,5,6,7};

	fill_bytes(k1, sizeof(k1), 13);
	fill_bytes(k2, sizeof(k2), 79);
	memset(&db, 0, sizeof(db));
	db.md_flags = agg|MDB_AGG_HASHSOURCE_FROM_KEY;
	db.md_hash_offset = -1;
	init_cursor(&mc, &db, 0);
	test_page_init(pagebuf, sizeof(pagebuf), P_LEAF);
	CHECK(test_add_node(mp, sizeof(pagebuf), k1, sizeof(k1), v1, sizeof(v1), 0, 0, NULL));
	CHECK(test_add_node(mp, sizeof(pagebuf), k2, sizeof(k2), v2, sizeof(v2), 0, 0, NULL));
	mdb_aggval_zero(&expected);
	CHECK(add_expected_record(agg, db.md_hash_offset, k1, sizeof(k1), &expected) == 0);
	CHECK(add_expected_record(agg, db.md_hash_offset, k2, sizeof(k2), &expected) == 0);
	CHECK(mdb_page_agg_local(&mc, mp, &got) == MDB_SUCCESS);
	CHECK(agg_equal(agg, &got, &expected));
	return 0;
}

static int
test_main_catalog_record_is_plain_value(void)
{
	const uint16_t agg = MDB_AGG_ENTRIES|MDB_AGG_KEYS|MDB_AGG_HASHSUM;
	unsigned char pagebuf[TEST_PAGE_SIZE];
	MDB_page *mp = (MDB_page *)pagebuf;
	MDB_db db, named;
	MDB_cursor mc;
	MDB_aggval expected, got;
	const char name[] = "named";

	memset(&db, 0, sizeof(db));
	db.md_flags = agg;
	db.md_hash_offset = 0;
	memset(&named, 0, sizeof(named));
	named.md_entries = 17;
	named.md_keys = 9;
	fill_hash(named.md_hashsum, 43);
	named.md_root = 123;
	init_cursor(&mc, &db, 0);
	test_page_init(pagebuf, sizeof(pagebuf), P_LEAF);
	CHECK(test_add_node(mp, sizeof(pagebuf), name, sizeof(name)-1,
		&named, sizeof(named), F_SUBDATA, 0, NULL));
	mdb_aggval_zero(&expected);
	CHECK(add_expected_record(agg, db.md_hash_offset, &named, sizeof(named), &expected) == 0);
	CHECK(mdb_page_agg_local(&mc, mp, &got) == MDB_SUCCESS);
	CHECK(agg_equal(agg, &got, &expected));
	return 0;
}

static int
test_dupsort_inline_subpage(void)
{
	const uint16_t agg = MDB_AGG_ENTRIES|MDB_AGG_KEYS|MDB_AGG_HASHSUM;
	unsigned char outerbuf[TEST_PAGE_SIZE], subbuf[TEST_SUBPAGE_SIZE];
	MDB_page *outer = (MDB_page *)outerbuf, *sub = (MDB_page *)subbuf;
	MDB_db db;
	MDB_cursor mc;
	MDB_aggval expected, got, internal;
	uint8_t d1[MDB_HASH_SIZE+4], d2[MDB_HASH_SIZE+4], d3[MDB_HASH_SIZE+4];
	const char key[] = "k";

	fill_bytes(d1, sizeof(d1), 5);
	fill_bytes(d2, sizeof(d2), 37);
	fill_bytes(d3, sizeof(d3), 101);
	memset(&db, 0, sizeof(db));
	db.md_flags = MDB_DUPSORT|agg;
	db.md_hash_offset = 2;
	init_cursor(&mc, &db, 0);
	test_page_init(subbuf, sizeof(subbuf), P_LEAF|P_SUBP);
	CHECK(test_add_node(sub, sizeof(subbuf), d1, sizeof(d1), NULL, 0, 0, 0, NULL));
	CHECK(test_add_node(sub, sizeof(subbuf), d2, sizeof(d2), NULL, 0, 0, 0, NULL));
	CHECK(test_add_node(sub, sizeof(subbuf), d3, sizeof(d3), NULL, 0, 0, 0, NULL));
	CHECK(mdb_dup_leaf_agg_local(&mc, sub, sizeof(subbuf), &internal) == MDB_SUCCESS);
	CHECK(internal.entries == 3 && internal.keys == 3);

	test_page_init(outerbuf, sizeof(outerbuf), P_LEAF);
	CHECK(test_add_node(outer, sizeof(outerbuf), key, sizeof(key)-1,
		subbuf, sizeof(subbuf), F_DUPDATA, 0, NULL));
	mdb_aggval_zero(&expected);
	CHECK(add_expected_record(agg, db.md_hash_offset, d1, sizeof(d1), &expected) == 0);
	CHECK(add_expected_record(agg, db.md_hash_offset, d2, sizeof(d2), &expected) == 0);
	CHECK(add_expected_record(agg, db.md_hash_offset, d3, sizeof(d3), &expected) == 0);
	expected.keys = 1;
	CHECK(mdb_page_agg_local(&mc, outer, &got) == MDB_SUCCESS);
	CHECK(agg_equal(agg, &got, &expected));
	return 0;
}

static int
test_dupsort_persistent_subdb(void)
{
	const uint16_t agg = MDB_AGG_ENTRIES|MDB_AGG_KEYS|MDB_AGG_HASHSUM;
	unsigned char pagebuf[TEST_PAGE_SIZE];
	MDB_page *mp = (MDB_page *)pagebuf;
	MDB_db db, dupdb;
	MDB_cursor mc;
	MDB_aggval expected, got;
	const char key[] = "dup";

	memset(&db, 0, sizeof(db));
	db.md_flags = MDB_DUPSORT|agg;
	db.md_hash_offset = -1;
	memset(&dupdb, 0, sizeof(dupdb));
	dupdb.md_flags = agg;
	dupdb.md_entries = 4;
	dupdb.md_keys = 4;
	dupdb.md_hash_offset = db.md_hash_offset;
	fill_hash(dupdb.md_hashsum, 17);
	dupdb.md_depth = 2;
	dupdb.md_root = 99;
	init_cursor(&mc, &db, 0);
	test_page_init(pagebuf, sizeof(pagebuf), P_LEAF);
	CHECK(test_add_node(mp, sizeof(pagebuf), key, sizeof(key)-1,
		&dupdb, sizeof(dupdb), F_DUPDATA|F_SUBDATA, 0, NULL));
	mdb_aggval_zero(&expected);
	expected.entries = 4;
	expected.keys = 1;
	memcpy(expected.hashsum, dupdb.md_hashsum, MDB_HASH_SIZE);
	CHECK(mdb_page_agg_local(&mc, mp, &got) == MDB_SUCCESS);
	CHECK(agg_equal(agg, &got, &expected));
	return 0;
}

static int
test_dup_subcursor_leaf2(void)
{
	const uint16_t agg = MDB_AGG_ENTRIES|MDB_AGG_KEYS|MDB_AGG_HASHSUM;
	unsigned char pagebuf[TEST_PAGE_SIZE];
	MDB_page *mp = (MDB_page *)pagebuf;
	MDB_db db;
	MDB_cursor mc;
	MDB_aggval expected, got;
	uint8_t d1[MDB_HASH_SIZE+3], d2[MDB_HASH_SIZE+3], d3[MDB_HASH_SIZE+3];

	fill_bytes(d1, sizeof(d1), 9);
	fill_bytes(d2, sizeof(d2), 71);
	fill_bytes(d3, sizeof(d3), 149);
	memset(&db, 0, sizeof(db));
	db.md_flags = MDB_DUPFIXED|agg;
	db.md_pad = sizeof(d1);
	db.md_hash_offset = -1;
	init_cursor(&mc, &db, C_SUB);
	test_page_init(pagebuf, sizeof(pagebuf), P_LEAF|P_LEAF2);
	MP_PAD(mp) = (uint16_t)sizeof(d1);
	CHECK(test_add_leaf2(mp, sizeof(pagebuf), d1, sizeof(d1)));
	CHECK(test_add_leaf2(mp, sizeof(pagebuf), d2, sizeof(d2)));
	CHECK(test_add_leaf2(mp, sizeof(pagebuf), d3, sizeof(d3)));
	mdb_aggval_zero(&expected);
	CHECK(add_expected_record(agg, db.md_hash_offset, d1, sizeof(d1), &expected) == 0);
	CHECK(add_expected_record(agg, db.md_hash_offset, d2, sizeof(d2), &expected) == 0);
	CHECK(add_expected_record(agg, db.md_hash_offset, d3, sizeof(d3), &expected) == 0);
	CHECK(mdb_page_agg_local(&mc, mp, &got) == MDB_SUCCESS);
	CHECK(agg_equal(agg, &got, &expected));
	return 0;
}

static int
test_failure_atomicity(void)
{
	const uint16_t agg = MDB_AGG_ENTRIES|MDB_AGG_KEYS|MDB_AGG_HASHSUM;
	unsigned char pagebuf[TEST_PAGE_SIZE];
	MDB_page *mp = (MDB_page *)pagebuf;
	MDB_db db;
	MDB_cursor mc;
	MDB_aggval got, sentinel;
	uint8_t shortv[MDB_HASH_SIZE ? MDB_HASH_SIZE : 1];
	const char key[] = "x";
	size_t shortn = MDB_HASH_SIZE - 1;

	memset(&db, 0, sizeof(db));
	db.md_flags = agg;
	db.md_hash_offset = 0;
	init_cursor(&mc, &db, 0);
	test_page_init(pagebuf, sizeof(pagebuf), P_LEAF);
	fill_bytes(shortv, sizeof(shortv), 23);
	CHECK(test_add_node(mp, sizeof(pagebuf), key, sizeof(key)-1,
		shortv, shortn, 0, 0, NULL));
	memset(&sentinel, 0xa7, sizeof(sentinel));
	got = sentinel;
	CHECK(mdb_page_agg_local(&mc, mp, &got) == MDB_BAD_VALSIZE);
	CHECK(memcmp(&got, &sentinel, sizeof(got)) == 0);

	/* Branch page schema mismatch is rejected before interpreting prefixes. */
	test_page_init(pagebuf, sizeof(pagebuf), P_BRANCH|MDB_AGG_ENTRIES);
	got = sentinel;
	CHECK(mdb_page_agg_local(&mc, mp, &got) == MDB_CORRUPTED);
	CHECK(memcmp(&got, &sentinel, sizeof(got)) == 0);
	return 0;
}

int
main(void)
{
	if (test_branch_local()) return 1;
	if (test_plain_leaf_value_source()) return 1;
	if (test_overflow_value_source()) return 1;
	if (test_plain_leaf_key_source()) return 1;
	if (test_main_catalog_record_is_plain_value()) return 1;
	if (test_dupsort_inline_subpage()) return 1;
	if (test_dupsort_persistent_subdb()) return 1;
	if (test_dup_subcursor_leaf2()) return 1;
	if (test_failure_atomicity()) return 1;
	printf("aggregate local-page tests passed (MDB_HASH_SIZE=%d)\n", MDB_HASH_SIZE);
	return 0;
}
