/* F1: aggregate schema and format validation (03-aggformat).
 *
 * API: schema requests (HASHSOURCE_FROM_KEY rules, explicit schema on
 * reopen, main-DB schema only while empty), mdb_agg_info, hash offset rules.
 *
 * Corrupted files (the data file is patched directly, then reopened): the
 * format version bound to MDB_HASH_SIZE, aggregate bits of a branch page or
 * of a leaf page, a dup sub-DB record with another schema or a root beyond
 * the file, a sub-page header, a named DB record whose root has the wrong
 * page type.  Each must be refused (MDB_VERSION_MISMATCH, MDB_CORRUPTED,
 * MDB_INCOMPATIBLE) without a crash; a write that meets a refused container
 * invalidates the transaction.
 *
 * E3 regressions: a non-empty aggregate DUPSORT main DB reopens (the layout
 * check needs an xcursor for it), and the schema queries refuse a blocked
 * transaction.
 *
 * White-box: includes mdb.c (any MDB_HASH_SIZE in [1, 256]).
 */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE 1
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "mdb.c"

#define DIR "testdb_f1"
#include "f_file.h"

#define SCHEMA (MDB_AGG_ENTRIES|MDB_AGG_KEYS|MDB_AGG_HASHSUM)
#define CK(x) do { int r_ = (x); if (r_) { fprintf(stderr, "%s:%d: %s: %s\n", \
	__FILE__, __LINE__, #x, mdb_strerror(r_)); exit(2); } } while (0)

static MDB_env *env;

static void
expect(int rc, int want, const char *what)
{
	char buf[256];
	if (rc == want)
		return;
	snprintf(buf, sizeof(buf), "%s: got %s, expected %s", what,
		rc ? mdb_strerror(rc) : "success", want ? mdb_strerror(want) : "success");
	fail(buf);
}

static int
env_open(void)
{
	int rc;
	CK(mdb_env_create(&env));
	CK(mdb_env_set_maxdbs(env, 8));
	CK(mdb_env_set_mapsize(env, (size_t)1 << 28));
	rc = mdb_env_open(env, DIR, MDB_NOSYNC, 0664);
	if (rc) {
		mdb_env_close(env);
		env = NULL;
	}
	return rc;
}

static void
env_close(void)
{
	if (env)
		mdb_env_close(env);
	env = NULL;
}

static void
mkval(unsigned char *b, MDB_val *v, unsigned ver, size_t len)
{
	memset(b, 0, len);
	b[0] = (unsigned char)(ver >> 8); b[1] = (unsigned char)ver;
	v->mv_size = len;
	v->mv_data = b;
}

/* -------------------------------------------------------------- file */

/* the leaf node with key k in the tree at root (linear walk) */
static MDB_node *
find_node(pgno_t root, int depth, const MDB_val *k)
{
	MDB_page *mp;
	unsigned i;
	if (!depth || root == P_INVALID)
		return NULL;
	mp = pg(root);
	for (i = 0; i < NUMKEYS(mp); i++) {
		MDB_node *node = NODEPTR(mp, i);
		if (IS_BRANCH(mp)) {
			MDB_node *r = find_node(NODEPGNO(node), depth - 1, k);
			if (r)
				return r;
		} else if (IS_LEAF(mp) && !IS_LEAF2(mp) && NODEKSZ(node) == k->mv_size &&
			!memcmp(NODEKEY(mp, node), k->mv_data, k->mv_size)) {
			return node;
		}
	}
	return NULL;
}

/* the MDB_db record of named DB name in the loaded file */
static MDB_node *
named_db(const char *name, MDB_db *db)
{
	MDB_meta *m = meta();
	MDB_val k = { strlen(name), (void *)name };
	MDB_node *node = find_node(m->mm_dbs[MAIN_DBI].md_root, m->mm_dbs[MAIN_DBI].md_depth, &k);
	if (!node) { fprintf(stderr, "named DB %s not found\n", name); exit(2); }
	memcpy(db, NODEDATA(node), sizeof(*db));
	return node;
}

/* keys: "p" plain aggregate DB of depth >= 2; "d" DUPSORT aggregate DB with
 * key "S" (a sub-DB) and key "s" (an inline sub-page) */
static void
build(void)
{
	MDB_txn *txn;
	MDB_dbi dp, dd;
	unsigned char kb[256], vb[128];
	MDB_val k, v;
	unsigned i;
	if (system("rm -rf " DIR " && mkdir " DIR)) {}
	CK(env_open());
	CK(mdb_txn_begin(env, NULL, 0, &txn));
	CK(mdb_dbi_open(txn, "p", MDB_CREATE | SCHEMA, &dp));
	CK(mdb_dbi_open(txn, "d", MDB_CREATE | MDB_DUPSORT | SCHEMA, &dd));
	CK(mdb_set_hash_offset(txn, dd, 3));
	for (i = 0; i < 4000; i++) {
		snprintf((char *)kb, sizeof(kb), "key-%06u-%0200u", i, i);
		k.mv_size = strlen((char *)kb); k.mv_data = kb;
		mkval(vb, &v, i, 40);
		CK(mdb_put(txn, dp, &k, &v, 0));
	}
	for (i = 0; i < 600; i++) {
		k.mv_size = 1; k.mv_data = "S";
		mkval(vb, &v, i, 100);
		CK(mdb_put(txn, dd, &k, &v, 0));
	}
	for (i = 0; i < 3; i++) {
		k.mv_size = 1; k.mv_data = "s";
		mkval(vb, &v, i, 10);
		CK(mdb_put(txn, dd, &k, &v, 0));
	}
	CK(mdb_txn_commit(txn));
	env_close();
}

/* --------------------------------------------------------------- API */

static void
api(void)
{
	MDB_txn *txn;
	MDB_dbi dbi, dd;
	unsigned flags;
	int off, rc;
	MDB_val k = { 1, "k" }, v = { 1, "v" };

	build();
	CK(env_open());
	CK(mdb_txn_begin(env, NULL, 0, &txn));
	expect(mdb_dbi_open(txn, "x1", MDB_CREATE | MDB_AGG_HASHSOURCE_FROM_KEY, &dbi),
		MDB_INCOMPATIBLE, "HASHSOURCE_FROM_KEY without HASHSUM");
	expect(mdb_dbi_open(txn, "x2", MDB_CREATE | MDB_DUPSORT | MDB_AGG_HASHSUM |
		MDB_AGG_HASHSOURCE_FROM_KEY, &dbi), MDB_INCOMPATIBLE, "HASHSOURCE_FROM_KEY with DUPSORT");
	expect(mdb_dbi_open(txn, "p", MDB_AGG_ENTRIES, &dbi), MDB_INCOMPATIBLE,
		"reopen with another explicit schema");
	expect(mdb_dbi_open(txn, "p", 0, &dbi), 0, "reopen without schema flags");
	expect(mdb_agg_info(txn, dbi, &flags), 0, "mdb_agg_info");
	if (flags != SCHEMA) fail("mdb_agg_info: wrong schema");
	expect(mdb_dbi_open(txn, "p", SCHEMA, &dbi), 0, "reopen with the same schema");
	expect(mdb_set_hash_offset(txn, dbi, 1), MDB_INCOMPATIBLE, "hash offset on a non-empty DB");
	expect(mdb_dbi_open(txn, "e", MDB_CREATE | MDB_AGG_ENTRIES, &dbi), 0, "create ENTRIES only");
	expect(mdb_set_hash_offset(txn, dbi, 1), MDB_INCOMPATIBLE, "hash offset without HASHSUM");
	expect(mdb_get_hash_offset(txn, dbi, &off), MDB_INCOMPATIBLE, "get hash offset without HASHSUM");
	expect(mdb_dbi_open(txn, "d", 0, &dd), 0, "open d");
	expect(mdb_get_hash_offset(txn, dd, &off), 0, "get hash offset");
	if (off != 3) fail("hash offset not persistent");
	/* main DB: schema only while empty */
	expect(mdb_dbi_open(txn, NULL, SCHEMA, &dbi), MDB_INCOMPATIBLE, "main-DB schema on a non-empty main DB");
	mdb_txn_abort(txn);
	env_close();

	if (system("rm -rf " DIR " && mkdir " DIR)) {}
	CK(env_open());
	CK(mdb_txn_begin(env, NULL, 0, &txn));
	expect(mdb_dbi_open(txn, NULL, MDB_AGG_KEYS, &dbi), 0, "main-DB schema while empty");
	CK(mdb_put(txn, dbi, &k, &v, 0));
	expect(mdb_dbi_open(txn, NULL, MDB_AGG_ENTRIES, &dbi), MDB_INCOMPATIBLE, "main-DB schema change after a put");
	rc = mdb_agg_info(txn, dbi, &flags);
	if (rc || flags != MDB_AGG_KEYS) fail("main-DB schema not kept");
	CK(mdb_txn_commit(txn));
	env_close();
}

/* ----------------------------------------------------------- corrupted */

/* open, run one read and one write on DB name/key, expect rc for both;
 * after a refused write the transaction must be unusable */
static void
probe(const char *name, const char *key, int want_open, int want, const char *what)
{
	MDB_txn *txn;
	MDB_dbi dbi;
	MDB_cursor *c;
	MDB_val k = { strlen(key), (void *)key }, v;
	unsigned char vb[128];
	char w[160];
	int rc;

	rc = env_open();
	if (rc) {
		snprintf(w, sizeof(w), "%s: env open", what);
		expect(rc, want_open, w);
		return;
	}
	CK(mdb_txn_begin(env, NULL, 0, &txn));
	rc = mdb_dbi_open(txn, name, 0, &dbi);
	snprintf(w, sizeof(w), "%s: dbi open", what);
	if (rc || want_open) {
		expect(rc, want_open, w);
		mdb_txn_abort(txn);
		env_close();
		return;
	}
	CK(mdb_cursor_open(txn, dbi, &c));
	if (!strcmp(key, "*")) {
		rc = mdb_cursor_get(c, &k, &v, MDB_LAST);
	} else {
		rc = mdb_cursor_get(c, &k, &v, MDB_SET_KEY);
		if (!rc)
			rc = mdb_cursor_get(c, &k, &v, MDB_LAST_DUP);
	}
	snprintf(w, sizeof(w), "%s: read", what);
	expect(rc, want, w);
	mdb_cursor_close(c);
	if (strcmp(key, "*")) {
		k.mv_size = strlen(key); k.mv_data = (void *)key;
		mkval(vb, &v, 9999, 100);
		rc = mdb_put(txn, dbi, &k, &v, 0);
		snprintf(w, sizeof(w), "%s: write", what);
		expect(rc, want, w);
		if (rc) {
			/* refused before any change (the search validates the
			 * container), or after one (the transaction is then
			 * invalidated): a retry must fail the same way */
			int rc2 = mdb_put(txn, dbi, &k, &v, 0);
			snprintf(w, sizeof(w), "%s: retry after a refused write", what);
			if (rc2 != rc && rc2 != MDB_BAD_TXN)
				expect(rc2, rc, w);
		}
	}
	mdb_txn_abort(txn);
	env_close();
}

static void
corrupted(void)
{
	MDB_db db;
	MDB_node *node;
	MDB_meta *m;
	MDB_val k;

	/* baseline: the intact file passes the probes */
	build();
	probe("p", "*", 0, 0, "intact p");
	probe("d", "S", 0, 0, "intact d/S");
	probe("d", "s", 0, 0, "intact d/s");

	/* format version bound to MDB_HASH_SIZE */
	build(); load();
	(void)meta();		/* sets psize */
	m = (MDB_meta *)METADATA(pg(0)); m->mm_version ^= 1;
	m = (MDB_meta *)METADATA(pg(1)); m->mm_version ^= 1;
	store();
	probe("p", "*", MDB_VERSION_MISMATCH, 0, "other MDB_HASH_SIZE");

	/* root branch page of p without its aggregate bits: refused when the
	 * DB is opened */
	build(); load();
	named_db("p", &db);
	if (db.md_depth < 3) { fail("p too shallow for the test"); return; }
	MP_FLAGS(pg(db.md_root)) &= ~MDB_AGG_ENTRIES;
	store();
	probe("p", "*", MDB_CORRUPTED, 0, "root branch page with other aggregate bits");

	/* inner branch page of p without its aggregate bits: refused by the
	 * search */
	build(); load();
	named_db("p", &db);
	{
		MDB_page *rp = pg(db.md_root);
		MP_FLAGS(pg(NODEPGNO(NODEPTR(rp, NUMKEYS(rp) - 1)))) &= ~MDB_AGG_ENTRIES;
	}
	store();
	probe("p", "*", 0, MDB_CORRUPTED, "inner branch page with other aggregate bits");

	/* leaf page of p with aggregate bits */
	build(); load();
	named_db("p", &db);
	{
		pgno_t n = db.md_root;
		while (IS_BRANCH(pg(n)))
			n = NODEPGNO(NODEPTR(pg(n), NUMKEYS(pg(n)) - 1));
		MP_FLAGS(pg(n)) |= MDB_AGG_KEYS;
	}
	store();
	probe("p", "*", 0, MDB_CORRUPTED, "leaf page with aggregate bits");

	/* named DB record: depth 2, but the root is a leaf */
	build(); load();
	node = named_db("p", &db);
	{
		pgno_t n = db.md_root;
		while (IS_BRANCH(pg(n)))
			n = NODEPGNO(NODEPTR(pg(n), 0));
		db.md_root = n;
		memcpy(NODEDATA(node), &db, sizeof(db));
	}
	store();
	probe("p", "*", MDB_CORRUPTED, 0, "named DB root of the wrong page type");

	/* dup sub-DB record with another schema */
	build(); load();
	named_db("d", &db);
	k.mv_size = 1; k.mv_data = "S";
	node = find_node(db.md_root, db.md_depth, &k);
	if (!node || !(node->mn_flags & F_SUBDATA)) { fail("d/S is not a sub-DB"); return; }
	memcpy(&db, NODEDATA(node), sizeof(db));
	db.md_flags &= ~MDB_AGG_KEYS;
	memcpy(NODEDATA(node), &db, sizeof(db));
	store();
	probe("d", "S", 0, MDB_INCOMPATIBLE, "dup sub-DB with another schema");

	/* dup sub-DB record with another hash offset */
	build(); load();
	named_db("d", &db);
	node = find_node(db.md_root, db.md_depth, &k);
	memcpy(&db, NODEDATA(node), sizeof(db));
	db.md_hash_offset++;
	memcpy(NODEDATA(node), &db, sizeof(db));
	store();
	probe("d", "S", 0, MDB_INCOMPATIBLE, "dup sub-DB with another hash offset");

	/* dup sub-DB root beyond the file */
	build(); load();
	named_db("d", &db);
	node = find_node(db.md_root, db.md_depth, &k);
	memcpy(&db, NODEDATA(node), sizeof(db));
	db.md_root = meta()->mm_last_pg + 100;
	memcpy(NODEDATA(node), &db, sizeof(db));
	store();
	probe("d", "S", 0, MDB_CORRUPTED, "dup sub-DB root beyond the file");

	/* inline sub-page with lower > upper */
	build(); load();
	named_db("d", &db);
	k.mv_size = 1; k.mv_data = "s";
	node = find_node(db.md_root, db.md_depth, &k);
	if (!node || !(node->mn_flags & F_DUPDATA) || (node->mn_flags & F_SUBDATA)) {
		fail("d/s is not an inline sub-page"); return;
	}
	{
		MDB_page *fp = NODEDATA(node);
		MP_LOWER(fp) = MP_UPPER(fp) + 2;
	}
	store();
	probe("d", "s", 0, MDB_CORRUPTED, "sub-page with lower > upper");

	/* inline sub-page with aggregate bits */
	build(); load();
	named_db("d", &db);
	node = find_node(db.md_root, db.md_depth, &k);
	MP_FLAGS((MDB_page *)NODEDATA(node)) |= MDB_AGG_ENTRIES;
	store();
	probe("d", "s", 0, MDB_CORRUPTED, "sub-page with aggregate bits");
}

/* ---------------------------------------------------------------- E3 */

static void
e3(void)
{
	MDB_txn *txn;
	MDB_dbi dm;
	unsigned char vb[MDB_HASH_SIZE + 8];
	MDB_val k, v;
	unsigned i, f = 0;
	int off;

	if (system("rm -rf " DIR " && mkdir " DIR)) {}
	CK(env_open());
	CK(mdb_txn_begin(env, NULL, 0, &txn));
	CK(mdb_dbi_open(txn, NULL, MDB_DUPSORT | SCHEMA, &dm));
	for (i = 0; i < 300; i++) {
		k.mv_size = 1; k.mv_data = (i & 1) ? "a" : "b";
		mkval(vb, &v, i, sizeof(vb));
		CK(mdb_put(txn, dm, &k, &v, 0));
	}
	CK(mdb_txn_commit(txn));
	env_close();

	/* reopen: the layout check of a non-empty DUPSORT main DB */
	CK(env_open());
	CK(mdb_txn_begin(env, NULL, MDB_RDONLY, &txn));
	expect(mdb_dbi_open(txn, NULL, 0, &dm), 0, "E3: reopen a non-empty DUPSORT main DB");
	expect(mdb_agg_info(txn, dm, &f), 0, "E3: mdb_agg_info on the DUPSORT main DB");
	if ((f & SCHEMA) != SCHEMA)
		fail("E3: schema of the DUPSORT main DB");
	mdb_txn_abort(txn);

	/* a blocked transaction: the schema queries refuse it */
	CK(mdb_txn_begin(env, NULL, 0, &txn));
	CK(mdb_dbi_open(txn, NULL, 0, &dm));
	txn->mt_flags |= MDB_TXN_ERROR;
	expect(mdb_agg_info(txn, dm, &f), MDB_BAD_TXN, "E3: mdb_agg_info on a blocked txn");
	expect(mdb_get_hash_offset(txn, dm, &off), MDB_BAD_TXN,
		"E3: mdb_get_hash_offset on a blocked txn");
	mdb_txn_abort(txn);
	env_close();
}

int
main(void)
{
	api();
	corrupted();
#ifndef F1_NO_E3	/* stage 03 tests only the format; E3 belongs to the final API layer */
	e3();
#endif
	free(fbuf);
	if (failures) {
		fprintf(stderr, "F1 schema and format validation: %u failures\n", failures);
		return 1;
	}
	printf("F1 schema and format validation passed (MDB_HASH_SIZE %d)\n", MDB_HASH_SIZE);
	return 0;
}
