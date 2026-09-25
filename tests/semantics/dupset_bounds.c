/* Q2 regression: record-order bounds just outside a dupset.
 *
 * A prefix bound (key, data) with data after the last duplicate of key covers
 * the same records as the key-only bound (key, inclusive) and as the prefix
 * up to the next key; with data before the first duplicate, the same as the
 * key-only bound (key, exclusive).  All three aggregate fields must then be
 * equal (entries, keys, hashsum), whatever the dupset container: an inline
 * sub-page or a persistent duplicate sub-DB.  Ranges are checked the same way.
 *
 * AELMDB found by the AELMDB unit suite ("dupsort round robin"): an inline
 * sub-page has no persistent totals of its own, and the query layer answered a
 * bound past its last duplicate with zero keys and hashsum (C5 checks only
 * entries).  Public API only; any MDB_HASH_SIZE.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "lmdb.h"

#define C(x) do { int r_ = (x); if (r_) { fprintf(stderr, "%s:%d: %s: %s\n", \
	__FILE__, __LINE__, #x, mdb_strerror(r_)); exit(2); } } while (0)

#define VLEN (MDB_HASH_SIZE > 32 ? MDB_HASH_SIZE : 32)
#define NKEYS 40

static MDB_env *env;
static MDB_txn *txn;
static unsigned failures;

static int
same(const MDB_agg *a, const MDB_agg *b)
{
	return a->mv_agg_entries == b->mv_agg_entries && a->mv_agg_keys == b->mv_agg_keys &&
		!memcmp(a->mv_agg_hashes, b->mv_agg_hashes, MDB_HASH_SIZE);
}

static void
expect(const char *what, unsigned flags, unsigned i, const MDB_agg *got, const MDB_agg *exp)
{
	if (same(got, exp))
		return;
	if (failures < 10)
		fprintf(stderr, "MISMATCH %s (db flags %#x, key %u): entries %llu/%llu keys %llu/%llu%s\n",
			what, flags, i, (unsigned long long)got->mv_agg_entries,
			(unsigned long long)exp->mv_agg_entries, (unsigned long long)got->mv_agg_keys,
			(unsigned long long)exp->mv_agg_keys,
			memcmp(got->mv_agg_hashes, exp->mv_agg_hashes, MDB_HASH_SIZE) ? " hash differs" : "");
	failures++;
}

/* dups of key i: 1..5 (inline), every 7th key 400 (a duplicate sub-DB) */
static unsigned
ndups(unsigned i)
{
	return i % 7 == 3 ? 400 : 1 + i % 5;
}

static void
mkkey(char *kb, unsigned i, MDB_val *k)
{
	snprintf(kb, 16, "K%05u", i);
	k->mv_size = strlen(kb);
	k->mv_data = kb;
}

static void
mkval(unsigned char *vb, unsigned i, unsigned j, MDB_val *v)
{
	unsigned n;
	vb[0] = 0x10;			/* above the lowest bound, below the highest */
	vb[1] = (unsigned char)(j >> 8);
	vb[2] = (unsigned char)j;
	for (n = 3; n < VLEN; n++)
		vb[n] = (unsigned char)(i * 131u + j * 7u + n * 29u);
	v->mv_size = VLEN;
	v->mv_data = vb;
}

static void
run(unsigned dbflags)
{
	MDB_dbi dbi;
	MDB_val k, k2, v, lo, hi;
	MDB_agg a, b, c, t;
	char kb[16], kb2[16];
	unsigned char vb[VLEN], lob[VLEN], hib[VLEN];
	unsigned i, j;

	C(mdb_txn_begin(env, NULL, 0, &txn));
	C(mdb_dbi_open(txn, "q2", MDB_CREATE | dbflags |
		MDB_AGG_ENTRIES | MDB_AGG_KEYS | MDB_AGG_HASHSUM, &dbi));
	for (i = 0; i < NKEYS; i++)
		for (j = 0; j < ndups(i); j++) {
			mkkey(kb, i, &k);
			mkval(vb, i, j, &v);
			C(mdb_put(txn, dbi, &k, &v, 0));
		}
	memset(lob, 0x00, VLEN); lo.mv_size = VLEN; lo.mv_data = lob;
	memset(hib, 0xff, VLEN); hi.mv_size = VLEN; hi.mv_data = hib;
	C(mdb_agg_totals(txn, dbi, &t));
	for (i = 0; i < NKEYS; i++) {
		mkkey(kb, i, &k);
		/* past the last duplicate == the whole key == up to the next key */
		C(mdb_agg_prefix(txn, dbi, &k, &hi, 0, &a));
		C(mdb_agg_prefix(txn, dbi, &k, NULL, MDB_AGG_PREFIX_INCL, &b));
		if (i + 1 < NKEYS) {
			mkkey(kb2, i + 1, &k2);
			C(mdb_agg_prefix(txn, dbi, &k2, NULL, 0, &c));
		} else {
			c = t;
		}
		expect("prefix (key, after last dup) == prefix (key, incl)", dbflags, i, &a, &b);
		expect("prefix (key, incl) == prefix (next key)", dbflags, i, &b, &c);
		/* before the first duplicate == the key excluded */
		C(mdb_agg_prefix(txn, dbi, &k, &lo, MDB_AGG_PREFIX_INCL, &a));
		C(mdb_agg_prefix(txn, dbi, &k, NULL, 0, &b));
		expect("prefix (key, before first dup) == prefix (key, excl)", dbflags, i, &a, &b);
		/* the dupset alone, as a record range and as a key range */
		C(mdb_agg_range(txn, dbi, &k, &lo, &k, &hi, 0, &a));
		C(mdb_agg_range(txn, dbi, &k, NULL, &k, NULL,
			MDB_RANGE_LOWER_INCL | MDB_RANGE_UPPER_INCL, &b));
		expect("range (key, lo)..(key, hi) == range [key, key]", dbflags, i, &a, &b);
		if (a.mv_agg_entries != ndups(i) || a.mv_agg_keys != 1)
			expect("range of one dupset: entries, one key", dbflags, i, &a, &t);
	}
	mdb_txn_abort(txn);
}

int
main(void)
{
	if (system("rm -rf testdb_q2 && mkdir testdb_q2")) {}
	C(mdb_env_create(&env));
	C(mdb_env_set_maxdbs(env, 2));
	C(mdb_env_set_mapsize(env, 1u << 28));
	C(mdb_env_open(env, "testdb_q2", MDB_NOSYNC, 0664));
	run(MDB_DUPSORT);
	run(MDB_DUPSORT | MDB_DUPFIXED);
	mdb_env_close(env);
	if (failures) {
		fprintf(stderr, "Q2 dupset-bound regression: %u failures\n", failures);
		return 1;
	}
	printf("Q2 dupset-bound tests passed (MDB_HASH_SIZE=%d)\n", MDB_HASH_SIZE);
	return 0;
}
