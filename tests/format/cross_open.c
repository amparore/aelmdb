/* X1: open an aggregate data file written by another 04 implementation
 * (cross-generation format compatibility).
 *
 *   X1_cross_open SRC.mdb       (or X1_SRC=SRC.mdb X1_cross_open)
 *
 * SRC.mdb is copied to ./db/data.mdb.  For every named aggregate DB of the
 * file the test runs the recursive integrity oracle and prints the totals,
 * then changes the DB in one write transaction (deletes every 5th record,
 * rewrites every 7th with a different value of the same size), and checks
 * and prints again, before and after the commit.  The output depends only
 * on the file and on the aggregate semantics: both implementations must print the same
 * lines for the same input, and the oracle must hold throughout.
 *
 * Build with -DMDB_DEBUG_AGG_INTEGRITY=1, linked with the stage's mdb.c.
 */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE 1
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include "lmdb.h"

#if !(defined(MDB_DEBUG_AGG_INTEGRITY) && MDB_DEBUG_AGG_INTEGRITY)
#error "X1 needs the integrity oracle: build with -DMDB_DEBUG_AGG_INTEGRITY=1"
#endif

#define C(x) do { int rc_ = (x); if (rc_) { \
	fprintf(stderr, "FAIL %s:%d: %s: %s\n", __FILE__, __LINE__, #x, mdb_strerror(rc_)); \
	exit(1); } } while (0)

#define MAXDB 8

typedef struct { MDB_val k, d; } Rec;

static MDB_env *env;
static char names[MAXDB][64];
static int ndb;

static void
copy_file(const char *src, const char *dst)
{
	char buf[65536];
	size_t n;
	FILE *in = fopen(src, "rb"), *out = fopen(dst, "wb");
	if (!in || !out) { fprintf(stderr, "FAIL copy %s\n", src); exit(1); }
	while ((n = fread(buf, 1, sizeof(buf), in)) > 0)
		if (fwrite(buf, 1, n, out) != n) { fprintf(stderr, "FAIL write\n"); exit(1); }
	fclose(in); fclose(out);
}

static void
report(MDB_txn *txn, const char *phase)
{
	int i;
	for (i = 0; i < ndb; i++) {
		MDB_dbi dbi;
		MDB_agg a;
		MDB_stat st;
		unsigned j;
		int rc;
		C(mdb_dbi_open(txn, names[i], 0, &dbi));
		rc = mdb_agg_check_integrity(txn, dbi);
		if (rc) {
			fprintf(stderr, "FAIL %s %s: integrity: %s\n", phase, names[i], mdb_strerror(rc));
			exit(1);
		}
		C(mdb_agg_totals(txn, dbi, &a));
		C(mdb_stat(txn, dbi, &st));
		printf("%s %s flags=%#x records=%zu entries=%llu keys=%llu hash=", phase, names[i],
			a.mv_flags, (size_t)st.ms_entries, (unsigned long long)a.mv_agg_entries,
			(unsigned long long)a.mv_agg_keys);
		for (j = 0; j < MDB_HASH_SIZE; j++)
			printf("%02x", a.mv_agg_hashes[j]);
		printf("\n");
	}
}

static void
mutate(MDB_txn *txn, const char *name)
{
	MDB_dbi dbi;
	MDB_cursor *mc;
	MDB_val k, d;
	unsigned int flags;
	Rec *r = NULL;
	size_t n = 0, cap = 0, i;
	int rc;

	C(mdb_dbi_open(txn, name, 0, &dbi));
	C(mdb_dbi_flags(txn, dbi, &flags));
	C(mdb_cursor_open(txn, dbi, &mc));
	for (rc = mdb_cursor_get(mc, &k, &d, MDB_FIRST); !rc;
		rc = mdb_cursor_get(mc, &k, &d, MDB_NEXT)) {
		if (n == cap) {
			cap = cap ? 2 * cap : 256;
			r = realloc(r, cap * sizeof(*r));
		}
		r[n].k.mv_size = k.mv_size;
		r[n].k.mv_data = memcpy(malloc(k.mv_size + 1), k.mv_data, k.mv_size);
		r[n].d.mv_size = d.mv_size;
		r[n].d.mv_data = memcpy(malloc(d.mv_size + 1), d.mv_data, d.mv_size);
		n++;
	}
	if (rc != MDB_NOTFOUND)
		C(rc);
	mdb_cursor_close(mc);
	for (i = 0; i < n; i++) {
		if (i % 5 == 0)
			C(mdb_del(txn, dbi, &r[i].k, (flags & MDB_DUPSORT) ? &r[i].d : NULL));
		if (i % 7 == 0 && r[i].d.mv_size) {
			/* same size (DUPFIXED), a different value; INTEGERDUP values
			 * stay valid integers */
			((unsigned char *)r[i].d.mv_data)[0] ^= 0x5a;
			rc = mdb_put(txn, dbi, &r[i].k, &r[i].d, 0);
			if (rc != MDB_KEYEXIST)
				C(rc);
		}
	}
	for (i = 0; i < n; i++) {
		free(r[i].k.mv_data);
		free(r[i].d.mv_data);
	}
	free(r);
}

int
main(int argc, char **argv)
{
	MDB_txn *txn;
	MDB_dbi main_dbi;
	MDB_cursor *mc;
	MDB_val k, d;
	const char *src = argc == 2 ? argv[1] : getenv("X1_SRC");
	int rc, i;

	if (!src) {
		fprintf(stderr, "usage: X1_cross_open SRC.mdb\n");
		return 2;
	}
	mkdir("db", 0700);
	copy_file(src, "db/data.mdb");
	remove("db/lock.mdb");
	C(mdb_env_create(&env));
	C(mdb_env_set_mapsize(env, (size_t)1 << 30));
	C(mdb_env_set_maxdbs(env, MAXDB));
	C(mdb_env_open(env, "db", 0, 0600));

	/* the named aggregate DBs of the file */
	C(mdb_txn_begin(env, NULL, MDB_RDONLY, &txn));
	C(mdb_dbi_open(txn, NULL, 0, &main_dbi));
	C(mdb_cursor_open(txn, main_dbi, &mc));
	for (rc = mdb_cursor_get(mc, &k, &d, MDB_FIRST); !rc;
		rc = mdb_cursor_get(mc, &k, &d, MDB_NEXT)) {
		char name[64];
		MDB_dbi dbi;
		unsigned agg;
		if (k.mv_size >= sizeof(name) || ndb == MAXDB)
			continue;
		memcpy(name, k.mv_data, k.mv_size);
		name[k.mv_size] = 0;
		if (mdb_dbi_open(txn, name, 0, &dbi) || mdb_agg_info(txn, dbi, &agg) || !agg)
			continue;
		strcpy(names[ndb++], name);
	}
	mdb_cursor_close(mc);
	mdb_txn_abort(txn);
	if (!ndb) {
		fprintf(stderr, "FAIL no aggregate DB in %s\n", src);
		return 1;
	}

	C(mdb_txn_begin(env, NULL, MDB_RDONLY, &txn));
	report(txn, "open");
	mdb_txn_abort(txn);

	C(mdb_txn_begin(env, NULL, 0, &txn));
	for (i = 0; i < ndb; i++)
		mutate(txn, names[i]);
	report(txn, "write");
	C(mdb_txn_commit(txn));

	C(mdb_txn_begin(env, NULL, MDB_RDONLY, &txn));
	report(txn, "commit");
	mdb_txn_abort(txn);
	mdb_env_close(env);
	return 0;
}
