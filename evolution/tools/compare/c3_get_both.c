/* Minimal regression for AELMDB C3: MDB_GET_BOTH must not succeed on an
 * inexact duplicate match. Compile unchanged against AELMDB or aggmaint. */
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include "lmdb.h"

static void die(const char *where, int rc)
{
	if (rc != MDB_SUCCESS) {
		fprintf(stderr, "%s: %d (%s)\n", where, rc, mdb_strerror(rc));
		exit(2);
	}
}

int main(void)
{
	char path[] = "/tmp/aelc3XXXXXX";
	char file[512];
	MDB_env *env = NULL;
	MDB_txn *txn = NULL;
	MDB_cursor *cur = NULL;
	MDB_dbi dbi;
	unsigned char keyb = 'k', a = 'a', b = 'b';
	MDB_val key = {1, &keyb}, data = {1, &a};
	int rc;

	if (!mkdtemp(path))
		return 2;
	die("env_create", mdb_env_create(&env));
	die("mapsize", mdb_env_set_mapsize(env, 16u * 1024u * 1024u));
	die("env_open", mdb_env_open(env, path, 0, 0664));
	die("txn_begin", mdb_txn_begin(env, NULL, 0, &txn));
	die("dbi_open", mdb_dbi_open(txn, NULL, MDB_DUPSORT, &dbi));
	die("put a", mdb_put(txn, dbi, &key, &data, 0));
	data.mv_data = &b;
	die("put b", mdb_put(txn, dbi, &key, &data, 0));
	die("commit", mdb_txn_commit(txn));

	die("read txn", mdb_txn_begin(env, NULL, MDB_RDONLY, &txn));
	die("cursor_open", mdb_cursor_open(txn, dbi, &cur));
	/* '0' sorts before both values and is absent; more importantly, 'aa' sorts
	 * strictly between 'a' and 'b' under the default lexicographic comparator. */
	{
		unsigned char missing[2] = {'a', 'a'};
		MDB_val probe = {2, missing};
		MDB_val k = key;
		rc = mdb_cursor_get(cur, &k, &probe, MDB_GET_BOTH);
		if (rc == MDB_SUCCESS) {
			fprintf(stderr, "C3 reproduced: MDB_GET_BOTH returned success for an absent duplicate; returned size=%zu first=0x%02x\n",
				probe.mv_size, probe.mv_size ? ((unsigned char *)probe.mv_data)[0] : 0);
			mdb_cursor_close(cur);
			mdb_txn_abort(txn);
			mdb_env_close(env);
			return 3;
		}
		if (rc != MDB_NOTFOUND) {
			fprintf(stderr, "unexpected rc=%d (%s)\n", rc, mdb_strerror(rc));
			return 4;
		}
	}

	mdb_cursor_close(cur);
	mdb_txn_abort(txn);
	mdb_env_close(env);
	snprintf(file, sizeof(file), "%s/data.mdb", path); unlink(file);
	snprintf(file, sizeof(file), "%s/lock.mdb", path); unlink(file);
	rmdir(path);
	puts("PASS: MDB_GET_BOTH correctly rejects an inexact duplicate match");
	return 0;
}
