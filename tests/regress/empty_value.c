/* Regression: empty LMDB values must not pass a null pointer to memcpy.
 *
 * The normal run verifies LMDB semantics.  When this test is executed by the
 * sanitizer target it also guards the zero-length memcpy hardening in 01-base.
 */
#include "lmdb.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#define DIR "/tmp/aelmdb-empty-value-regress"
#define CHECK(call) do { int rc_ = (call); if (rc_) { \
    fprintf(stderr, "%s failed: %s (%d)\n", #call, mdb_strerror(rc_), rc_); \
    exit(1); } } while (0)

int main(void)
{
    MDB_env *env = NULL;
    MDB_txn *txn = NULL;
    MDB_dbi dbi;
    MDB_val key = { 1, (void *)"k" };
    MDB_val empty = { 0, NULL };
    MDB_val got = { 0, NULL };

    (void)system("rm -rf " DIR);
    if (mkdir(DIR, 0700) && access(DIR, F_OK)) {
        perror("mkdir");
        return 1;
    }

    CHECK(mdb_env_create(&env));
    CHECK(mdb_env_set_mapsize(env, 1u << 20));
    CHECK(mdb_env_open(env, DIR, MDB_NOSYNC | MDB_NOMETASYNC, 0600));
    CHECK(mdb_txn_begin(env, NULL, 0, &txn));
    CHECK(mdb_dbi_open(txn, NULL, 0, &dbi));
    CHECK(mdb_put(txn, dbi, &key, &empty, 0));
    CHECK(mdb_get(txn, dbi, &key, &got));
    if (got.mv_size != 0) {
        fprintf(stderr, "empty value came back with size %zu\n", got.mv_size);
        return 1;
    }
    CHECK(mdb_del(txn, dbi, &key, NULL));
    CHECK(mdb_txn_commit(txn));
    mdb_env_close(env);
    (void)system("rm -rf " DIR);
    puts("empty-value regression passed");
    return 0;
}
