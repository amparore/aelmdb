#include "lmdb.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#define N 30000

static int die(const char *where, int rc) {
    fprintf(stderr, "%s: %s (%d)\n", where, mdb_strerror(rc), rc);
    return 1;
}

static void make_key(char key[32], unsigned i) {
    snprintf(key, 32, "k-%08u-%08x", i, i * 2654435761u);
}

int main(void) {
    MDB_env *env;
    MDB_txn *txn;
    MDB_dbi dbi;
    MDB_val key, data;
    char kbuf[32];
    uint64_t value;
    int rc;

    system("rm -rf testdb_bu3; mkdir testdb_bu3");
    if ((rc = mdb_env_create(&env))) return die("env_create", rc);
    if ((rc = mdb_env_set_mapsize(env, 1ULL << 29))) return die("mapsize", rc);
    if ((rc = mdb_env_open(env, "testdb_bu3", 0, 0664))) return die("env_open", rc);
    if ((rc = mdb_txn_begin(env, NULL, 0, &txn))) return die("txn_begin", rc);
    if ((rc = mdb_dbi_open(txn, NULL, 0, &dbi))) return die("dbi_open", rc);

    for (unsigned i = 0; i < N; ++i) {
        make_key(kbuf, i);
        value = i;
        key.mv_data = kbuf;
        key.mv_size = strlen(kbuf);
        data.mv_data = &value;
        data.mv_size = sizeof(value);
        if ((rc = mdb_put(txn, dbi, &key, &data, 0))) return die("put", rc);
    }
    if ((rc = mdb_txn_commit(txn))) return die("commit insert", rc);

    /* Delete in alternating low/high order to force redistribution and merge
     * on both sides of internal pages, leaving only the final 8 records. */
    unsigned lo = 0, hi = N - 9;
    unsigned deleted = 0;
    while (lo <= hi) {
        if ((rc = mdb_txn_begin(env, NULL, 0, &txn))) return die("txn delete", rc);
        for (unsigned batch = 0; batch < 250 && lo <= hi; ++batch) {
            unsigned i = (batch & 1) ? hi-- : lo++;
            make_key(kbuf, i);
            key.mv_data = kbuf;
            key.mv_size = strlen(kbuf);
            if ((rc = mdb_del(txn, dbi, &key, NULL))) return die("delete", rc);
            ++deleted;
        }
        if ((rc = mdb_txn_commit(txn))) return die("commit delete", rc);
    }

    if (deleted != N - 8) {
        fprintf(stderr, "deleted=%u expected=%u\n", deleted, N - 8);
        return 2;
    }

    if ((rc = mdb_txn_begin(env, NULL, MDB_RDONLY, &txn))) return die("txn verify", rc);
    for (unsigned i = N - 8; i < N; ++i) {
        make_key(kbuf, i);
        key.mv_data = kbuf;
        key.mv_size = strlen(kbuf);
        if ((rc = mdb_get(txn, dbi, &key, &data))) return die("get remaining", rc);
        uint64_t got;
        if (data.mv_size != sizeof(uint64_t)) {
            fprintf(stderr, "bad remaining size %u\n", i);
            return 3;
        }
        memcpy(&got, data.mv_data, sizeof(got));
        if (got != i) {
            fprintf(stderr, "bad remaining value %u\n", i);
            return 3;
        }
    }
    mdb_txn_abort(txn);

    /* Finish deleting everything to exercise the empty-root path too. */
    if ((rc = mdb_txn_begin(env, NULL, 0, &txn))) return die("txn final delete", rc);
    for (unsigned i = N - 8; i < N; ++i) {
        make_key(kbuf, i);
        key.mv_data = kbuf;
        key.mv_size = strlen(kbuf);
        if ((rc = mdb_del(txn, dbi, &key, NULL))) return die("final delete", rc);
    }
    if ((rc = mdb_txn_commit(txn))) return die("commit final delete", rc);

    if ((rc = mdb_txn_begin(env, NULL, MDB_RDONLY, &txn))) return die("txn empty verify", rc);
    make_key(kbuf, N - 1);
    key.mv_data = kbuf;
    key.mv_size = strlen(kbuf);
    rc = mdb_get(txn, dbi, &key, &data);
    if (rc != MDB_NOTFOUND) return die("expected empty", rc);
    mdb_txn_abort(txn);

    mdb_env_close(env);
    printf("BU3 rebalance stress PASS\n");
    return 0;
}
