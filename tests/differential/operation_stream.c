/* diff_ops.c - deterministic operation stream for differential testing.
 *
 * The program drives a pseudo-random but fully deterministic sequence of
 * LMDB API calls and prints one trace line per call: the operation, its
 * arguments (as ids), the return code, the record the cursor ends on, and -
 * after every write - the position of every open cursor.  At checkpoints it
 * adds an ordered digest of every DB and mdb_stat.
 *
 * Two builds of the same program (REF, usually baselines/lmdb, and CAND, the
 * stage under test) must print identical traces:
 *   diff <(REF) <(CAND)
 * The trace captures the observable LMDB semantics that a structural
 * refactoring must preserve: results and error codes, record order, duplicate
 * handling, cursor positions after writes through other cursors (cursor
 * fix-ups), nested transaction commit/abort, drop, overflow values,
 * MDB_APPEND/APPENDDUP/NOOVERWRITE/NODUPDATA/CURRENT/RESERVE/MULTIPLE.
 *
 * With DS_STRUCT=1 the checkpoint lines also carry depth and page counts, and
 * DS_SNAPSHOT_DIR receives a copy of data.mdb after every committed
 * checkpoint, for tools/lmdb_fingerprint.py.  Structural lines are expected to
 * match only between stages that share the page format (lmdb, 01, 02).
 *
 * Aggregate stages: -DDS_AGG=1 creates the DBs with ENTRIES|KEYS|HASHSUM
 * and, when the library was built with -DMDB_DEBUG_AGG_INTEGRITY=1, runs the
 * top-down integrity oracle at every checkpoint (DS_AGG_CHECK=2: after every
 * write).  Such a build must run with DS_VMIN >= MDB_HASH_SIZE and
 * DS_NO_RESERVE=1; its trace equals the trace of a plain LMDB build run with
 * the same two settings.
 *
 * Environment: DS_SEED (default 1), DS_OPS (20000), DS_CHECK_EVERY (500),
 * DS_KEYS (key id space, 600), DS_DIR (testdb_ds), DS_STRUCT, DS_SNAPSHOT_DIR,
 * DS_AGG_CHECK (1), DS_HAZARD (0: re-seat stale cursors before writes),
 * DS_VMIN (1: minimum value length), DS_NO_RESERVE (0), DS_PORTABLE (0),
 * DS_STOP (debugging:
 * commit everything and exit just before op N, leaving the DB in DS_DIR),
 * DS_FOCUS (0).
 *
 * DS_FOCUS=1 (coverage focus, docs/reviews/2026-09-24_copertura_BU7.md): a
 * small key space (DS_KEYS default 48), grow and shrink phases of
 * DS_PHASE ops (default 1500), large dupsets (DS_DUPSPAN distinct values per
 * key, default 1000) and DUPFIXED values of DS_FIXLEN bytes (default 256),
 * so that duplicate trees reach depth 3-4; cursors mostly share one DB and
 * MDB_CURRENT is frequent.  A fourth DB, "intdups" (DUPSORT|DUPFIXED|
 * INTEGERDUP, size_t values), is added unless DS_VMIN > sizeof(size_t).  It reaches the REKEY climb of a duplicate tree,
 * cursor fix-ups of root collapse / emptied trees / moves with several
 * cursors on the same pages, and branch moves and splits in LEAF2 trees.
 * The default stream (DS_FOCUS=0) is unchanged.
 *
 * Aliasing rule (docs/findings/key_aliasing.md): every key and value passed
 * to a write is built in caller-owned buffers, never taken from a read.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <errno.h>
#include "lmdb.h"

#ifndef DS_AGG
#define DS_AGG 0
#endif
#if DS_AGG
# ifndef MDB_AGG_HASHSUM
#  error "DS_AGG requires an aggregate stage header"
# endif
# define DS_SCHEMA (MDB_AGG_ENTRIES | MDB_AGG_KEYS | MDB_AGG_HASHSUM)
# if defined(MDB_DEBUG_AGG_INTEGRITY) && MDB_DEBUG_AGG_INTEGRITY
#  define DS_INTEGRITY(t, d) mdb_agg_check_integrity((t), (d))
# endif
#else
# define DS_SCHEMA 0
#endif

#define NDB 4			/* DBs 0-2 always; 3 only with DS_FOCUS */
#define NCUR 3
#define FIXLEN (fixlen ? fixlen : vmin > 16 ? vmin : 16)

static const char *dbname[NDB] = { "plain", "dups", "fixed", "intdups" };
static const unsigned dbflags[NDB] = { 0, MDB_DUPSORT, MDB_DUPSORT | MDB_DUPFIXED,
	MDB_DUPSORT | MDB_DUPFIXED | MDB_INTEGERDUP };
static int ndb = 3;

static MDB_env *env;
static MDB_txn *txn_stack[4];
static int depth;			/* number of live txns in txn_stack */
static MDB_dbi dbi[NDB];
static MDB_cursor *cur[NCUR];
static int curdb[NCUR];
/* 0 after a FAILED absolute positioning (SET*, GET_BOTH*): LMDB leaves the
 * cursor (and its sub-cursor) in an unspecified state, and a following
 * relative op or MDB_GET_CURRENT may read stale memory - observed on LMDB
 * 0.9.70 itself (PREV_DUP after a failed GET_BOTH_RANGE returns garbage).
 * Such a cursor is used again only after an absolute op succeeds. */
static int curok[NCUR];
/* Write generations.  LMDB 0.9.70 does not keep a cursor's DUPSORT
 * sub-cursor coherent in every case: after mdb_cursor_del moved the cursor
 * onto the next key, or after another cursor added duplicates with
 * MDB_MULTIPLE to the dupset it is on, a following mdb_cursor_del through it
 * corrupts md_entries (docs/findings/lmdb_cursor_hazards.md).  By default a
 * cursor that has not been positioned by an absolute op (or by its own put)
 * since the last write is re-seated (MDB_SET / MDB_GET_BOTH on its current
 * record) before it is used for a write; relative moves (NEXT, PREV, ...)
 * inside a dupset do not refresh the sub-cursor's MDB_db copy.
 *
 * The default mode also re-seats a cursor after its own successful put
 * (H5), after a delete through it that moved it onto another key (H4), and
 * before MDB_FIRST_DUP / MDB_LAST_DUP when it is stale (H6).
 *
 * DS_HAZARD=1 disables all these guards (curok[] and the re-seats) and exercises
 * those paths.  01-base corrects H1 and H2 (B1), so in hazard mode the
 * reference is 01-base: LMDB 0.9.70 is expected to diverge (and may read
 * stale memory), every later lineage stage must match 01-base. */
static unsigned long write_gen, pos_gen[NCUR];
static unsigned hazard;
/* value-shape policy, part of the trace: an aggregate build and its LMDB
 * reference must use the same DS_VMIN / DS_NO_RESERVE */
static size_t vmin = 1;
static unsigned no_reserve;
/* DS_PORTABLE=1: no DUP-only cursor ops on the non-DUPSORT DB (LMDB answers
 * MDB_INCOMPATIBLE/EINVAL there; the AELMDB branch does not) */
static unsigned portable;
/* DS_FOCUS: see the header */
static unsigned focus, phase_len = 1500, dupspan = 1000;
static size_t fixlen;
static unsigned long opno;
static unsigned nkeys = 600, check_every = 500, show_struct, agg_check = 1;
static const char *dir = "testdb_ds", *snapdir;
static size_t maxkey;
static uint64_t rng;

#define DIE(...) do { fprintf(stderr, "FATAL op %lu: ", opno); \
	fprintf(stderr, __VA_ARGS__); fputc('\n', stderr); exit(2); } while (0)
#define C(x) do { int r_ = (x); if (r_) DIE("%s: %s", #x, mdb_strerror(r_)); } while (0)

static uint64_t
mix64(uint64_t x)
{
	x ^= x >> 30; x *= UINT64_C(0xbf58476d1ce4e5b9);
	x ^= x >> 27; x *= UINT64_C(0x94d049bb133111eb);
	x ^= x >> 31;
	return x;
}

static unsigned
rnd(unsigned n)
{
	rng += UINT64_C(0x9e3779b97f4a7c15);
	return (unsigned)(mix64(rng) % n);
}

/* ---------------------------------------------------------------- content */

/* key id -> length: mostly short, some medium, some near maxkey */
static size_t
key_len(unsigned id)
{
	unsigned c = (unsigned)(mix64(id * 7919u + 17) % 10);
	if (c < 6) return 4 + (unsigned)(mix64(id) % 13);
	if (c < 9) return 40 + (unsigned)(mix64(id + 1) % 160);
	return maxkey - (unsigned)(mix64(id + 2) % 32);
}

static void
make_key(unsigned char *b, size_t len, unsigned id)
{
	size_t i;
	uint64_t x = mix64(id ^ 0x6b6579);
	b[0] = (unsigned char)(id >> 24); b[1] = (unsigned char)(id >> 16);
	b[2] = (unsigned char)(id >> 8); b[3] = (unsigned char)id;
	for (i = 4; i < len; i++) {
		if ((i & 7) == 0) x = mix64(x + i);
		b[i] = (unsigned char)(x >> ((i & 7) * 8));
	}
}

/* value = (version, length): first 4 bytes carry the version for traces */
static void
make_val(unsigned char *b, size_t len, unsigned ver)
{
	size_t i;
	uint64_t x = mix64(ver ^ 0x76616c);
	for (i = 0; i < len; i++) {
		if ((i & 7) == 0) x = mix64(x + i);
		b[i] = (unsigned char)(x >> ((i & 7) * 8));
	}
	if (len >= 4) {
		b[0] = (unsigned char)(ver >> 24); b[1] = (unsigned char)(ver >> 16);
		b[2] = (unsigned char)(ver >> 8); b[3] = (unsigned char)ver;
	}
}

static size_t
plain_vlen(void)
{
	unsigned c = rnd(20);
	size_t n;
	if (c < 12) n = 1 + rnd(64);
	else if (c < 18) n = 100 + rnd(1400);
	else n = 3000 + rnd(9000);		/* overflow pages */
	return n < vmin ? vmin : n;
}

static size_t
dup_vlen(unsigned ver)
{
	/* a dup value's length is a function of its version: equal versions
	 * are equal values, so NODUPDATA/GET_BOTH hit */
	unsigned c = (unsigned)(mix64(ver) % 10);
	size_t n = c < 7 ? 4 + (unsigned)(mix64(ver + 3) % 40) :
		60 + (unsigned)(mix64(ver + 5) % (maxkey - 60));
	return n < vmin ? vmin : n;
}

static uint64_t
digest_bytes(uint64_t h, const void *p, size_t n)
{
	const unsigned char *b = p;
	size_t i;
	for (i = 0; i < n; i++)
		h = (h ^ b[i]) * UINT64_C(0x100000001b3);
	return h;
}

/* short description of a record: key id/length, value version/length */
static void
fmt_rec(char *out, size_t sz, const MDB_val *k, const MDB_val *v)
{
	unsigned kid = 0, ver = 0;
	const unsigned char *kb = k->mv_data, *vb = v->mv_data;
	if (kb && k->mv_size >= 4)
		kid = (unsigned)kb[0] << 24 | (unsigned)kb[1] << 16 | (unsigned)kb[2] << 8 | kb[3];
	if (v->mv_size >= 4)
		ver = (unsigned)vb[0] << 24 | (unsigned)vb[1] << 16 | (unsigned)vb[2] << 8 | vb[3];
	snprintf(out, sz, "k%u/%zu v%u/%zu#%04x", kid, k->mv_size, ver, v->mv_size,
		(unsigned)(digest_bytes(0, v->mv_data, v->mv_size) & 0xffff));
}

/* ------------------------------------------------------------ transactions */

static MDB_txn *
txn(void)
{
	return txn_stack[depth - 1];
}

static void
open_cursors(void)
{
	int i;
	for (i = 0; i < NCUR; i++) {
		curdb[i] = focus ? (int)(opno / 97 % ndb) : i % ndb;
		curok[i] = 1;
		C(mdb_cursor_open(txn(), dbi[curdb[i]], &cur[i]));
	}
}

static void
close_cursors(void)
{
	int i;
	for (i = 0; i < NCUR; i++) {
		mdb_cursor_close(cur[i]);
		cur[i] = NULL;
	}
}

static void
begin_top(void)
{
	int i;
	C(mdb_txn_begin(env, NULL, 0, &txn_stack[0]));
	depth = 1;
	for (i = 0; i < ndb; i++)
		C(mdb_dbi_open(txn(), dbname[i], MDB_CREATE | dbflags[i] | DS_SCHEMA, &dbi[i]));
#if DS_AGG
	/* the hash offset may be set only while a DB is empty */
	for (i = 0; i < ndb; i++) {
		MDB_stat st;
		C(mdb_stat(txn(), dbi[i], &st));
		if (!st.ms_entries)
			C(mdb_set_hash_offset(txn(), dbi[i], 0));
	}
#endif
	open_cursors();
}

static void
snapshot(const char *tag)
{
	static unsigned seq;
	char src[1024], dst[1024], buf[65536];
	FILE *in, *out;
	size_t n;
	if (!snapdir)
		return;
	snprintf(src, sizeof(src), "%s/data.mdb", dir);
	snprintf(dst, sizeof(dst), "%s/%04u-%s.mdb", snapdir, seq++, tag);
	in = fopen(src, "rb"); out = fopen(dst, "wb");
	if (!in || !out) DIE("snapshot %s", dst);
	while ((n = fread(buf, 1, sizeof(buf), in)) > 0)
		if (fwrite(buf, 1, n, out) != n) DIE("snapshot write");
	fclose(in); fclose(out);
}

/* --------------------------------------------------------------- checkers */

static void
agg_integrity(void)
{
#ifdef DS_INTEGRITY
	int i;
	for (i = 0; i < ndb; i++) {
		int rc = DS_INTEGRITY(txn(), dbi[i]);
		if (rc) DIE("aggregate integrity of %s: %s", dbname[i], mdb_strerror(rc));
	}
#endif
}

static void
checkpoint(const char *why)
{
	int i;
	for (i = 0; i < ndb; i++) {
		MDB_cursor *c;
		MDB_val k, v;
		MDB_stat st;
		uint64_t h = UINT64_C(0xcbf29ce484222325), n = 0;
		int rc;
		C(mdb_cursor_open(txn(), dbi[i], &c));
		for (rc = mdb_cursor_get(c, &k, &v, MDB_FIRST); rc == 0;
			rc = mdb_cursor_get(c, &k, &v, MDB_NEXT)) {
			h = digest_bytes(h, &k.mv_size, sizeof(k.mv_size));
			h = digest_bytes(h, k.mv_data, k.mv_size);
			h = digest_bytes(h, &v.mv_size, sizeof(v.mv_size));
			h = digest_bytes(h, v.mv_data, v.mv_size);
			n++;
		}
		if (rc != MDB_NOTFOUND) DIE("scan %s: %s", dbname[i], mdb_strerror(rc));
		mdb_cursor_close(c);
		C(mdb_stat(txn(), dbi[i], &st));
		if (st.ms_entries != n) DIE("%s: ms_entries %zu, scan %llu", dbname[i],
			(size_t)st.ms_entries, (unsigned long long)n);
		printf("CHECK %s %s n=%llu h=%016llx", why, dbname[i],
			(unsigned long long)n, (unsigned long long)h);
		if (show_struct)
			printf(" depth=%u branch=%zu leaf=%zu ovf=%zu", st.ms_depth,
				(size_t)st.ms_branch_pages, (size_t)st.ms_leaf_pages,
				(size_t)st.ms_overflow_pages);
		putchar('\n');
	}
	if (agg_check)
		agg_integrity();
}

/* position of every cursor after a write (cursor fix-up semantics) */
static void
trace_cursors(void)
{
	int i;
	char rec[96];
	write_gen++;
	for (i = 0; i < NCUR; i++) {
		MDB_val k, v;
		int rc;
		if (!curok[i]) {
			printf(" c%d=unset", i);
			continue;
		}
		rc = mdb_cursor_get(cur[i], &k, &v, MDB_GET_CURRENT);
		if (rc == 0) {
			fmt_rec(rec, sizeof(rec), &k, &v);
			printf(" c%d=%s", i, rec);
		} else {
			printf(" c%d=rc%d", i, rc);
		}
	}
}

/* ------------------------------------------------------------- operations */

static unsigned char kbuf[4096], vbuf[16384];
static unsigned ver_counter = 1;

static int
pick_key(MDB_val *k)
{
	unsigned id = rnd(nkeys);
	k->mv_size = key_len(id);
	k->mv_data = kbuf;
	make_key(kbuf, k->mv_size, id);
	return (int)id;
}

/* value for db i; dup versions come from a small per-key set so that exact
 * duplicates recur */
static unsigned
pick_val(int db, unsigned kid, MDB_val *v)
{
	unsigned ver;
	if (db == 0) {
		ver = ver_counter++;
		v->mv_size = plain_vlen();
	} else if (db == 1) {
		ver = focus ? kid * 4096 + rnd(dupspan) : kid * 64 + rnd(24);
		v->mv_size = dup_vlen(ver);
	} else if (db == 2) {
		ver = focus ? kid * 4096 + rnd(dupspan) : kid * 64 + rnd(40);
		v->mv_size = FIXLEN;
	} else {
		/* INTEGERDUP: native size_t values */
		ver = kid * 4096 + rnd(dupspan);
		v->mv_size = sizeof(size_t);
	}
	v->mv_data = vbuf;
	make_val(vbuf, v->mv_size, ver);
	return ver;
}

static const unsigned get_ops[] = {
	MDB_FIRST, MDB_LAST, MDB_NEXT, MDB_PREV, MDB_NEXT_DUP, MDB_PREV_DUP,
	MDB_NEXT_NODUP, MDB_PREV_NODUP, MDB_FIRST_DUP, MDB_LAST_DUP,
	MDB_SET, MDB_SET_KEY, MDB_SET_RANGE, MDB_GET_BOTH, MDB_GET_BOTH_RANGE,
	MDB_GET_CURRENT
};

static void
op_cursor_get(void)
{
	int ci = rnd(NCUR), rc;
	unsigned op = get_ops[rnd(sizeof(get_ops) / sizeof(get_ops[0]))];
	MDB_val k = { 0, NULL }, v = { 0, NULL };
	char rec[96];
	int kid = -1;
	/* DUP ops on a non-dup DB return EINVAL in LMDB: keep them, rc is traced */
	if (op == MDB_SET || op == MDB_SET_KEY || op == MDB_SET_RANGE ||
		op == MDB_GET_BOTH || op == MDB_GET_BOTH_RANGE) {
		kid = pick_key(&k);
		if (op == MDB_GET_BOTH || op == MDB_GET_BOTH_RANGE) {
			if (curdb[ci] == 0) {
				op = MDB_SET;		/* GET_BOTH needs DUPSORT */
			} else {
				pick_val(curdb[ci], (unsigned)kid, &v);
			}
		}
	}
	if (portable && curdb[ci] == 0 && (op == MDB_FIRST_DUP || op == MDB_LAST_DUP ||
		op == MDB_NEXT_DUP || op == MDB_PREV_DUP))
		op = op == MDB_FIRST_DUP ? MDB_FIRST : op == MDB_LAST_DUP ? MDB_LAST :
			op == MDB_NEXT_DUP ? MDB_NEXT : MDB_PREV;
	if (!curok[ci] && kid < 0 && op != MDB_FIRST && op != MDB_LAST)
		op = rnd(2) ? MDB_FIRST : MDB_LAST;	/* reposition first */
	if ((op == MDB_FIRST_DUP || op == MDB_LAST_DUP) && curdb[ci]) {
		/* H6 (docs/findings/lmdb_cursor_hazards.md): these re-search the
		 * duplicate tree from the sub-cursor's copy of its root, which
		 * a write through another cursor may have made stale (LMDB
		 * 0.9.70 then reads an old page; 01-base B1f refreshes it).
		 * Default mode re-seats a stale cursor on its key first (these
		 * ops depend only on the key); a cursor without a current key
		 * is repositioned with MDB_FIRST / MDB_LAST instead. */
		if (!hazard && curok[ci] && pos_gen[ci] != write_gen) {
			static unsigned char gkb[4096];
			MDB_val gk = { 0, NULL }, gv;
			int grc = mdb_cursor_get(cur[ci], &gk, &gv, MDB_GET_CURRENT);
			if ((grc == 0 || grc == MDB_NOTFOUND || grc == EINVAL) && gk.mv_data) {
				memcpy(gkb, gk.mv_data, gk.mv_size);
				gk.mv_data = gkb;
				if (mdb_cursor_get(cur[ci], &gk, &gv, MDB_SET))
					DIE("re-seat of c%d before op %u failed", ci, op);
				pos_gen[ci] = write_gen;
			} else {
				op = op == MDB_FIRST_DUP ? MDB_FIRST : MDB_LAST;
			}
		}
	}
	rc = mdb_cursor_get(cur[ci], &k, &v, op);
	if (kid >= 0 || op == MDB_FIRST || op == MDB_LAST) {
		curok[ci] = rc == 0 || hazard;
		if (rc == 0)	/* absolute ops re-initialise the sub-cursor */
			pos_gen[ci] = write_gen;
	}
	printf("%lu get c%d(%s) op%u", opno, ci, dbname[curdb[ci]], op);
	if (kid >= 0) printf(" k%d", kid);
	if (rc == 0) {
		/* LMDB does not return the key for FIRST_DUP/LAST_DUP (AELMDB
		 * does, for aggregate DBs): the key is not part of the contract */
		if (op == MDB_FIRST_DUP || op == MDB_LAST_DUP) {
			k.mv_size = 0;
			k.mv_data = NULL;
		}
		fmt_rec(rec, sizeof(rec), &k, &v);
		printf(" -> %s\n", rec);
	} else {
		printf(" -> rc%d\n", rc);
	}
}

static void
op_put(void)
{
	int db = rnd(ndb), kid, rc;
	unsigned flags = 0, ver, r = rnd(20);
	MDB_val k, v;
	kid = pick_key(&k);
	ver = pick_val(db, (unsigned)kid, &v);
	if (r < 3) flags = MDB_NOOVERWRITE;
	else if (r < 6 && db) flags = MDB_NODUPDATA;
	else if (r == 6) flags = MDB_APPEND;
	else if (r == 7 && db) flags = MDB_APPENDDUP;
	else if (r == 8 && db == 0 && !no_reserve) flags = MDB_RESERVE;
	if (flags == MDB_RESERVE) {
		size_t n = v.mv_size;
		rc = mdb_put(txn(), dbi[db], &k, &v, flags);
		if (rc == 0) {
			if (v.mv_size != n) DIE("RESERVE size");
			make_val(v.mv_data, n, ver);
		}
	} else {
		rc = mdb_put(txn(), dbi[db], &k, &v, flags);
	}
	printf("%lu put %s k%d v%u/%zu f%x -> rc%d", opno, dbname[db], kid, ver,
		v.mv_size, flags, rc);
	if (rc == MDB_KEYEXIST && (flags & MDB_NOOVERWRITE) && db == 0) {
		char rec[96];
		MDB_val kk = { k.mv_size, kbuf };
		fmt_rec(rec, sizeof(rec), &kk, &v);	/* v holds the existing data */
		printf(" existing=%s", rec);
	}
	trace_cursors();
	putchar('\n');
}

static void
op_del(void)
{
	/* focus: deleting a whole dupset drops the duplicate tree without
	 * rebalancing it; keep that rare so that duplicate trees shrink */
	int db = rnd(ndb), kid, rc, withdata = db && (focus ? rnd(8) != 0 : rnd(2));
	unsigned ver = 0;
	MDB_val k, v;
	kid = pick_key(&k);
	if (withdata)
		ver = pick_val(db, (unsigned)kid, &v);
	rc = mdb_del(txn(), dbi[db], &k, withdata ? &v : NULL);
	printf("%lu del %s k%d", opno, dbname[db], kid);
	if (withdata) printf(" v%u", ver);
	printf(" -> rc%d", rc);
	trace_cursors();
	putchar('\n');
}

/* current record of cursor ci, re-seated if stale; rc != 0: unpositioned */
static int
current_for_write(int ci, MDB_val *ck, MDB_val *cv)
{
	static unsigned char rk[4096], rv[16384];
	MDB_val k2, v2;
	int rc;
	if (!curok[ci])
		return EINVAL;
	rc = mdb_cursor_get(cur[ci], ck, cv, MDB_GET_CURRENT);
	if (rc || hazard || pos_gen[ci] == write_gen)
		return rc;
	memcpy(rk, ck->mv_data, ck->mv_size);
	memcpy(rv, cv->mv_data, cv->mv_size);
	k2.mv_size = ck->mv_size; k2.mv_data = rk;
	v2.mv_size = cv->mv_size; v2.mv_data = rv;
	rc = mdb_cursor_get(cur[ci], &k2, &v2, curdb[ci] ? MDB_GET_BOTH : MDB_SET);
	if (rc)
		DIE("re-seat of c%d failed: %s", ci, mdb_strerror(rc));
	pos_gen[ci] = write_gen;
	return mdb_cursor_get(cur[ci], ck, cv, MDB_GET_CURRENT);
}

/* Default mode: after a put through cursor ci, re-seat it on the item it
 * reports.  LMDB 0.9.70 leaves C_EOF set after inserting a key beyond the
 * last one when the search went through the slow path, so a later
 * MDB_NEXT stops even if the dupset grew in the meantime (H5 in
 * docs/findings/lmdb_cursor_hazards.md; 01-base B1e clears it). */
static void
reseat_after_put(int ci)
{
	static unsigned char rk[4096], rv[16384];
	MDB_val k, v;
	if (hazard || mdb_cursor_get(cur[ci], &k, &v, MDB_GET_CURRENT))
		return;
	memcpy(rk, k.mv_data, k.mv_size);
	memcpy(rv, v.mv_data, v.mv_size);
	k.mv_data = rk;
	v.mv_data = rv;
	if (mdb_cursor_get(cur[ci], &k, &v, curdb[ci] ? MDB_GET_BOTH : MDB_SET))
		DIE("re-seat of c%d after put failed", ci);
}

static void
op_cursor_put(void)
{
	int ci = rnd(NCUR), db = curdb[ci], rc, kid;
	unsigned flags = 0, ver, r = rnd(10);
	MDB_val k, v;
	if (focus && db && r >= 4 && r < 7) {
		/* focus: re-seat on a random existing duplicate first, so that
		 * MDB_CURRENT lands anywhere in deep duplicate trees */
		MDB_val sk, sv;
		kid = pick_key(&sk);
		pick_val(db, (unsigned)kid, &sv);
		rc = mdb_cursor_get(cur[ci], &sk, &sv, MDB_GET_BOTH_RANGE);
		curok[ci] = rc == 0 || hazard;
		if (rc == 0)
			pos_gen[ci] = write_gen;
		printf("%lu seek c%d(%s) k%d -> rc%d\n", opno, ci, dbname[db], kid, rc);
		r = 0;
	}
	if (r < 4) {
		/* MDB_CURRENT: replace the current item; the key is copied into
		 * kbuf first (aliasing rule) */
		MDB_val ck, cv;
		rc = current_for_write(ci, &ck, &cv);
		if (rc) {
			printf("%lu cput c%d(%s) CURRENT -> unpositioned rc%d\n", opno, ci, dbname[db], rc);
			return;
		}
		memcpy(kbuf, ck.mv_data, ck.mv_size);
		k.mv_size = ck.mv_size; k.mv_data = kbuf;
		kid = (int)((unsigned)kbuf[0] << 24 | (unsigned)kbuf[1] << 16 |
			(unsigned)kbuf[2] << 8 | kbuf[3]);
		if (db == 0) {
			ver = pick_val(db, (unsigned)kid, &v);
		} else {
			/* DUPSORT: the new data must sort identically; LMDB
			 * requires equal size, so re-put the same data */
			memcpy(vbuf, cv.mv_data, cv.mv_size);
			v.mv_size = cv.mv_size; v.mv_data = vbuf;
			ver = 0;
		}
		flags = MDB_CURRENT;
	} else {
		kid = pick_key(&k);
		ver = pick_val(db, (unsigned)kid, &v);
		if (r == 4) flags = MDB_NOOVERWRITE;
		else if (r == 5 && db) flags = MDB_NODUPDATA;
		else if (r == 6) flags = MDB_APPEND;
		else if (r == 7 && db) flags = MDB_APPENDDUP;
	}
	rc = mdb_cursor_put(cur[ci], &k, &v, flags);
	if (flags != MDB_CURRENT)
		curok[ci] = rc == 0 || hazard;
	if (rc == 0 && flags != MDB_CURRENT)
		reseat_after_put(ci);
	printf("%lu cput c%d(%s) k%d v%u/%zu f%x -> rc%d", opno, ci, dbname[db], kid,
		ver, v.mv_size, flags, rc);
	trace_cursors();
	if (rc == 0 && flags != MDB_CURRENT)
		pos_gen[ci] = write_gen;
	putchar('\n');
}

static void
op_cursor_del(void)
{
	int ci = rnd(NCUR), db = curdb[ci], rc;
	unsigned flags = db && rnd(focus ? 12 : 3) == 0 ? MDB_NODUPDATA : 0;
	MDB_val ck, cv;
	char rec[96] = "-";
	static unsigned char dk[4096];
	size_t dksz;
	rc = current_for_write(ci, &ck, &cv);
	if (rc) {
		printf("%lu cdel c%d(%s) -> unpositioned rc%d\n", opno, ci, dbname[db], rc);
		return;
	}
	fmt_rec(rec, sizeof(rec), &ck, &cv);
	memcpy(dk, ck.mv_data, ck.mv_size);
	dksz = ck.mv_size;
	rc = mdb_cursor_del(cur[ci], flags);
	if (rc == 0 && db && !hazard) {
		/* H4 (docs/findings/lmdb_cursor_hazards.md): when the delete
		 * removed the key and moved the cursor onto the next one, LMDB
		 * 0.9.70 keeps the sub-cursor of an earlier key and reports a
		 * duplicate at a stale index, or another key's value.  01-base
		 * (B1d) puts it on the first duplicate.  The reference run
		 * re-seats the cursor on its new key (first duplicate). */
		MDB_val nk = { 0, NULL }, nv;
		int grc = mdb_cursor_get(cur[ci], &nk, &nv, MDB_GET_CURRENT);
		/* LMDB sets the key even when the stale sub-cursor answers
		 * MDB_NOTFOUND or EINVAL for the data */
		if ((grc == 0 || grc == MDB_NOTFOUND || grc == EINVAL) && nk.mv_data &&
			(nk.mv_size != dksz || memcmp(nk.mv_data, dk, dksz))) {
			memcpy(dk, nk.mv_data, nk.mv_size);
			nk.mv_data = dk;
			if (mdb_cursor_get(cur[ci], &nk, &nv, MDB_SET))
				DIE("re-seat of c%d after cdel failed", ci);
			pos_gen[ci] = write_gen + 1;	/* trace_cursors bumps it */
		}
	}
	printf("%lu cdel c%d(%s) at %s f%x -> rc%d", opno, ci, dbname[db], rec, flags, rc);
	trace_cursors();
	putchar('\n');
}

static void
op_multiple(void)
{
	/* MDB_MULTIPLE on the DUPFIXED DB: a run of fixed-size values */
	int ci, kid, rc;
	unsigned n = 1 + rnd(12), i, base;
	MDB_val k, v[2];
	for (ci = 0; ci < NCUR && curdb[ci] != 2; ci++)
		;
	if (ci == NCUR) {			/* no cursor on "fixed": rebind one */
		ci = rnd(NCUR);
		mdb_cursor_close(cur[ci]);
		curdb[ci] = 2;
		curok[ci] = 1;
		C(mdb_cursor_open(txn(), dbi[2], &cur[ci]));
	}
	kid = pick_key(&k);
	base = (unsigned)kid * 64 + rnd(40);
	for (i = 0; i < n; i++)
		make_val(vbuf + (size_t)i * FIXLEN, FIXLEN, base + i);
	v[0].mv_size = FIXLEN; v[0].mv_data = vbuf;
	v[1].mv_size = n; v[1].mv_data = NULL;
	rc = mdb_cursor_put(cur[ci], &k, v, MDB_MULTIPLE);
	curok[ci] = rc == 0 || hazard;
	if (rc == 0)
		reseat_after_put(ci);
	printf("%lu mput c%d(fixed) k%d v%u..%u -> rc%d written=%zu", opno, ci, kid,
		base, base + n - 1, rc, v[1].mv_size);
	trace_cursors();
	if (rc == 0)
		pos_gen[ci] = write_gen;
	putchar('\n');
}

/* focus only: at every phase boundary, grow one dupset of a DUPSORT DB
 * by a few hundred to a few thousand values (duplicate trees of depth 3-4);
 * phase_len/8 ops later, shrink it by deleting random values or a run of
 * values through a cursor (branch-level rebalance,
 * moves and merges of LEAF2 branch pages, root collapse), with the other
 * cursors bound to the same DB */
static void
op_bulk(int shrink)
{
	static int last_db = 1, last_kid = -1;
	int db = 1 + rnd(ndb - 1), ci = rnd(NCUR), kid, rc = 0, i;
	unsigned n, done = 0, base, span;
	MDB_val k, v[2];
	if (shrink && last_kid >= 0)
		db = last_db;		/* shrink the dupset grown last */
	for (i = 0; i < NCUR; i++) {		/* every cursor on this DB */
		if (curdb[i] != db) {
			mdb_cursor_close(cur[i]);
			curdb[i] = db;
			curok[i] = 1;
			C(mdb_cursor_open(txn(), dbi[db], &cur[i]));
		}
	}
	kid = pick_key(&k);
	if (shrink && last_kid >= 0) {
		kid = last_kid;
		k.mv_size = key_len((unsigned)kid);
		make_key(kbuf, k.mv_size, (unsigned)kid);
	}
	if (!shrink) {
		last_db = db;
		last_kid = kid;
		n = 300 + rnd(2700);
		span = dupspan > n ? dupspan : n;
		if (span > 4096) span = 4096;
		base = (unsigned)kid * 4096 + rnd(span - n + 1);
		while (done < n && !rc) {
			if (db >= 2) {
				size_t vs = db == 2 ? FIXLEN : sizeof(size_t);
				unsigned m = (unsigned)(sizeof(vbuf) / vs), j;
				if (m > 64) m = 64;
				if (m > n - done) m = n - done;
				for (j = 0; j < m; j++)
					make_val(vbuf + (size_t)j * vs, vs, base + done + j);
				v[0].mv_size = vs; v[0].mv_data = vbuf;
				v[1].mv_size = m; v[1].mv_data = NULL;
				rc = mdb_cursor_put(cur[ci], &k, v, MDB_MULTIPLE);
				done += (unsigned)v[1].mv_size;
				if (!rc && v[1].mv_size != m) DIE("bulk MULTIPLE wrote %zu of %u", v[1].mv_size, m);
			} else {
				v[0].mv_size = dup_vlen(base + done);
				v[0].mv_data = vbuf;
				make_val(vbuf, v[0].mv_size, base + done);
				rc = mdb_cursor_put(cur[ci], &k, v, 0);
				done++;
			}
		}
		curok[ci] = rc == 0 || hazard;
		if (rc == 0)
			reseat_after_put(ci);
		printf("%lu bulk-grow c%d(%s) k%d v%u.. n=%u -> rc%d", opno, ci, dbname[db],
			kid, base, done, rc);
	} else {
		mdb_size_t cnt = 0, cnt0;
		MDB_val dv;
		rc = mdb_cursor_get(cur[ci], &k, &dv, MDB_SET_KEY);
		if (rc == 0)
			rc = mdb_cursor_count(cur[ci], &cnt);
		cnt0 = cnt;
		if (rc == 0 && cnt > 1 && rnd(2)) {
			/* random duplicates (GET_BOTH_RANGE on a random version),
			 * never the last one */
			unsigned todo = 1 + rnd((unsigned)cnt - 1), ver;
			size_t ksz = k.mv_size;
			MDB_val bv;
			while (done < todo && !rc) {
				k.mv_size = ksz;	/* caller-owned key (aliasing rule) */
				k.mv_data = kbuf;
				ver = (unsigned)kid * 4096 + rnd(4096);
				bv.mv_size = db == 2 ? FIXLEN : db == 3 ? sizeof(size_t) : dup_vlen(ver);
				bv.mv_data = vbuf;
				make_val(vbuf, bv.mv_size, ver);
				rc = mdb_cursor_get(cur[ci], &k, &bv, MDB_GET_BOTH_RANGE);
				if (rc == MDB_NOTFOUND) {
					/* beyond the last duplicate: take the first one */
					k.mv_size = ksz;
					k.mv_data = kbuf;
					rc = mdb_cursor_get(cur[ci], &k, &bv, MDB_SET_KEY);
				}
				if (!rc)
					rc = mdb_cursor_count(cur[ci], &cnt);
				if (rc || cnt < 2)
					break;
				rc = mdb_cursor_del(cur[ci], 0);
				done++;
			}
		} else if (rc == 0 && cnt > 1) {
			/* from a random duplicate, delete a run that stops before
			 * the last one: the cursor never leaves the dupset */
			unsigned skip = rnd((unsigned)cnt - 1), todo;
			for (i = 0; (unsigned)i < skip && !rc; i++)
				rc = mdb_cursor_get(cur[ci], &k, &dv, MDB_NEXT_DUP);
			todo = rc ? 0 : 1 + rnd((unsigned)cnt - 1);
			rc = 0;
			if (todo > cnt - skip - 1) todo = (unsigned)cnt - skip - 1;
			while (done < todo && !rc) {
				rc = mdb_cursor_del(cur[ci], 0);
				done++;
			}
		}
		curok[ci] = rc == 0 || hazard;
		pos_gen[ci] = write_gen + 1;
		printf("%lu bulk-shrink c%d(%s) k%d n=%u of %zu -> rc%d", opno, ci, dbname[db],
			kid, done, (size_t)cnt0, rc);
	}
	trace_cursors();
	putchar('\n');
}

static void
op_rebind(void)
{
	/* move a cursor to another DB (mdb_cursor_renew needs a read txn;
	 * close/open is the write-txn equivalent) */
	int ci = rnd(NCUR), db = rnd(ndb);
	if (focus && rnd(4))
		db = curdb[(ci + 1 + rnd(NCUR - 1)) % NCUR];	/* share a DB */
	mdb_cursor_close(cur[ci]);
	curdb[ci] = db;
	curok[ci] = 1;
	C(mdb_cursor_open(txn(), dbi[db], &cur[ci]));
	printf("%lu rebind c%d -> %s\n", opno, ci, dbname[db]);
}

static void
op_txn(void)
{
	unsigned r = rnd(10);
	if (r < 4 && depth < 4) {
		MDB_txn *child;
		close_cursors();
		C(mdb_txn_begin(env, txn(), 0, &child));
		txn_stack[depth++] = child;
		open_cursors();
		printf("%lu txn begin-nested depth=%d\n", opno, depth);
	} else if (depth > 1) {
		int commit = r < 8;
		close_cursors();
		if (commit) C(mdb_txn_commit(txn()));
		else mdb_txn_abort(txn());
		depth--;
		open_cursors();
		printf("%lu txn %s-nested depth=%d\n", opno, commit ? "commit" : "abort", depth);
	} else {
		int commit = r != 9;
		close_cursors();
		if (commit) C(mdb_txn_commit(txn()));
		else mdb_txn_abort(txn());
		printf("%lu txn %s-top\n", opno, commit ? "commit" : "abort");
		depth = 0;
		if (commit) snapshot("commit");
		begin_top();
	}
}

static void
op_drop(void)
{
	int db = rnd(ndb);
	C(mdb_drop(txn(), dbi[db], 0));
	printf("%lu drop %s", opno, dbname[db]);
	trace_cursors();
	putchar('\n');
}

int
main(void)
{
	unsigned long nops, stop_at = 0;
	char cmd[1200];
	const char *e;

	rng = mix64(strtoull((e = getenv("DS_SEED")) ? e : "1", NULL, 0));
	nops = strtoul((e = getenv("DS_OPS")) ? e : "20000", NULL, 0);
	if ((e = getenv("DS_CHECK_EVERY"))) check_every = (unsigned)strtoul(e, NULL, 0);
	if ((e = getenv("DS_KEYS"))) nkeys = (unsigned)strtoul(e, NULL, 0);
	if ((e = getenv("DS_DIR"))) dir = e;
	if ((e = getenv("DS_STRUCT"))) show_struct = (unsigned)strtoul(e, NULL, 0);
	if ((e = getenv("DS_AGG_CHECK"))) agg_check = (unsigned)strtoul(e, NULL, 0);
	if ((e = getenv("DS_PORTABLE"))) portable = (unsigned)strtoul(e, NULL, 0);
	if ((e = getenv("DS_STOP"))) stop_at = strtoul(e, NULL, 0);
	if ((e = getenv("DS_HAZARD"))) hazard = (unsigned)strtoul(e, NULL, 0);
	if ((e = getenv("DS_VMIN"))) vmin = (size_t)strtoul(e, NULL, 0);
	if ((e = getenv("DS_NO_RESERVE"))) no_reserve = (unsigned)strtoul(e, NULL, 0);
	if ((e = getenv("DS_FOCUS"))) focus = (unsigned)strtoul(e, NULL, 0);
	if (focus) {
		nkeys = 48;
		fixlen = 256;
		if ((e = getenv("DS_KEYS"))) nkeys = (unsigned)strtoul(e, NULL, 0);
		if ((e = getenv("DS_PHASE"))) phase_len = (unsigned)strtoul(e, NULL, 0);
		if ((e = getenv("DS_DUPSPAN"))) dupspan = (unsigned)strtoul(e, NULL, 0);
		if ((e = getenv("DS_FIXLEN"))) fixlen = (size_t)strtoul(e, NULL, 0);
		if (!phase_len) phase_len = 1;
		if (!dupspan || dupspan > 4096) DIE("DS_DUPSPAN must be in [1, 4096]");
		if (vmin <= sizeof(size_t) && !DS_AGG)
			ndb = 4;
	}
#if DS_AGG
	/* value-source HASHSUM: every value holds a hash slice; RESERVE is
	 * not compatible with value-source HASHSUM */
	if (vmin < MDB_HASH_SIZE)
		DIE("DS_AGG build: run with DS_VMIN >= %d (and DS_NO_RESERVE=1)", MDB_HASH_SIZE);
	no_reserve = 1;
#endif
	if (vmin < 1 || vmin > 256)
		DIE("DS_VMIN must be in [1, 256]");
	snapdir = getenv("DS_SNAPSHOT_DIR");
	if (!check_every) check_every = 1;

	snprintf(cmd, sizeof(cmd), "rm -rf '%s' && mkdir -p '%s'", dir, dir);
	if (system(cmd)) DIE("prepare DS_DIR");
	if (snapdir) {
		snprintf(cmd, sizeof(cmd), "rm -rf '%s' && mkdir -p '%s'", snapdir, snapdir);
		if (system(cmd)) DIE("prepare DS_SNAPSHOT_DIR");
	}
	C(mdb_env_create(&env));
	C(mdb_env_set_mapsize(env, (size_t)1 << 30));
	C(mdb_env_set_maxdbs(env, 8));
	C(mdb_env_open(env, dir, MDB_NOSYNC, 0664));
	maxkey = (size_t)mdb_env_get_maxkeysize(env);
	if (fixlen && (fixlen < 16 || fixlen > maxkey || fixlen < vmin))
		DIE("DS_FIXLEN must be in [max(16, DS_VMIN), %zu]", maxkey);
	fprintf(stderr, "DS-CONFIG ops=%lu keys=%u check_every=%u maxkey=%zu agg=%d "
		"hazard=%u vmin=%zu no_reserve=%u focus=%u\n", nops, nkeys, check_every, maxkey,
		DS_AGG, hazard, vmin, no_reserve, focus);
	begin_top();

	for (opno = 1; opno <= nops; opno++) {
		unsigned r;
		if (opno == stop_at) {
			/* debugging aid: persist the state reached before this op */
			close_cursors();
			while (depth > 0)
				C(mdb_txn_commit(txn_stack[--depth]));
			mdb_env_close(env);
			printf("STOP before op %lu\n", opno);
			return 0;
		}
		r = rnd(1000);
		if (focus) {
			/* grow phase: puts dominate; shrink phase: deletes */
			int shrink = (opno / phase_len) & 1;
			if (r >= 300 && r < 780)
				r = (r < 300 + (shrink ? 120 : 360)) ? 300 : 600;
		}
		if (focus && (opno % phase_len == 0 ||
			opno % phase_len == phase_len / 8 + 1)) {
			/* grow a dupset, shrink it a little later */
			op_bulk(opno % phase_len != 0);
			goto checks;
		}
		if (r < 300) op_cursor_get();
		else if (r < 600) op_put();
		else if (r < 780) op_del();
		else if (r < 880) op_cursor_put();
		else if (r < 950) op_cursor_del();
		else if (r < 965) op_multiple();
		else if (r < 985) op_rebind();
		else if (r < 999) op_txn();
		else if (rnd(4) == 0) op_drop();
		else op_cursor_get();
checks:
#ifdef DS_INTEGRITY
		if (agg_check >= 2) agg_integrity();
#endif
		if (opno % check_every == 0) {
			char why[32];
			snprintf(why, sizeof(why), "op%lu", opno);
			checkpoint(why);
		}
	}
	/* commit everything and verify the persisted state after a reopen */
	close_cursors();
	while (depth > 0)
		C(mdb_txn_commit(txn_stack[--depth]));
	snapshot("final");
	mdb_env_close(env);
	C(mdb_env_create(&env));
	C(mdb_env_set_maxdbs(env, 8));
	C(mdb_env_open(env, dir, MDB_NOSYNC, 0664));
	begin_top();
	checkpoint("reopen");
	close_cursors();
	mdb_txn_abort(txn());
	mdb_env_close(env);
	printf("END ops=%lu\n", nops);
	return 0;
}
