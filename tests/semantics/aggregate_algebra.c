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

static void
fill_hash(uint8_t h[MDB_HASH_SIZE], unsigned seed)
{
	size_t i;
	for (i = 0; i < MDB_HASH_SIZE; ++i)
		h[i] = (uint8_t)(seed + 37u * (unsigned)i);
}

static int
test_blob_roundtrip(void)
{
	static const uint16_t schemas[] = {
		0,
		MDB_AGG_ENTRIES,
		MDB_AGG_KEYS,
		MDB_AGG_HASHSUM,
		MDB_AGG_ENTRIES|MDB_AGG_KEYS,
		MDB_AGG_ENTRIES|MDB_AGG_HASHSUM,
		MDB_AGG_KEYS|MDB_AGG_HASHSUM,
		MDB_AGG_ENTRIES|MDB_AGG_KEYS|MDB_AGG_HASHSUM
	};
	MDB_aggval in, out;
	MDB_aggblob blob;
	size_t i, j;

	mdb_aggval_zero(&in);
	in.entries = UINT64_C(0x1122334455667788);
	in.keys = UINT64_C(0x8877665544332211);
	fill_hash(in.hashsum, 11);

	for (i = 0; i < sizeof(schemas)/sizeof(schemas[0]); ++i) {
		memset(&blob, 0xa5, sizeof(blob));
		mdb_aggval_to_blob(schemas[i], &in, &blob);
		mdb_aggval_from_blob(schemas[i], &blob, &out);
		CHECK(out.entries == ((schemas[i] & MDB_AGG_ENTRIES) ? in.entries : 0));
		CHECK(out.keys == ((schemas[i] & MDB_AGG_KEYS) ? in.keys : 0));
		if (schemas[i] & MDB_AGG_HASHSUM)
			CHECK(memcmp(out.hashsum, in.hashsum, MDB_HASH_SIZE) == 0);
		else {
			for (j = 0; j < MDB_HASH_SIZE; ++j)
				CHECK(out.hashsum[j] == 0);
		}
		for (j = mdb_agg_value_size_flags(schemas[i]); j < sizeof(blob.v); ++j)
			CHECK(blob.v[j] == 0);
	}
	return 0;
}

static int
test_hash_arithmetic(void)
{
	uint8_t a[MDB_HASH_SIZE], b[MDB_HASH_SIZE], saved[MDB_HASH_SIZE];
	size_t i;

	memset(a, 0xff, sizeof(a));
	memset(b, 0, sizeof(b));
	b[0] = 1;
	mdb_agg_hash_add(a, b);
	for (i = 0; i < MDB_HASH_SIZE; ++i)
		CHECK(a[i] == 0);

	mdb_agg_hash_sub(a, b);
	for (i = 0; i < MDB_HASH_SIZE; ++i)
		CHECK(a[i] == 0xff);

	fill_hash(a, 3);
	fill_hash(b, 201);
	memcpy(saved, a, sizeof(saved));
	mdb_agg_hash_add(a, b);
	mdb_agg_hash_sub(a, b);
	CHECK(memcmp(a, saved, sizeof(a)) == 0);
	return 0;
}

static int
test_count_arithmetic(void)
{
	const uint16_t agg = MDB_AGG_ENTRIES|MDB_AGG_KEYS|MDB_AGG_HASHSUM;
	MDB_aggval a, b, before;
	int rc;

	mdb_aggval_zero(&a);
	mdb_aggval_zero(&b);
	a.entries = 10;
	a.keys = 7;
	b.entries = 3;
	b.keys = 2;
	fill_hash(a.hashsum, 9);
	fill_hash(b.hashsum, 44);
	before = a;

	CHECK(mdb_aggval_add(agg, &a, &b) == MDB_SUCCESS);
	CHECK(a.entries == 13 && a.keys == 9);
	CHECK(mdb_aggval_sub(agg, &a, &b) == MDB_SUCCESS);
	CHECK(memcmp(&a, &before, sizeof(a)) == 0);

	a = before;
	a.entries = UINT64_MAX;
	before = a;
	b.entries = 1;
	rc = mdb_aggval_add(MDB_AGG_ENTRIES, &a, &b);
	CHECK(rc == MDB_CORRUPTED);
	CHECK(memcmp(&a, &before, sizeof(a)) == 0);

	mdb_aggval_zero(&a);
	mdb_aggval_zero(&b);
	a.keys = 1;
	b.keys = 2;
	before = a;
	rc = mdb_aggval_sub(MDB_AGG_KEYS, &a, &b);
	CHECK(rc == MDB_CORRUPTED);
	CHECK(memcmp(&a, &before, sizeof(a)) == 0);
	return 0;
}

static int
test_change(void)
{
	const uint16_t agg = MDB_AGG_ENTRIES|MDB_AGG_KEYS|MDB_AGG_HASHSUM;
	MDB_aggval base, expected;
	MDB_aggchange ch;

	mdb_aggval_zero(&base);
	mdb_aggval_zero(&ch.before);
	mdb_aggval_zero(&ch.after);
	base.entries = 20;
	base.keys = 8;
	fill_hash(base.hashsum, 7);
	ch.before.entries = 4;
	ch.before.keys = 1;
	fill_hash(ch.before.hashsum, 21);
	ch.after.entries = 6;
	ch.after.keys = 1;
	fill_hash(ch.after.hashsum, 73);

	expected = base;
	CHECK(mdb_aggval_sub(agg, &expected, &ch.before) == MDB_SUCCESS);
	CHECK(mdb_aggval_add(agg, &expected, &ch.after) == MDB_SUCCESS);
	CHECK(mdb_aggval_apply_change(agg, &base, &ch) == MDB_SUCCESS);
	CHECK(memcmp(&base, &expected, sizeof(base)) == 0);
	return 0;
}

static int
test_change_atomicity(void)
{
	MDB_aggval base, before;
	MDB_aggchange ch;

	mdb_aggval_zero(&base);
	mdb_aggval_zero(&ch.before);
	mdb_aggval_zero(&ch.after);
	base.entries = 4;
	ch.before.entries = 5;
	before = base;
	CHECK(mdb_aggval_apply_change(MDB_AGG_ENTRIES, &base, &ch) == MDB_CORRUPTED);
	CHECK(memcmp(&base, &before, sizeof(base)) == 0);

	mdb_aggval_zero(&base);
	mdb_aggval_zero(&ch.before);
	mdb_aggval_zero(&ch.after);
	base.entries = UINT64_MAX;
	ch.after.entries = 1;
	before = base;
	CHECK(mdb_aggval_apply_change(MDB_AGG_ENTRIES, &base, &ch) == MDB_CORRUPTED);
	CHECK(memcmp(&base, &before, sizeof(base)) == 0);
	return 0;
}

static int
test_hash_slice(void)
{
	uint8_t source_bytes[MDB_HASH_SIZE + 8];
	uint8_t out[MDB_HASH_SIZE];
	MDB_val source;
	size_t i;

	for (i = 0; i < sizeof(source_bytes); ++i)
		source_bytes[i] = (uint8_t)i;
	source.mv_data = source_bytes;
	source.mv_size = sizeof(source_bytes);

	CHECK(mdb_agg_hash_slice(&source, 3, out) == MDB_SUCCESS);
	CHECK(memcmp(out, source_bytes + 3, MDB_HASH_SIZE) == 0);
	CHECK(mdb_agg_hash_slice(&source, -1, out) == MDB_SUCCESS);
	CHECK(memcmp(out, source_bytes + 8, MDB_HASH_SIZE) == 0);
	CHECK(mdb_agg_hash_slice(&source, -2, out) == MDB_SUCCESS);
	CHECK(memcmp(out, source_bytes + 7, MDB_HASH_SIZE) == 0);
	CHECK(mdb_agg_hash_slice(&source, 9, out) == MDB_BAD_VALSIZE);
	CHECK(mdb_agg_hash_slice(&source, -10, out) == MDB_BAD_VALSIZE);

	source.mv_size = MDB_HASH_SIZE - 1;
	CHECK(mdb_agg_hash_slice(&source, 0, out) == MDB_BAD_VALSIZE);
	CHECK(mdb_agg_hash_slice(&source, -1, out) == MDB_BAD_VALSIZE);
	return 0;
}

int
main(void)
{
	if (test_blob_roundtrip()) return 1;
	if (test_hash_arithmetic()) return 1;
	if (test_count_arithmetic()) return 1;
	if (test_change()) return 1;
	if (test_change_atomicity()) return 1;
	if (test_hash_slice()) return 1;
	printf("aggregate semantic algebra tests passed (MDB_HASH_SIZE=%d)\n", MDB_HASH_SIZE);
	return 0;
}
