/* agg_test_compat.h - test-local helpers not exposed by the AELMDB API.
 *
 * The suite uses the current AELMDB public API directly.  This header retains
 * only convenience helpers for inspecting hash slices in reference checks.
 *
 * Semantics are identical to AELMDB's helpers: hash sums are little-endian
 * integers modulo 2^(8*MDB_HASH_SIZE); a non-negative offset selects the
 * slice starting at that byte, a negative offset counts from the end
 * (-1 = last MDB_HASH_SIZE bytes, -2 = one byte earlier, ...).  The helpers
 * are written byte-wise so that any MDB_HASH_SIZE (also odd) is supported.
 */
#ifndef AGG_TEST_COMPAT_H
#define AGG_TEST_COMPAT_H

#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include "lmdb.h"

/* Current AELMDB public API. */
#define AGG_TEST_HAVE_WINDOW 1
/* Value-source HASHSUM is supported with MDB_WRITEMAP; only MDB_RESERVE is
 * incompatible because the final bytes are not available at put time. */
#define AGG_TEST_WRITEMAP_VALUE_HASHSUM 1

static inline int
mdb_hashslice_ptr_bytes(const void *p, size_t sz, int16_t off,
	const uint8_t **slice)
{
	ptrdiff_t start;
	if (!p || !slice || sz < (size_t)MDB_HASH_SIZE)
		return MDB_BAD_VALSIZE;
	start = off >= 0 ? (ptrdiff_t)off
		: (ptrdiff_t)sz - (ptrdiff_t)MDB_HASH_SIZE + 1 + (ptrdiff_t)off;
	if (start < 0 || (size_t)start + (size_t)MDB_HASH_SIZE > sz)
		return MDB_BAD_VALSIZE;
	*slice = (const uint8_t *)p + start;
	return MDB_SUCCESS;
}

static inline int
mdb_hashslice_copy_bytes(const void *p, size_t sz, int16_t off,
	uint8_t out[MDB_HASH_SIZE])
{
	const uint8_t *slice;
	int rc = mdb_hashslice_ptr_bytes(p, sz, off, &slice);
	if (rc)
		return rc;
	memcpy(out, slice, MDB_HASH_SIZE);
	return MDB_SUCCESS;
}

#endif /* AGG_TEST_COMPAT_H */
