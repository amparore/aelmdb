/* Internal aggregate representation and arithmetic.
 *
 * This file is textually included by mdb.c after MDB_node and page-layout
 * internals are defined.  It is not a public header and deliberately relies
 * on LMDB private types/macros.
 */

/* Aggregate branch-node prefix layout.  Component bytes are packed in schema
 * order, then the entire prefix is padded to EVEN() so the following key starts
 * at the same 2-byte alignment used throughout LMDB.  Accesses to u64 fields
 * use memcpy and therefore do not assume stronger alignment of MDB_node. */
#define MDB_AGG_U64_SIZE ((size_t)sizeof(uint64_t))
/* Semantic aggregate bytes and physical prefix bytes are deliberately distinct:
 * the semantic blob is packed, while the branch-node prefix is EVEN-padded so
 * the following key keeps LMDB's 2-byte alignment invariant. */
#define MDB_AGG_MAXSZ (MDB_AGG_U64_SIZE + MDB_AGG_U64_SIZE + MDB_HASH_SIZE)
typedef struct MDB_aggblob {
	uint8_t v[MDB_AGG_MAXSZ];
} MDB_aggblob;

/* Normalized semantic aggregate.  Unlike MDB_aggblob this representation is
 * independent of the selected packed fields.  Disabled components are kept at
 * zero by decode/construction helpers and are ignored by arithmetic. */
typedef struct MDB_aggval {
	uint64_t entries;
	uint64_t keys;
	uint8_t hashsum[MDB_HASH_SIZE];
} MDB_aggval;

/* Net logical mutation.  Structural page rewrites do not create an additional
 * change: every ancestor observes A' = A - before + after. */
typedef struct MDB_aggchange {
	MDB_aggval before;
	MDB_aggval after;
} MDB_aggchange;

static inline void
mdb_aggval_zero(MDB_aggval *val)
{
	memset(val, 0, sizeof(*val));
}

static inline int
mdb_aggval_equal(uint16_t agg, const MDB_aggval *a, const MDB_aggval *b)
{
	agg &= MDB_AGG_MASK;
	if ((agg & MDB_AGG_ENTRIES) && a->entries != b->entries)
		return 0;
	if ((agg & MDB_AGG_KEYS) && a->keys != b->keys)
		return 0;
	if ((agg & MDB_AGG_HASHSUM) &&
		memcmp(a->hashsum, b->hashsum, MDB_HASH_SIZE) != 0)
		return 0;
	return 1;
}

/* MDB_aggblob stores only enabled semantic fields, in persistent schema order.
 * Physical EVEN() padding is deliberately not part of this conversion. */
static inline void
mdb_aggval_from_blob(uint16_t agg, const MDB_aggblob *blob, MDB_aggval *val)
{
	const uint8_t *p = blob->v;

	agg &= MDB_AGG_MASK;
	mdb_aggval_zero(val);
	if (agg & MDB_AGG_ENTRIES) {
		memcpy(&val->entries, p, MDB_AGG_U64_SIZE);
		p += MDB_AGG_U64_SIZE;
	}
	if (agg & MDB_AGG_KEYS) {
		memcpy(&val->keys, p, MDB_AGG_U64_SIZE);
		p += MDB_AGG_U64_SIZE;
	}
	if (agg & MDB_AGG_HASHSUM)
		memcpy(val->hashsum, p, MDB_HASH_SIZE);
}

static inline void
mdb_aggval_to_blob(uint16_t agg, const MDB_aggval *val, MDB_aggblob *blob)
{
	uint8_t *p = blob->v;

	agg &= MDB_AGG_MASK;
	memset(blob, 0, sizeof(*blob));
	if (agg & MDB_AGG_ENTRIES) {
		memcpy(p, &val->entries, MDB_AGG_U64_SIZE);
		p += MDB_AGG_U64_SIZE;
	}
	if (agg & MDB_AGG_KEYS) {
		memcpy(p, &val->keys, MDB_AGG_U64_SIZE);
		p += MDB_AGG_U64_SIZE;
	}
	if (agg & MDB_AGG_HASHSUM)
		memcpy(p, val->hashsum, MDB_HASH_SIZE);
}

/* Hash sums are little-endian integers modulo 2^(8*MDB_HASH_SIZE).
 *
 * Production builds use 64-bit limbs with carry/borrow; this is substantially
 * cheaper for the normal hash sizes (8, 16, 24, 32, ... bytes).  Validation
 * builds may define MDB_AGG_GENERIC_HASH_ARITH=1 to use byte-wise arithmetic,
 * allowing odd hash sizes that deliberately stress representation/alignment.
 * Both backends implement exactly the same modular little-endian algebra. */
#if MDB_AGG_GENERIC_HASH_ARITH
static inline void
mdb_agg_hash_add(uint8_t dst[MDB_HASH_SIZE], const uint8_t src[MDB_HASH_SIZE])
{
	size_t i;
	unsigned carry = 0;

	for (i = 0; i < MDB_HASH_SIZE; ++i) {
		unsigned sum = (unsigned)dst[i] + (unsigned)src[i] + carry;
		dst[i] = (uint8_t)sum;
		carry = sum >> 8;
	}
}

static inline void
mdb_agg_hash_sub(uint8_t dst[MDB_HASH_SIZE], const uint8_t src[MDB_HASH_SIZE])
{
	size_t i;
	unsigned borrow = 0;

	for (i = 0; i < MDB_HASH_SIZE; ++i) {
		unsigned lhs = (unsigned)dst[i];
		unsigned rhs = (unsigned)src[i] + borrow;
		dst[i] = (uint8_t)(lhs - rhs);
		borrow = lhs < rhs;
	}
}
#else
static inline uint64_t
mdb_agg_load_le64(const uint8_t *p)
{
	uint64_t x;
	memcpy(&x, p, sizeof(x));
#if BYTE_ORDER == LITTLE_ENDIAN
	return x;
#else
	return ((x & UINT64_C(0x00000000000000ff)) << 56) |
	       ((x & UINT64_C(0x000000000000ff00)) << 40) |
	       ((x & UINT64_C(0x0000000000ff0000)) << 24) |
	       ((x & UINT64_C(0x00000000ff000000)) << 8)  |
	       ((x & UINT64_C(0x000000ff00000000)) >> 8)  |
	       ((x & UINT64_C(0x0000ff0000000000)) >> 24) |
	       ((x & UINT64_C(0x00ff000000000000)) >> 40) |
	       ((x & UINT64_C(0xff00000000000000)) >> 56);
#endif
}

static inline void
mdb_agg_store_le64(uint8_t *p, uint64_t x)
{
#if BYTE_ORDER != LITTLE_ENDIAN
	x = ((x & UINT64_C(0x00000000000000ff)) << 56) |
	    ((x & UINT64_C(0x000000000000ff00)) << 40) |
	    ((x & UINT64_C(0x0000000000ff0000)) << 24) |
	    ((x & UINT64_C(0x00000000ff000000)) << 8)  |
	    ((x & UINT64_C(0x000000ff00000000)) >> 8)  |
	    ((x & UINT64_C(0x0000ff0000000000)) >> 24) |
	    ((x & UINT64_C(0x00ff000000000000)) >> 40) |
	    ((x & UINT64_C(0xff00000000000000)) >> 56);
#endif
	memcpy(p, &x, sizeof(x));
}

#if defined(__GNUC__) || defined(__clang__)
#define MDB_AGG_ADD3_U64(a,b,cin,out,cout) do { \
	uint64_t _t; int _c1 = __builtin_add_overflow((a),(b),&_t); \
	int _c2 = __builtin_add_overflow(_t,(uint64_t)(cin),&(out)); \
	(cout) = (uint64_t)(_c1 | _c2); \
} while (0)
#define MDB_AGG_SUB3_U64(a,b,bin,out,bout) do { \
	uint64_t _t; int _b1 = __builtin_sub_overflow((a),(b),&_t); \
	int _b2 = __builtin_sub_overflow(_t,(uint64_t)(bin),&(out)); \
	(bout) = (uint64_t)(_b1 | _b2); \
} while (0)
#elif defined(_MSC_VER) && (defined(_M_X64) || defined(_M_AMD64))
#include <intrin.h>
#define MDB_AGG_ADD3_U64(a,b,cin,out,cout) do { \
	unsigned char _c = _addcarry_u64((unsigned char)(cin),(a),(b),&(out)); \
	(cout) = _c; \
} while (0)
#define MDB_AGG_SUB3_U64(a,b,bin,out,bout) do { \
	unsigned char _b = _subborrow_u64((unsigned char)(bin),(a),(b),&(out)); \
	(bout) = _b; \
} while (0)
#else
#define MDB_AGG_ADD3_U64(a,b,cin,out,cout) do { \
	uint64_t _a=(a), _b=(b), _c=(cin), _s1=_a+_b; \
	uint64_t _c1=_s1<_a, _s=_s1+_c, _c2=_s<_s1; \
	(out)=_s; (cout)=_c1|_c2; \
} while (0)
#define MDB_AGG_SUB3_U64(a,b,bin,out,bout) do { \
	uint64_t _a=(a), _b=(b), _bi=(bin), _d1=_a-_b; \
	uint64_t _b1=_a<_b, _d=_d1-_bi, _b2=_d1<_bi; \
	(out)=_d; (bout)=_b1|_b2; \
} while (0)
#endif

static inline void
mdb_agg_hash_add(uint8_t dst[MDB_HASH_SIZE], const uint8_t src[MDB_HASH_SIZE])
{
	size_t i;
	uint64_t carry = 0;

	for (i = 0; i < MDB_HASH_SIZE; i += 8) {
		uint64_t a = mdb_agg_load_le64(dst + i);
		uint64_t b = mdb_agg_load_le64(src + i);
		uint64_t sum;
		MDB_AGG_ADD3_U64(a, b, carry, sum, carry);
		mdb_agg_store_le64(dst + i, sum);
	}
}

static inline void
mdb_agg_hash_sub(uint8_t dst[MDB_HASH_SIZE], const uint8_t src[MDB_HASH_SIZE])
{
	size_t i;
	uint64_t borrow = 0;

	for (i = 0; i < MDB_HASH_SIZE; i += 8) {
		uint64_t a = mdb_agg_load_le64(dst + i);
		uint64_t b = mdb_agg_load_le64(src + i);
		uint64_t diff;
		MDB_AGG_SUB3_U64(a, b, borrow, diff, borrow);
		mdb_agg_store_le64(dst + i, diff);
	}
}

#undef MDB_AGG_ADD3_U64
#undef MDB_AGG_SUB3_U64
#endif

/* Count overflow/underflow cannot occur for a valid maintainable tree.  Treat
 * it as an aggregate invariant failure and leave the destination unchanged. */
static inline int
mdb_aggval_add(uint16_t agg, MDB_aggval *dst, const MDB_aggval *src)
{
	MDB_aggval next = *dst;

	agg &= MDB_AGG_MASK;
	if ((agg & MDB_AGG_ENTRIES) && UINT64_MAX - next.entries < src->entries)
		return MDB_CORRUPTED;
	if ((agg & MDB_AGG_KEYS) && UINT64_MAX - next.keys < src->keys)
		return MDB_CORRUPTED;
	if (agg & MDB_AGG_ENTRIES)
		next.entries += src->entries;
	if (agg & MDB_AGG_KEYS)
		next.keys += src->keys;
	if (agg & MDB_AGG_HASHSUM)
		mdb_agg_hash_add(next.hashsum, src->hashsum);
	*dst = next;
	return MDB_SUCCESS;
}

static inline int
mdb_aggval_sub(uint16_t agg, MDB_aggval *dst, const MDB_aggval *src)
{
	MDB_aggval next = *dst;

	agg &= MDB_AGG_MASK;
	if ((agg & MDB_AGG_ENTRIES) && next.entries < src->entries)
		return MDB_CORRUPTED;
	if ((agg & MDB_AGG_KEYS) && next.keys < src->keys)
		return MDB_CORRUPTED;
	if (agg & MDB_AGG_ENTRIES)
		next.entries -= src->entries;
	if (agg & MDB_AGG_KEYS)
		next.keys -= src->keys;
	if (agg & MDB_AGG_HASHSUM)
		mdb_agg_hash_sub(next.hashsum, src->hashsum);
	*dst = next;
	return MDB_SUCCESS;
}

static inline int
mdb_aggval_apply_change(uint16_t agg, MDB_aggval *dst,
	const MDB_aggchange *change)
{
	MDB_aggval next = *dst;
	int rc;

	rc = mdb_aggval_sub(agg, &next, &change->before);
	if (rc)
		return rc;
	rc = mdb_aggval_add(agg, &next, &change->after);
	if (rc)
		return rc;
	*dst = next;
	return MDB_SUCCESS;
}

/* Extract the configured fixed-width hash source slice.  Negative offsets are
 * relative to the last complete MDB_HASH_SIZE-byte slice: -1 selects the last
 * slice, -2 starts one byte earlier, and so on. */
static inline int
mdb_agg_hash_slice(const MDB_val *source, int hash_offset,
	uint8_t out[MDB_HASH_SIZE])
{
	size_t start, back;

	if (!source || (source->mv_size && !source->mv_data))
		return EINVAL;
	if (hash_offset >= 0) {
		start = (size_t)hash_offset;
		if (start > source->mv_size ||
			MDB_HASH_SIZE > source->mv_size - start)
			return MDB_BAD_VALSIZE;
	} else {
		/* -(hash_offset + 1) is safe for the configured int16_t range and
		 * maps -1 -> 0, -2 -> 1, ... without signed overflow. */
		back = (size_t)(-(hash_offset + 1));
		if (source->mv_size < MDB_HASH_SIZE ||
			back > source->mv_size - MDB_HASH_SIZE)
			return MDB_BAD_VALSIZE;
		start = source->mv_size - MDB_HASH_SIZE - back;
	}
	memcpy(out, (const uint8_t *)source->mv_data + start, MDB_HASH_SIZE);
	return MDB_SUCCESS;
}

static inline size_t
mdb_agg_value_size_flags(uint16_t agg)
{
	size_t n = 0;
	if (agg & MDB_AGG_ENTRIES) n += MDB_AGG_U64_SIZE;
	if (agg & MDB_AGG_KEYS) n += MDB_AGG_U64_SIZE;
	if (agg & MDB_AGG_HASHSUM) n += MDB_HASH_SIZE;
	return n;
}

static inline size_t
mdb_agg_prefix_len_flags(uint16_t agg)
{
	return EVEN(mdb_agg_value_size_flags(agg));
}

static inline size_t
mdb_agg_prefix_bytes(const MDB_page *mp)
{
	return (mp && IS_BRANCH(mp)) ?
		mdb_agg_prefix_len_flags(PAGE_AGGFLAGS(mp)) : 0;
}

static inline void
mdb_node_get_agg_blob(const MDB_page *mp, const MDB_node *node, MDB_aggblob *out)
{
	uint16_t agg = PAGE_AGGFLAGS(mp);
	size_t raw = mdb_agg_value_size_flags(agg);
	if (raw) memcpy(out->v, node->mn_data, raw);
}

static inline void
mdb_node_set_agg_blob(MDB_page *mp, MDB_node *node, const MDB_aggblob *in)
{
	uint16_t agg = PAGE_AGGFLAGS(mp);
	size_t raw = mdb_agg_value_size_flags(agg);
	size_t pfx = mdb_agg_prefix_len_flags(agg);
	if (raw) memcpy(node->mn_data, in->v, raw);
	if (pfx > raw) memset(node->mn_data + raw, 0, pfx - raw);
}


static inline void
mdb_node_get_aggval(const MDB_page *mp, const MDB_node *node, MDB_aggval *out)
{
	MDB_aggblob blob;
	uint16_t agg = PAGE_AGGFLAGS(mp);

	memset(&blob, 0, sizeof(blob));
	mdb_node_get_agg_blob(mp, node, &blob);
	mdb_aggval_from_blob(agg, &blob, out);
}

static inline void
mdb_node_set_aggval(MDB_page *mp, MDB_node *node, const MDB_aggval *in)
{
	MDB_aggblob blob;
	uint16_t agg = PAGE_AGGFLAGS(mp);

	mdb_aggval_to_blob(agg, in, &blob);
	mdb_node_set_agg_blob(mp, node, &blob);
}

static inline void *
mdb_nodekey(const MDB_page *mp, const MDB_node *node)
{
	return (void *)((char *)node->mn_data + mdb_agg_prefix_bytes(mp));
}

