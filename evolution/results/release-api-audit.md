# AELMDB 0.2.0 release/API audit

This audit validates the public AELMDB 0.2.0 API against the principal consumer
stack and records the final pre-release compatibility decisions.

## Version identity

AELMDB preserves the inherited LMDB `MDB_VERSION_*` lineage macros and exposes
an independent project version:

- `MDB_AELMDB_VERSION_MAJOR` = 0
- `MDB_AELMDB_VERSION_MINOR` = 2
- `MDB_AELMDB_VERSION_PATCH` = 0
- `MDB_AELMDB_VERSION`
- `MDB_AELMDB_VERSION_STRING` = `"AELMDB 0.2.0"`

`MDB_AGGFORMAT_VERSION` remains a separate persistent-format identifier.

## Public aggregate API

The 0.2.0 public surface is retained without further renaming:

- aggregate schema: `MDB_AGG_*` flags;
- signed per-DBI hash offset: `mdb_set_hash_offset()` / `mdb_get_hash_offset()`;
- hashsum helpers: `mdb_hashsum_*`;
- totals/prefix/range: `mdb_agg_totals()`, `mdb_agg_prefix()`, `mdb_agg_range()`;
- order statistics: `mdb_agg_rank()`, `mdb_agg_select()`,
  `mdb_agg_cursor_seek_rank()`;
- reconciliation windows: `MDB_agg_window`, `mdb_agg_window_aggregate()`,
  `mdb_agg_window_rank()`;
- debug-only integrity oracle: `mdb_agg_check_integrity()` when
  `MDB_DEBUG_AGG_INTEGRITY` is enabled.

No legacy `MDB_AGG_CHECK`, `mdb_dbg_check_agg_db()`, `AGG_V2`, or `aggmaint`
compatibility names remain in the product or current test suite.

## Consumer validation

### lmdbxx-aelmdb

Validated against AELMDB 0.2.0 with the dedicated `check-aelmdb` target.
Coverage includes project-version detection, signed negative hash offsets,
aggregate totals, rank/select, and window queries.  The test is built with
ASan/UBSan by the consumer project.

### negentropy-aelmdb

`SliceAELMDB` requires `MDB_AELMDB_VERSION >= MDB_VERINT(0,2,0)` and compiles
through the current lmdb++ wrapper.  Its required schema remains
`MDB_AGG_ENTRIES | MDB_AGG_HASHSUM | MDB_AGG_HASHSOURCE_FROM_KEY` with
`MDB_HASH_SIZE == negentropy::ID_SIZE`.

### bench-aelmdb

The AELMDB slice reconciliation tests pass for AELMDB-to-AELMDB and
AELMDB-to-BTreeLMDB.  The advanced benchmark builds against the new repository
layout and completes reduced release-audit scenarios with `have/need` equal to
the expected reconciliation result.

## Empty-value hardening

The inherited LMDB node insertion path now skips `memcpy` when `mv_size == 0`,
so an ordinary empty value with `mv_data == NULL` does not rely on zero-length
`memcpy` accepting a null pointer.  This is classified as `01-base` hardening,
not an aggregate change.  `tests/regress/empty_value.c` verifies normal LMDB
semantics and is also executed by `make test-sanitize`.
