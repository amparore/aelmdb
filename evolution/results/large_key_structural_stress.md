# 06d — large-key structural stress

## Goal

This extension of the shared AELMDB/aggmaint harness stresses the B+tree shape
rather than only increasing the number of random operations.  Large primary
keys reduce both leaf capacity and branch fanout because separator keys retain
the large key payload as they propagate upward.  The result is a deeper tree
with substantially more split/rebalance/merge activity for a comparatively
small number of logical records.

The test remains in `tests/compare/agg_compare_shared.c` and is executed unchanged
against both implementations by `tests/compare/run_agg_compare.sh`.

## Profiles

The physical key is generated deterministically from a sortable 64-bit logical
ID followed by deterministic padding.  Therefore the logical order is stable,
and every delete/reinsert operation reconstructs exactly the same key bytes.

Two profiles are defined from `mdb_env_get_maxkeysize(env)`:

- `large-key-near-max`: fixed key length `near_max`, currently `maxkey - 16`
  when the configured maximum is at least 64 bytes;
- `large-key-half-to-near-max`: deterministic per-key length in
  `[ceil(maxkey/2), near_max]`.

The first profile minimizes fanout and maximizes vertical propagation.  The
second keeps fanout low while deliberately making page packing and branch arity
irregular.

Values remain small (`MDB_HASH_SIZE + 16`) so the principal pressure comes from
the large primary keys rather than overflow values.  The test uses both a plain
DB and a DUPSORT DB.  The DUPSORT DB has one small duplicate value per primary
key, so an exact `mdb_del(key,data)` removes the outer-tree key and directly
exercises the DUPSORT maintenance boundary without introducing large duplicate
subtrees.

## Structural workload

For the default `LARGE_COUNT=768`, the full state contains 1536 primary keys per
DB.

1. **Sparse ordered grow** — insert only even logical IDs.  This creates the
   initial tall tree while leaving one deterministic gap between every pair.
2. **Interior fill** — insert all odd IDs.  These are interior insertions, not a
   right-edge append workload.
3. **Plain delete/regrow waves** — for each round, remove alternating contiguous
   blocks of keys, verify, then rebuild the removed blocks in reverse order.
   The block width changes by round.
4. **DUPSORT delete/regrow waves** — repeat the same structural pattern using
   exact duplicate deletes.
5. **Environment reopen** — verify the final aggregate/query state after a full
   close/reopen.

Plain and DUPSORT churn are intentionally separated into distinct transactions.
This lets the complete plain maintenance workload execute before a
DUPSORT-specific failure can terminate a run.

Every verification phase runs the existing totals/prefix/range/rank/select and
cursor-seek oracle.  Deletes also check their immediate ordinary-LMDB
postcondition before the aggregate oracle is consulted.

## Structural telemetry

Each phase records `mdb_stat()` data for both DBs:

- tree depth;
- branch page count;
- leaf page count;
- overflow page count;
- record count.

The harness tracks peak depth/page counts and depth changes and requires a
minimum depth (`LARGE_MIN_DEPTH`, default 4).  Physical page statistics are
reported but are intentionally not folded into the cross-implementation logical
signature.

On the current 4 KiB-page test environment with `MDB_HASH_SIZE=32`,
`mdb_env_get_maxkeysize()` is 511, so the profiles use:

```text
large-key-near-max:          495 bytes fixed
large-key-half-to-near-max:  256..495 bytes variable
```

With only 1536 full-state primary keys, the fixed profile reaches depth 5 and
the variable profile reaches depth 4.  In the fixed profile the first 50%-class
plain delete wave reduces leaf pages from 256 to 178; reverse regrowth produces
384 leaf pages, demonstrating that the workload is exercising materially
different page layouts rather than merely scanning a tall static tree.

## Current comparison result

Command:

```sh
make compare-large-keys
```

Defaults:

```text
LARGE_HASHES=32
LARGE_COUNT=768
LARGE_ROUNDS=4
LARGE_MIN_DEPTH=4
```

The current result is stored in `results/compare_large_keys.tsv`.

### AELMDB, fixed near-max

The complete four-round workload passes.  The tree reaches depth 5.

### AELMDB, half-to-near-max

The first variable-length delete/regrow round completes, but the second plain
round fails on logical key 1371:

```text
MDB_CORRUPTED: Located page was wrong type
```

This occurs before DUPSORT churn.  aggmaint completes all four corresponding
plain rounds, so the failure is specific to the AELMDB line under irregular
large-key packing.  It is distinct from the existing C3 exact-DUPSORT issue.
The root cause is not yet assigned; the workload should be preserved unchanged
for later localization.

### aggmaint, both profiles

Before C4, aggmaint completed all four plain delete/regrow rounds for both
profiles and then crashed on entry into the first DUPSORT exact-delete wave.
AddressSanitizer localized the crash to:

```text
mdb_node_set_agg_blob
  -> mdb_node_set_aggval
  -> mdb_agg_publish_page_up
  -> mdb_agg_dupsort_finish
  -> _mdb_cursor_del
  -> mdb_del
```

**Root cause (C4, closed).** A structural DUPSORT delete made
`mdb_agg_dupsort_finish()` re-fold the primary path along the delete cursor's
stack.  By that point `mdb_cursor_del0()` had already relocated the cursor to
the next sibling leaf, which can sit in a clean subtree, so the write hit
read-only map pages.  M5 had already published exact aggregates on the mutated
path, so the re-fold was also unnecessary.  The unconditional re-fold dates
from M6.  C4 keeps it only for puts and adds a dirty-parent guard to
`mdb_agg_publish_page_up()`.  See `patches/lineage/04-aggmaint/10-C4-dupsort-structural-delete.patch` and the C4
section of `lineage/04-aggmaint/README.md`.

After C4, aggmaint passes both profiles.  For the fixed near-max profile, its
signatures equal AELMDB's at H = 8, 32, 64 and 256.  The same runs pass with
the recursive integrity oracle called after every large-key mutation, under
ASan and UBSan.

### Additional AELMDB observation: other hash widths

`make compare-large-keys LARGE_HASHES="8 64 256"` (post-C4) gives:

```text
large-key-near-max          H=8,64,256   AELMDB PASS, signatures equal aggmaint
large-key-half-to-near-max  H=8          AELMDB PASS, signature equals aggmaint
large-key-half-to-near-max  H=64         AELMDB FAIL: plain-delete-0 "select entries value"
large-key-half-to-near-max  H=256        AELMDB FAIL: plain-delete-0 aggregate prefix mismatch
                                          (e=558/560, k=558/560)
```

The variable-length AELMDB failure is therefore not specific to H=32.  At H=64
and H=256 it shows up earlier, in the first plain delete wave, and as a silent
aggregate miscount caught by the query oracle rather than as
`MDB_CORRUPTED`.  Its root cause is still not assigned.

## Regression check

The harness extension does not change the established short comparison result.
A fresh H=32 regression run is stored in `results/short_regression_06d.tsv`.
With `MDB_HASH_SIZE=32`, the existing modes still pass with identical
AELMDB/aggmaint signatures:

```text
query   0d47be1ce619f8bb
growth  540f6ba1c866836f
stress  5858bc4640074533
```

The frozen AELMDB long baseline also continues to expose C3 through the
independent DUPSORT pre-state oracle; this is a separate known finding and is
not weakened or masked by 06d.

## Targets

```sh
make compare-large-keys
make compare-large-near-max
make compare-large-half-to-near-max
make compare-large-smoke
```

The large-key modes are diagnostic stress targets.  They stay out of `make
check` while the AELMDB variable-length failure is open.
