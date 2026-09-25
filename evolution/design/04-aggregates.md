# 04 - Aggregate maintenance and queries

## Objective

Turn the stage-03 representation into maintained subtree metadata and expose
queries over it, using the stage-02 bottom-up mutation contract rather than
the capture/replay/repair architecture of first-generation AELMDB.

## Algebra

An item has an aggregate contribution (`entries`, optional `keys`, optional
modular `hashsum`).  A logical replacement is represented by its **before/after
contribution**, rather than by a sequence of physical mutations.  This is
particularly important for DUPSORT: one primary item may represent an entire
duplicate set, so before/after captures the logical item semantics independently
of how many structural operations happened below it.  Hash arithmetic is modulo
2^(8*MDB_HASH_SIZE).

The production writer uses two operations only:

1. **exact local fold** of one final page when structure changed at that level;
2. **logical delta** on an ancestor whose child structure did not change.

No production maintenance path recursively folds a subtree.

## Structural publication

When the bottom-up engine publishes a split, move, merge or new duplicate
subtree, aggregate bytes for the final child state are available before the
parent link is written.  Thus branch metadata is information carried by the
mutation, not information rediscovered after it.

## Logical settlement

The current implementation deliberately separates structural unwind from the
final logical settlement.  After the mutation has produced its final cursor
path, `mdb_agg_settle()` uses `mt_unwind_prefix` as follows:

- structurally changed levels below the boundary are recomputed by exact local
  page folds;
- the untouched ancestor prefix receives the logical before/after delta;
- database totals receive the same logical delta.

This is O(tree height), uses the final path, and performs no pre-search or
repair.  The separation is intentional: the generic mutation engine carries
structural information, while aggregate-specific logical settlement is a final
phase over the path that the unwind has made definitive.  Keeping the delta out
of `MDB_level` avoids coupling the stage-02 mutation engine to aggregate
semantics.

The debug unwind oracle verifies the key assumption directly: before settlement,
every path element above the recorded boundary is byte-for-byte unchanged.
Primary and DUPSORT duplicate trees have separate snapshots/boundaries.  The
snapshots are thread-local so independent environments can run concurrent
writers in diagnostic builds.  Full LMDB pages snapshot `me_psize` bytes;
inline duplicate sub-pages snapshot their actual embedded extent, never bytes
beyond the containing node.

## Error semantics

Aggregate maintenance follows the mutation engine's transaction-error rule. A
failure before mutation may be returned without poisoning the write transaction.
Once dirty/allocative state has been acquired, or a structural/aggregate update
has been partially applied, an error marks the transaction `MDB_TXN_ERROR`; the
partial state can therefore only be aborted, never committed. This includes the
inline-DUPSORT to persistent-subDB transition: after the new root page has been
allocated, failure while folding its initial aggregate totals invalidates the
transaction even though the primary node has not yet been published.

## DUPSORT

Nested duplicate-tree mutation completes first.  Persistent sub-DB totals are
settled before the primary item is settled.  Inline duplicate sub-pages do not
own persistent aggregate totals; their contribution is folded locally when
needed.  The read path is not modified to maintain metadata.

## Hash-sum arithmetic

The production default uses 64-bit little-endian limbs with carry/borrow and
requires `MDB_HASH_SIZE` to be a multiple of 8 bytes.  This is the normal and
performance-oriented configuration.

For validation builds, `MDB_AGG_GENERIC_HASH_ARITH=1` selects a byte-wise
backend implementing the same modular algebra.  It deliberately permits odd or
unusual hash sizes (for example 7, 13, or 31 bytes), which stress aggregate
layout and alignment without weakening the optimized production path.

## Source modularity

Aggregate responsibilities are kept out of the monolithic LMDB core whenever
they can be separated without hiding structural mutation:

- `mdb_agg_internal.h`: private persistent representation, normalized values,
  hash arithmetic, blob conversion and branch-prefix accessors;
- `mdb_agg_maint.c`: record/dupset contribution semantics, exact local page
  fold, before/after settlement, and exact branch-link publication helpers;
- `mdb_agg_query.c`: production aggregate, prefix/range, rank/select, cursor-rank
  and window-relative rank-range queries;
- `mdb_agg_debug.c`: recursive integrity oracle, compiled for diagnostics and
  tests only;
- `mdb.c`: the bottom-up mutation engine and the explicit integration points
  where page structure changes or aggregate maintenance is invoked.

The textual inclusion of these focused modules is intentional: they use LMDB
private internals while avoiding a large artificial private API.  Structural
operations themselves remain visible in `mdb.c`; the split is by responsibility,
not an attempt to hide the mutation engine behind callbacks.  All four aggregate
files are part of the final design.  The oracle detects inconsistency; it never
repairs production state.

## Feature parity

The optimized multi-limb hash arithmetic and
`mdb_agg_window_aggregate()` / `mdb_agg_window_rank()` have been recovered from
first-generation AELMDB at the stage-04 boundary.  Their integration depends
only on the current aggregate algebra/query contracts; no first-generation
maintenance machinery is reintroduced.

## Completion criteria

Stage 04 is complete when:

- structural and logical aggregate invariants pass the independent oracle;
- fixed query signatures remain stable;
- differential LMDB behavior is preserved where intentionally comparable;
- large-key and DUPSORT stress exercise deep split/merge/rebalance paths;
- optimized hashsum and window/rank-range functionality pass feature-parity
  tests;
- generic odd-size hash builds pass alignment/layout stress;
- `mdb_agg_settle()` remains the documented final logical-settlement boundary.

### Query fast paths

The query module is read-only but is not required to materialize the full semantic
aggregate when a query needs only ENTRIES or KEYS.  Rank/select therefore read the
selected 64-bit branch weight directly from the packed prefix and use leaf
multiplicity-only helpers.  Full prefix/range queries accumulate packed branch
aggregates directly into a local accumulator.  Plain-tree window aggregation uses
a one-pass rank-space prefix traversal; DUPSORT retains the generic boundary
fallback when a rank cuts inside a duplicate set.  These are query-only
optimizations and do not alter persistent format or maintenance invariants.
