# Performance audit — consolidated AELMDB

## Scope and methodology

This audit measures the consolidated product after feature parity and error-path
hardening.  The goal is to separate four costs:

1. structural drift from LMDB (`01`/`02`/`03`/current with aggregates disabled);
2. aggregate maintenance cost;
3. hash arithmetic/width cost;
4. aggregate-query cost relative to first-generation AELMDB.

Measurements were collected on an x86-64 AMD EPYC 9V74 container with GCC 14.2,
`-O3 -DNDEBUG`, one pinned CPU, `MDB_NOSYNC`, and repeated runs.  Timings are
relative engineering measurements, not portable absolute performance claims.
Short LMDB microbenchmarks are scheduler-sensitive, so medians and broad ratios
are more meaningful than sub-5% differences.

## 1. Structural drift without aggregates

A tmpfs run with 500k random plain puts gives the following representative
median times (seconds):

| Variant | put | get | scan | delete | put vs LMDB |
|---|---:|---:|---:|---:|---:|
| LMDB 0.9.70 | 0.701 | 0.331 | 0.012 | 0.555 | 1.00x |
| 01-base | 0.728 | 0.343 | 0.013 | 0.536 | 1.04x |
| 02-bottomup | 0.711 | 0.327 | 0.013 | 0.555 | 1.01x |
| 03-aggformat | 0.724 | 0.336 | 0.013 | 0.529 | 1.03x |
| current AELMDB, aggregates off | 0.733 | 0.360 | 0.013 | 0.536 | 1.05x |

The key result is that the bottom-up mutation engine is not responsible for a
large performance regression.  Its measured overhead is within the noise band
of a few percent on this workload.

## 2. Aggregate maintenance cost

On the same plain workload, full ENTRIES+KEYS+HASHSUM maintenance at H=32 gives
approximately 0.841 s put versus 0.701 s baseline: about **1.20x**.  Delete is
about **1.16x** baseline.

A component decomposition at H=32 shows that ENTRIES/KEYS alone are relatively
cheap; HASHSUM is the dominant incremental cost.  On DUPSORT the same pattern is
stronger: count-only maintenance is around 1.4x the current no-aggregate path,
while HASHSUM/full maintenance is around 1.6x in the repeated microbenchmark.

## 3. DUPSORT cardinality scaling

With 30k primary keys and D duplicate values per key, median put time per logical
insert is approximately:

| D | no aggregates (us/insert) | full aggregates (us/insert) |
|---:|---:|---:|
| 1 | 0.40 | 0.57 |
| 2 | 0.50 | 0.77 |
| 4 | 0.53 | 0.88 |
| 8 | 0.49 | 0.78 |
| 16 | 0.53 | 0.88 |
| 32 | 0.52 | 1.12 |

This does **not** show cost proportional to dupset cardinality.  The step at
D=32 is compatible with representation/tree-height changes; there is no sign of
an O(number-of-duplicates) full-dupset fold on every put.

## 4. Optimized limb arithmetic

At H=32, the default 64-bit-limb backend is materially faster than the generic
byte backend while producing identical semantics:

| Workload | optimized put | generic put | speedup | optimized delete | generic delete |
|---|---:|---:|---:|---:|---:|
| plain | 0.086 | 0.108 | 1.26x | 0.065 | 0.084 |
| DUPSORT x8 | 0.215 | 0.292 | 1.36x | 0.043 | 0.060 |

Keeping the generic backend only for odd-width/alignment validation is therefore
the correct default policy.

## 5. Hash-width scaling

Using a fixed 256-byte value payload so that H=8/32/64/256 all hash valid data,
H=8..64 remains in the same broad range.  H=256 is substantially more expensive,
particularly for DUPSORT:

| H | plain put | DUPSORT x8 put | plain delete | DUPSORT delete |
|---:|---:|---:|---:|---:|
| 8 | 0.155 | 0.190 | 0.126 | 0.035 |
| 32 | 0.202 | 0.254 | 0.132 | 0.032 |
| 64 | 0.192 | 0.285 | 0.147 | 0.060 |
| 256 | 0.326 | 0.764 | 0.221 | 0.113 |

The H=256 cost is expected to combine wider arithmetic with much larger branch
prefixes and therefore lower branch fanout.  It should be treated as a stress
configuration rather than the normal production point.

## 6. Query-layer optimization

The initial audit found the query layer to be the remaining performance
regression.  The cause was confined to `mdb_agg_query.c`: rank/select decoded
complete aggregate values when only one 64-bit weight was needed, and window
aggregation implemented a rank prefix as `select(rank)` followed by a second
key-prefix search.

The optimization pass keeps the query layer read-only and introduces three
query-specific fast paths:

- direct branch weight reads for ENTRIES/KEYS without materializing HASHSUM;
- leaf weight reads that use constant weight 1 where possible and inspect only
  DUPSORT multiplicity when required;
- direct rank-space prefix traversal for plain trees, accumulating complete
  stored subtree aggregates in one descent instead of select + second search.

Full aggregate prefix/range traversal was also tightened to accumulate the packed
branch prefix directly into the query accumulator, avoiding temporary normalized
`MDB_aggval` copies.

On the same 200k-record H=32 plain aggregate DB with 100k random queries, the
post-optimization median times are approximately:

| Query | current | first-generation | current / legacy |
|---|---:|---:|---:|
| totals | 0.00165 | 0.00209 | 0.79x |
| prefix | 0.142 | 0.161 | 0.88x |
| range | 0.282 | 0.306 | 0.92x |
| rank | 0.0523 | 0.0764 | 0.68x |
| select | 0.0356 | 0.0488 | 0.73x |
| cursor seek-rank | 0.0347 | 0.0462 | 0.75x |
| window aggregate | 0.201 | 0.218 | 0.92x |
| window rank | 0.0267 | 0.0354 | 0.75x |

Thus the current query implementation is no longer a performance regression
against first-generation AELMDB on this workload; it is modestly faster on
prefix/range/window aggregation and substantially faster on rank/select.

For DUPSORT, rank/select use the same weight-only traversal.  Rank-prefix inside
a duplicate container deliberately retains the generic fallback for now because
a boundary may split a dupset; this preserves the simpler current semantics and
can be optimized separately only if a DUPSORT-specific benchmark demonstrates a
need.

## 7. Performance status

- **bottom-up mutation engine:** satisfactory; no material structural regression;
- **plain aggregate maintenance:** satisfactory (~20% put overhead at H=32 in the
  representative run);
- **DUPSORT maintenance:** asymptotically satisfactory; higher constant factor but
  no dupset-cardinality blow-up;
- **optimized hash backend:** effective and should remain the production default;
- **very wide hashes:** intentionally expensive stress configuration;
- **query layer:** performance-consolidated for the measured plain workload; all
  measured query classes are now at or faster than the first-generation reference.

Further query optimization should be evidence-driven, in particular by dedicated
DUPSORT window/rank workloads rather than by modifying the mutation engine.

