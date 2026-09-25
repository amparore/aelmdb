# AELMDB evolution

This directory records how the product sources in `../src/` are derived from
the canonical LMDB 0.9.70 baseline used by this project.  It is provenance and
design material; it is not part of the normal product build.

The canonical chain is deliberately coarse-grained and organized by purpose:

```text
LMDB 0.9.70
    |
    | 01-base.patch
    v
hardened LMDB base
    |
    | 02-bottomup.patch
    v
bottom-up mutation engine
    |
    | 03-aggformat.patch
    v
aggregate-aware persistent format
    |
    | 04-aggregates.patch
    v
AELMDB == ../src/
```

`make verify-evolution` replays these four patches on `base/` and requires the
result to be byte-identical to every product source in `src/`.

## What is and is not ancestry

`reference/dlmdb/` and `reference/aelmdb-first-generation/` are lateral
references.  DLMDB supplied selected ideas and implementation techniques;
first-generation AELMDB supplied requirements, regressions and comparison
oracles.  Neither is a base of the derivation above.

The former development lineage (BU3..BU7, V2M1..V2Q1 and the v1 repair path)
was intentionally collapsed.  Its purpose was to make development auditable
while the architecture was changing.  The four patches here instead describe
the stable architectural transformations.

## Directory contents

- `base/`: canonical LMDB 0.9.70 snapshot used by the project.
- `patches/`: the four canonical transformations.
- `design/`: design contract for each transformation.
- `tools/`: replay/fingerprint/comparison utilities retained for engineering.
- `bench/`: benchmark source.
- `results/`: selected historical evidence and measurements.
- `reference/`: non-ancestral implementations used as sources of ideas or
  behavioral references.

## Product status

The product has feature parity with first-generation AELMDB for the aggregate
API retained by the project, including the optimized multi-64-bit-limb hashsum
and window/rank-range queries.  The first-generation implementation remains a
performance and behavioral reference, not an ancestor of the product.

Performance tooling and the current audit are under `bench/` and `results/`.
