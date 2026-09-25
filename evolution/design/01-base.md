# 01 - Hardened LMDB base

## Objective

Establish a small, reviewable base derived directly from the canonical LMDB
0.9.70 snapshot.  This stage contains no aggregate-tree functionality and no
AELMDB file-format extension.

## Changes

The stage imports only independent hardening needed by the later work:

- concrete transaction error propagation and interrupt support;
- the DLMDB alignment/access discipline for page numbers;
- the complete `EVEN(keysize)` node-layout/accounting rule;
- corrections for LMDB cursor synchronization hazards found by the
  differential test stream;
- zero-length value insertion avoids passing a null data pointer to `memcpy`,
  preserving ordinary LMDB empty-value semantics without undefined behavior.

DLMDB is a reference for these ideas, not an ancestor.  `01-base.patch` is a
direct patch from LMDB to this state.

## Invariants

- LMDB transaction/COW/page-allocation architecture is retained.
- No aggregate metadata exists.
- No counted-tree or prefix-compression feature from DLMDB is imported.
- MIDL is unchanged.
- Behavioral deviations from frozen LMDB are restricted to intentional bug
  fixes and the selected hardening above.

## Contract exported to stage 02

Stage 02 can assume consistent aligned page-number access, correct padded node
space accounting, usable concrete transaction error state, and synchronized
cursor state on the mutation paths exercised by the project.

## Validation

The permanent regressions are under `tests/regress/`; ordinary LMDB tests are
under `tests/lmdb/`.  The deterministic differential stream remains under
`tests/differential/` and documents where fixed 01 behavior intentionally
differs from frozen LMDB.
