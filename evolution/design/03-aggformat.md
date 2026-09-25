# 03 - Aggregate persistent format

## Objective

Introduce the on-disk and public-schema representation required for generic
subtree aggregates while keeping aggregate maintenance out of scope.

## Representation

Aggregate state is described by a schema and transported as an opaque prefix
on branch links plus aggregate totals stored where an `MDB_db` represents a
subtree.  The supported logical components are entries, keys and a modular
hash sum whose byte width is selected by `MDB_HASH_SIZE`.

The format layer owns:

- aggregate schema/flags and size computation;
- branch-link prefix layout and accessors;
- `MDB_db` aggregate totals;
- validation of malformed or incompatible aggregate metadata;
- format tagging needed because the stage-01 alignment rule changes physical
  node layout relative to unmodified LMDB.

## Separation from maintenance

Stage 03 defines how aggregate bytes are represented and validated.  It does
not define the mutation algorithm that keeps them correct.  That separation is
intentional: format correctness can be tested independently of maintenance.

## Invariants

- Aggregate bytes are opaque to the generic structural engine except for
  moving/copying branch-link storage correctly.
- The width of the branch prefix is a pure function of the aggregate schema.
- Odd and non-word-sized `MDB_HASH_SIZE` values remain valid at the format
  level; the future optimized limb implementation must preserve this external
  representation or explicitly constrain the API.
- Corrupt schema/layout combinations are rejected rather than repaired.

## Validation

`tests/format/schema.c` exercises schema and corruption handling;
`tests/format/opaque_prefix.c` stamps opaque link metadata and verifies that
structural changes transport it correctly.
