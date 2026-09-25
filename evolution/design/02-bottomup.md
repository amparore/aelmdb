# 02 - Bottom-up mutation engine

## Objective

Refactor LMDB structural mutation so that a parent is changed only after the
child-level transformation is in its final state.  This is a generic B+tree
mutation-engine change: it introduces no aggregate format and no aggregate
public API.

## Core contract

> A mutation at level L may inspect its parent, but it does not publish a
> structural change to an ancestor until the transformation at level L is
> final.

A local level therefore reports a structural result upward instead of
recursively completing ancestor work while the child is still changing.
The relevant results are split, separator rekey and underflow; an unwind loop
propagates them toward the root.

For split propagation the ordering is:

```text
finish split(child)
    -> publish final left/right children to parent
    -> finish parent transformation
    -> publish parent result upward
```

The same principle applies to move/merge underflow handling and to root
changes.

## Nested DUPSORT contract

A duplicate subtree is mutated to completion before its owning primary item is
settled.  The final `MDB_db`/inline dupset state is therefore available when
the primary level is processed.

## `mt_unwind_prefix`

The engine records the boundary between the structurally changed suffix of the
cursor path and the untouched ancestor prefix.  Debug builds can verify that
pages above this boundary did not change.  Stage 04 uses this information but
does not need to reconstruct a pre-operation path.

## Invariants

- File format and public API are unchanged from stage 01.
- Search, transaction, COW and page-allocation semantics remain LMDB-like.
- Structural recursion in split/rebalance propagation is replaced by explicit
  local results and an unwind.
- A parent observes the final state of any child result it publishes.
- Final cursor paths are valid when control returns from structural mutation.

## What the patch achieves

After `02-bottomup.patch`, the tree mutation engine itself provides the order
needed by persistent subtree metadata.  Later aggregate maintenance can carry
or recompute only local child information; it no longer needs to discover what
LMDB did after the fact.
