# Tests

Tests are organized by the property they protect rather than by the historical
milestone that introduced them.

| Directory | Purpose |
|---|---|
| `lmdb/` | Upstream LMDB `mtest` programs, used as compatibility smoke tests. |
| `api/` | First-generation AELMDB API/unit/keyhash/advanced suite, built against the current product through a small compatibility header for test-only helper APIs. |
| `semantics/` | Aggregate algebra, local fold, split/rebalance/DUPSORT maintenance, integration and aggregate-query semantics. Historical milestone filenames were replaced by semantic names; defect tags are retained only where they identify a specific regression. |
| `format/` | Aggregate schema/corruption checks, opaque branch-prefix transport, and cross-open support. |
| `regress/` | Concrete defects corrected in the LMDB-derived base: EVEN-key accounting and cursor hazards; key-aliasing characterization. |
| `differential/` | Deterministic operation stream, large-key structural stress and bottom-up rebalance stress. |
| `compare/` | Product-only deterministic workloads checked against frozen expected signatures. Historical multi-implementation comparison tooling lives under `evolution/tools/compare/`. |

Top-level targets are `make test-*`; `make test` runs the product-focused core
suite.  Expensive LMDB/differential stress is available separately.

The API compatibility header now exposes the production window query and retains
only test-local helper shims that are intentionally not part of the public API.
`make test-generic-hash` additionally runs semantic stress with odd hash sizes
using the generic byte-wise arithmetic backend.

## Strong diagnostic builds

`make test-differential` builds the product with both aggregate-integrity and
unwind-boundary diagnostics enabled, so the structural stress tests exercise
the same prefix invariant as the semantics suite.

`make test-sanitize` runs a representative structural/query semantics battery
with AddressSanitizer and UndefinedBehaviorSanitizer (split, rebalance, aggregate queries, current-subpage and dupset bounds). It is
deliberately separate from `make check` because sanitizer builds are
substantially heavier. Leak detection is disabled
for this target because the debug-only unwind snapshot retains a per-thread
buffer for the process lifetime.
