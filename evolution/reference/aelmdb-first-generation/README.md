# Branch: AELMDB (initial snapshot)

AELMDB is a **separate branch derived from LMDB**, not a successor of the
lineage 01..04.  It is kept frozen, byte for byte, as the behavioural
reference for the aggregate query API and as the subject of the comparison in
`docs/compare/`.

- Format tag: **A333** (aggmaint uses A335; the files are not interchangeable).
- This is the *initial* AELMDB: `MDB_AGG_SUBTREE_STRICT_FAST_BRANCH`,
  `MDB_DEBUG_AGG_PRINT` tracing and counted-tree residues are still present.
  The corrections made during the 19/09/2026 audit (A334) are described in
  `docs/history/` but are not part of this snapshot.
- `midl.[ch]` differ from LMDB's only by MSVC warning pragmas; they are kept to
  preserve the snapshot exactly.
- `MDB_HASH_SIZE` must be a multiple of 8 for this branch.

## Known findings (frozen, not corrected here)

| Id | Finding | Reproduce |
|---|---|---|
| C2 | named-DB alignment UB | see `docs/compare/aelmdb_vs_aggmaint.md` |
| C3 | `MDB_GET_BOTH` accepts an inexact duplicate; `mdb_del(key,data)` can delete the next greater duplicate | `make c3-repro`; audit fix: `make c3-fix-check` (`patches/branches/aelmdb/C3-get-both.patch`) |
| L1 | large-key variable-length profile, H=32: `MDB_CORRUPTED` (Located page was wrong type) on plain delete, key 1371, round 1 | `make compare-large-keys` |
| L2 | same profile, H=64: wrong `select` result in the first plain delete wave | `make compare-large-keys LARGE_HASHES=64` |
| L3 | same profile, H=256: wrong prefix aggregate (e=558/560, k=558/560), no internal error | `make compare-large-keys LARGE_HASHES=256` |

L1–L3 are confirmed and deterministic; root cause not yet investigated.

Also confirmed by the phase-0 battery (2026-09-24, `make test-large-keys`,
`make test-struct`), still without investigation:

- C3 appears in every stream with more than one duplicate per key; with the
  C3 audit patch applied (`tools/stage_src.sh aelmdb-c3`) the branch gives
  traces identical to LMDB on the LMDB API.
- L1 reproduces without any query, through the LMDB API only: long-key
  "half" profile, and 36/50 differential streams on aggregate DBs
  (`MDB_CORRUPTED` on delete).
- API differences: `MDB_FIRST_DUP`/`MDB_LAST_DUP` on a non-DUPSORT DB
  succeed (LMDB: `MDB_INCOMPATIBLE`); `mdb_cursor_put(MDB_CURRENT)` keeps an
  aliased key (LMDB loses it, `docs/findings/key_aliasing.md`).
- Contract difference: AELMDB rejects value-source HASHSUM updates under
  `MDB_WRITEMAP` (`MDB_INCOMPATIBLE`); aggmaint supports them.
