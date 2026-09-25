# **AELMDB**: Anti-Entropy Lightning Memory-Mapped Database

**Current release: AELMDB 0.2.0**

**AELMDB (Anti-Entropy LMDB)** is a fork of **[LMDB](https://github.com/LMDB/lmdb)** that augments its B+tree with **maintained subtree aggregates** and a **bottom-up mutation engine**. In practice, this turns LMDB’s ordered keyspace into something you can query by **position** and **summarize by range** without scanning leaf pages, while keeping aggregate maintenance local to the pages affected by each update.

This enables:
- **Order-statistics** on the database order (fast `rank` / `select`), extending the counted/order-statistics B-tree design used in **[DLMDB](https://github.com/datalevin/dlmdb)**
- **Fast range summaries** (counts + fixed-size **hashsums**) that are composable and efficient for large datasets
- **Anti-entropy / reconciliation-friendly primitives**, specifically targeting **Range-Based Set Reconciliation ([RBSR](https://logperiodic.com/rbsr.html))**, where peers repeatedly compare and split ordered ranges using compact aggregates
- **Bottom-up B+tree mutation**, where structural changes are settled from child to parent and aggregate changes are propagated upward as local delta updates



<br/>

# 1) Core concept: aggregate-enabled DBIs (counts + hashsum)

LMDB is a **low-level, embedded key–value database** built around a memory-mapped B+tree: it stores **sorted `(key, value)` pairs** and provides very fast point lookups and ordered iteration.
Keys and values are treated as **opaque byte strings** (`MDB_val`: pointer + length). <br/>
AELMDB keeps that model, but adds an optional *interpretation layer* for **anti-entropy hash aggregates**:

* a record is still just a `(key,value)` tuple of raw bytes
* but *if you opt in* (via DBI flags), AELMDB assumes that **within each record there exists a fixed-size “hash slice”**:

  * either inside the **value bytes** (default), or inside the **key bytes** (when `MDB_AGG_HASHSOURCE_FROM_KEY` is enabled)
  * at a configurable **signed byte offset** (hash_offset): >=0 from start, <0 from end (-1 = last bytes)
  * with a fixed width of `MDB_HASH_SIZE` bytes

When enabled, AELMDB maintains a **hashsum aggregate** by summing (with wraparound arithmetic) that `MDB_HASH_SIZE`-byte slice for every record in a subtree. This makes it possible to compute **range aggregates** efficiently (a building block for anti-entropy / reconciliation), without scanning all records in the range.

In the same mechanism, AELMDB can also maintain **counts** (entries and/or distinct keys), enabling efficient order-statistics queries (rank/select) and fast range counts.

## Bottom-up mutation and aggregate maintenance

AELMDB reworks LMDB’s B+tree mutation path around a **bottom-up structural update model**. This is a core property of the database engine itself, not an aggregate-specific mechanism.

During structural operations such as splits, rekeys, moves, and merges, each affected child level is brought to its final state **before** the corresponding update is published to its parent. Parent links therefore always refer to child pages whose structure and contents are already definitive for the current mutation.

This child-before-parent ordering provides a strong invariant for aggregate maintenance. For a logical record change, AELMDB computes the contribution **before** and **after** the operation and propagates that delta bottom-up through the structurally unchanged ancestor prefix. Levels affected by structural rewrites instead publish the **exact aggregate of their finalized child pages**. Aggregate state is therefore maintained from the final state produced by the mutation, without subtree scans or post-hoc repair passes.

For `MDB_DUPSORT`, the duplicate tree is completed first; only then is the final contribution of the primary `(key, duplicate-set)` item settled in the primary tree.

Compared with LMDB’s original mutation flow, this gives AELMDB a more explicit and locally verifiable structural update contract, and provides the foundation for maintaining counts and hashsums incrementally with bounded work along the modified B+tree path.

## Validation, stress testing and performance

AELMDB has been developed with an **invariant-driven, differential test strategy**. The test suite does not only exercise the public aggregate API: it compares LMDB-compatible behavior against the upstream baseline, checks persistent tree structure, independently recomputes aggregate state, stresses deep split/merge paths, and validates the bottom-up mutation contract itself.

Validation includes:

- the upstream LMDB `mtest` programs and the AELMDB API/unit/keyhash/advanced suites;
- deterministic differential workloads covering `put`/`delete`, cursor updates, `MDB_MULTIPLE`, `MDB_RESERVE`, `MDB_APPEND*`, nested transactions, `drop`, plain DBs, `MDB_DUPSORT`, and `MDB_DUPFIXED`;
- large-key structural stress designed to create low-fanout and taller trees, plus targeted split, rebalance, root-collapse, deep-duplicate-tree, and cursor-fixup workloads;
- an **independent aggregate-integrity oracle** that recursively recomputes subtree aggregates, and a separate **unwind-boundary oracle** that verifies the child-before-parent structural invariant, including duplicate subtrees;
- frozen aggregate signatures, cross-format/open tests, regression tests for previously discovered defects, and byte-for-byte replay of the four-stage evolution chain;
- dedicated ASan/UBSan builds and a portable byte-wise hash backend used with odd aggregate widths to stress representation and alignment independently of the production 64-bit-limb backend.

The long differential campaigns used during consolidation included **200 seeds × 20,000 operations**, **20 seeds × 100,000 operations**, focused duplicate-set churn, and **300 aggregate-enabled seeds**; an additional 20-seed campaign ran the independent aggregate-integrity oracle after every write. The resulting operation traces matched LMDB for the LMDB-compatible behavior under test.

Coverage was measured specifically on the rewritten **bottom-up mutation core** after the focused structural campaign: **1,407 / 1,534 lines (91.7%)** and **799 / 962 branches (83.1%)** were exercised. Core structural functions reached similarly high line coverage (`mdb_page_split_local`: 97%, `mdb_rebalance_root`: 94%, `mdb_node_move_local`: 91%). The remaining uncovered code is dominated by rare allocation/I/O error paths, spill paths, and defensive branches.

### Performance impact

Performance has been measured against the **LMDB 0.9.70 baseline** using the same `-O3 -DNDEBUG`, `MDB_NOSYNC` benchmark configuration on a pinned CPU. These figures are engineering measurements from the release audit, not portable absolute performance guarantees; small differences of a few percent should be treated as noise.

For a representative workload of 500k random plain puts:

| Configuration | Put time vs LMDB | Interpretation |
|---|---:|---|
| LMDB 0.9.70 | **1.00×** | baseline |
| bottom-up engine (`02-bottomup`) | **1.01×** | structural rewrite alone |
| AELMDB 0.2.0, aggregate features disabled | **1.05×** | full AELMDB engine with ordinary LMDB DBIs |
| AELMDB 0.2.0, `ENTRIES + KEYS + HASHSUM`, 32-byte hash | **1.20×** | full aggregate maintenance enabled |

On the same aggregate-enabled workload, delete measured approximately **1.16×** the LMDB baseline. Thus the bottom-up engine itself adds only a few-percent structural cost in the measured workload, while the optional aggregate maintenance accounts for most of the additional write overhead. The optimized 64-bit-limb hashsum backend is materially faster than the portable byte-wise validation backend (about **1.26×** faster for plain aggregate puts and **1.36×** for the measured DUPSORT workload at a 32-byte hash width).

DUPSORT stress also shows no evidence of per-update work growing linearly with duplicate-set cardinality: the measured cost per inserted duplicate remains broadly stable as duplicate sets grow, with step changes attributable to representation/tree-height transitions rather than full duplicate-set rescans.

The query layer is likewise benchmarked as part of the release process. Aggregate prefix/range/window queries are at least comparable to the first-generation AELMDB reference on the measured plain workload, while rank/select are faster. Detailed methodology and raw release-audit results are kept in [`evolution/results/performance-audit.md`](evolution/results/performance-audit.md); structural coverage details are in [`evolution/results/bottomup-coverage.md`](evolution/results/bottomup-coverage.md).

### New DBI flags (aggregate schema)

These flags (passed to `mdb_dbi_open()`) select which aggregate components are maintained in branch pages:

* `MDB_AGG_ENTRIES`: counts logical **records** (key/value pairs). For `MDB_DUPSORT`, each duplicate counts as one record.
* `MDB_AGG_KEYS`: counts **distinct keys** in the primary tree.
* `MDB_AGG_HASHSUM`: maintain a fixed-size wraparound accumulator of _hash slices_ from each entry.
* `MDB_AGG_HASHSOURCE_FROM_KEY`: uses the **key bytes** as the hash source instead of value bytes. 
  * Only meaningful with `MDB_AGG_HASHSUM`, and **incompatible with `MDB_DUPSORT`**.

**`MDB_RESERVE` note:** value-sourced `MDB_AGG_HASHSUM` is incompatible with `MDB_RESERVE`, because the reserved value bytes are filled after the put call returns and therefore cannot be included in the maintained hash contribution at mutation time. Key-sourced hashsums on plain DBIs may use `MDB_RESERVE`.



## Hash slice definition

When `MDB_AGG_HASHSUM` is enabled, AELMDB assumes that each record (entry) contains a fixed-size **hash slice**—a contiguous byte window that will be summed into the range hashsum.
For every entry, AELMDB extracts exactly **`MDB_HASH_SIZE` bytes**:

* **From where (hash source):**
  * by default, from the entry’s **value** bytes
  * if `MDB_AGG_HASHSOURCE_FROM_KEY` is set, from the entry’s **key** bytes instead
* **From which position:** the DBI’s configured **`hash_offset`** (set once via `mdb_set_hash_offset()` on an empty DBI) selects the **start** of the `MDB_HASH_SIZE` slice:

  * `hash_offset >= 0`: start at `hash_offset` bytes from the beginning
  * `hash_offset < 0`: start from the end, where `-1` means “use the last `MDB_HASH_SIZE` bytes” (and `-k` means `(k-1)` bytes earlier)

The extracted slice is `source[start .. start + MDB_HASH_SIZE)` where `source` is `value` (default) or `key` (when `MDB_AGG_HASHSOURCE_FROM_KEY` is set).

`MDB_HASH_SIZE` defaults to **32 bytes**. Production builds use the optimized **64-bit-limb** arithmetic backend and therefore require `MDB_HASH_SIZE` to be a **multiple of 8**. For validation and alignment stress testing, defining `MDB_AGG_GENERIC_HASH_ARITH=1` enables the portable byte-oriented backend and permits arbitrary hash widths in the supported range **1..256 bytes**. The selected width is part of the AELMDB environment format: opening an environment with a build configured for a different `MDB_HASH_SIZE` is rejected with `MDB_VERSION_MISMATCH`.


---
## Plain vs `MDB_DUPSORT` semantics

In LMDB, a stored item (entry, record) is always a **single `(key, value)` pair**. 
A DBI can be configured in one of two relevant modes:

* **Plain DBI (no `MDB_DUPSORT`)**
  Each key appears **at most once**, so there is a 1:1 relationship between keys and entries. 
  Iteration order is by **key**.

* **`MDB_DUPSORT` DBI (duplicates enabled)**
  A key may have **multiple values**. 
  Each `(key, value)` pair is still a separate **entry**, but entries are ordered by the pair **(key, value)**: keys are ordered first, and for the same key the values are ordered and iterated in order.

| Concept               | Plain DB                                             | `MDB_DUPSORT` DB                                                             |
| --------------------- | ---------------------------------------------------- | ---------------------------------------------------------------------------- |
| Total order           | by `key`                                             | by `(key, value)`                                                            |
| What an “entry” is    | one `(key,value)`                                    | one `(key, value)` (duplicates are additional entries)                       |
| `ENTRIES` counts      | number of entries                                    | number of entries (includes duplicates)                                      |
| `KEYS` counts         | number of distinct keys                              | number of distinct keys that appear (regardless of how many values each has) |
| `HASHSUM` sums        | hash slices from *hash source* (either value or key) | hash slices from **value** *hash source* (key-hash mode not allowed)         |
| Key-based hash source | allowed                                              | **not allowed** (`MDB_AGG_HASHSOURCE_FROM_KEY` incompatible)                 |

**Note:** In a plain DBI, **`MDB_AGG_KEYS` and `MDB_AGG_ENTRIES` are equivalent** because every key has exactly one value. 
These counters diverge only with `MDB_DUPSORT`, where one key can contribute many entries.


---
### Opening DBIs with aggregate flags

When you open (or create) a DBI, you choose an **aggregate schema** by OR-ing `MDB_AGG_*` flags into `mdb_dbi_open()`. That schema determines which per-subtree aggregates are maintained in branch pages and therefore which queries are available later (range summaries, rank/select).

If you enable `MDB_AGG_HASHSUM`, you must also configure **where the hash slice lives** inside the chosen hash source (value by default, or key in key-hash mode) using `mdb_set_hash_offset()`. This configuration is per-DBI, must be done in a **write transaction**, and only while the DBI is **still empty**.

**Example: plain aggregate DBI (hash slice taken from value bytes)**
This DBI maintains entry counts, distinct-key counts, and a hashsum aggregate from the defined hash slice.

```c
unsigned flags =
  MDB_CREATE |
  MDB_AGG_ENTRIES |   /* maintain entry (record) counts */
  MDB_AGG_KEYS    |   /* maintain distinct-key counts */
  MDB_AGG_HASHSUM;    /* maintain hashsum aggregate */

MDB_dbi dbi;
mdb_dbi_open(txn, "plain_agg", flags, &dbi);

/* hash_offset: 0 = first bytes, -1 = last bytes */
mdb_set_hash_offset(txn, dbi, /*hash_offset = */0);
```

**Example: `MDB_DUPSORT` aggregate DBI (multiple values per key)**
With `MDB_DUPSORT`, each `(key,value)` pair is still an entry, but the database’s total order is by `(key,value)` (values ordered within each key). Here, `ENTRIES` counts duplicates as additional entries, while `KEYS` counts distinct keys.

```c
unsigned flags =
  MDB_CREATE |
  MDB_DUPSORT |
  MDB_AGG_ENTRIES | MDB_AGG_KEYS | MDB_AGG_HASHSUM;

MDB_dbi dbi;
mdb_dbi_open(txn, "dups_agg", flags, &dbi);

/* DUPSORT: still value-based hashsum; hash_offset uses the same signed semantics */
mdb_set_hash_offset(txn, dbi, 0);
```

**Example: key-based hashsum (hash slice taken from key bytes)**
This mode is useful when your key embeds a stable fixed-size identifier (e.g., `[timestamp | 32-byte-id]`) and you want the hash slice derived from keys rather than values. 
In this configuration the hash slice is `key[hash_offset .. hash_offset + MDB_HASH_SIZE)`.

```c
unsigned flags =
  MDB_CREATE |
  MDB_AGG_ENTRIES | MDB_AGG_KEYS | MDB_AGG_HASHSUM |
  MDB_AGG_HASHSOURCE_FROM_KEY; /* use KEY bytes as hash source */

MDB_dbi dbi;
mdb_dbi_open(txn, "key_hash_agg", flags, &dbi);

/* If key has structure [prefix | 32-byte-hash | suffix] use
 *  - if prefix has fixed length PREFIX_LEN then use hash_offset = PREFIX_LEN
 *  - if suffix has fixed length SUFFIX_LEN then use hash_offset = -1 - SUFFIX_LEN */
mdb_set_hash_offset(txn, dbi, /*hash_offset=*/PREFIX_LEN);
```

As said before, the hash source is the value by default, and becomes the key when `MDB_AGG_HASHSOURCE_FROM_KEY` is set.





---
### Hashsum wraparound algebra

AELMDB’s hashsum is a fixed-size **`MDB_HASH_SIZE`-byte accumulator**. Conceptually it behaves like a single unsigned integer and all operations are performed **modulo `2^(8*MDB_HASH_SIZE)`** (classical wraparound arithmetic). Addition and subtraction therefore compose naturally across subtree summaries and range differences.

The default backend operates on **64-bit limbs** with carry/borrow support optimized for the host compiler, while preserving safe loads/stores for unaligned data. A portable byte-oriented backend can be enabled with `MDB_AGG_GENERIC_HASH_ARITH=1`; it is intentionally retained for validation builds with odd or unusual aggregate widths.

AELMDB also exposes a small helper layer using exactly the same arithmetic as the maintained aggregates:

```c
void mdb_hashsum_add(uint8_t *acc, const uint8_t *x);
void mdb_hashsum_sub(uint8_t *acc, const uint8_t *x);
void mdb_hashsum_diff(uint8_t *out, const uint8_t *a, const uint8_t *b);
int  mdb_hashsum_is_zero(const uint8_t *p);
int  mdb_hashsum_extract_bytes(const void *p, size_t sz, int hash_offset,
                               uint8_t out[MDB_HASH_SIZE]);
int  mdb_hashsum_extract(const MDB_val *data, int hash_offset,
                         uint8_t out[MDB_HASH_SIZE]);
```

The logical hash slice itself remains defined solely by the DBI schema: source (`value` or `key`), signed `hash_offset`, and `MDB_HASH_SIZE`.








<br/>

# 2) Aggregate API (totals / prefix / range)

The Aggregate API provides **fast summaries over sets of records** (entire DB, a prefix, or a bounded range) using the aggregate metadata stored in branch pages. Each function returns an `MDB_agg` structure containing the components enabled for that DBI (entries, keys, and/or hashsum). If a DBI wasn’t created with the required aggregate components, these functions return `MDB_INCOMPATIBLE`.

## Result type

```c
/* Aggregate summary (returned by totals/prefix/range queries) */
typedef struct MDB_agg {
  unsigned mv_flags;                      // OUT: which aggregate components are valid (MDB_AGG_* bits)
                                          //      set by aggregate functions to match the DBI's schema
  uint64_t mv_agg_entries;                // OUT: ENTRIES component (number of records / entries)
  uint64_t mv_agg_keys;                   // OUT: KEYS component (number of distinct keys)
  uint8_t  mv_agg_hashes[MDB_HASH_SIZE];  // OUT: HASHSUM component (wraparound accumulator)
} MDB_agg;
```

**How to read `mv_flags`:** it tells you which fields are meaningful for this DBI. For example, if the DBI was opened with `MDB_AGG_ENTRIES | MDB_AGG_HASHSUM`, then `mv_flags` will include those bits and you should read `mv_agg_entries` and `mv_agg_hashes`, but ignore `mv_agg_keys`.



## Basic aggregate functions

### `mdb_agg_info()`

```c
/* Query the aggregate schema enabled for a DBI */
int mdb_agg_info(
  MDB_txn   *txn,        // IN: transaction handle
  MDB_dbi    dbi,        // IN: target DBI
  unsigned  *agg_flags   // OUT: MDB_AGG_* bitmask enabled for this DBI
);
```

Use this when you want to **detect at runtime** which aggregate components are available (e.g., to decide whether you can run rank/select on that DBI).

### `mdb_agg_totals()`

```c
/* Compute aggregate summary over the entire DBI */
int mdb_agg_totals(
  MDB_txn  *txn,   // IN: transaction handle
  MDB_dbi   dbi,   // IN: target DBI
  MDB_agg  *out    // OUT: totals for the whole DBI (mv_flags set to schema)
);
```

Returns the **full-database aggregate (incl. hashsum)** in `*out`—the quickest way to obtain overall entry count / key count / full hashsum.



### Prefix aggregate

The **prefix aggregate** API computes aggregate totals over the *prefix* of the DBI’s ordered records: i.e., **all records that come before a given pivot** in the DB’s total order.
It returns those totals in `MDB_agg` (entries/keys/hashsum as enabled), letting you get “counts and hashsums up to here” without scanning.


```c
#define MDB_AGG_PREFIX_INCL 0x01  // include the pivot record if it exists (when using record-order pivots)

int mdb_agg_prefix(
  MDB_txn       *txn,    // IN: transaction handle
  MDB_dbi        dbi,    // IN: target DBI
  const MDB_val *key,    // IN: pivot key
  const MDB_val *data,   // IN: optional pivot data (only meaningful for MDB_DUPSORT record-order pivots)
  unsigned       flags,  // IN: 0 or MDB_AGG_PREFIX_INCL
  MDB_agg       *out     // OUT: aggregates of the prefix set (mv_flags set to schema)
);
```

**What you get in `*out`:** the aggregate summary of the prefix set (entries/keys/hashsum per the DBI schema).

**Pivot semantics:**
* Plain DB (or `data == NULL`): pivot is **key-only** (`key`).
* `MDB_DUPSORT` with `data != NULL`: pivot is the record-order pair **`(key,data)`**; `MDB_AGG_PREFIX_INCL` optionally includes that exact record.





### Range aggregate

The range aggregate computes aggregates over all records **between two bounds**, with explicit control over **whether each bound is included or excluded**. This is the core primitive for range aggregates.

```c
int mdb_agg_range(
  MDB_txn       *txn,        // IN: transaction handle
  MDB_dbi        dbi,        // IN: target DBI
  const MDB_val *low_key,    // IN: optional lower-bound key (NULL => open-ended)
  const MDB_val *low_data,   // IN: optional lower-bound data (record-order bound for MDB_DUPSORT)
  const MDB_val *high_key,   // IN: optional upper-bound key (NULL => open-ended)
  const MDB_val *high_data,  // IN: optional upper-bound data (record-order bound for MDB_DUPSORT)
  unsigned       flags,      // IN: MDB_RANGE_* flags controlling inclusion of bounds
  MDB_agg       *out         // OUT: aggregates over the selected range (mv_flags set to schema)
);
```

**Bound inclusion rules:**

* bounds are **exclusive by default**
* include the lower bound with `MDB_RANGE_LOWER_INCL`
* include the upper bound with `MDB_RANGE_UPPER_INCL`

**Boundary interpretation:**

* Plain DB (or when `*_data` is NULL): bounds are evaluated **by key**
* `MDB_DUPSORT` with `*_data` provided: bounds can be evaluated in **record order `(key,data)`**

`*out` contains the same components as other aggregate calls (entries / keys / hashsum), computed over exactly the records that fall inside the chosen bounds.









<br/>

# 3) Order-statistics (rank / select / seek-by-rank)

**Order-statistics** are “position-based” queries on an ordered collection. They let you ask:

* **rank(x)**: “how many items come before *x*?”
* **select(i)**: “what is the item at position *i*?”

In a standard LMDB database, answering these usually requires scanning forward and counting. In AELMDB, they are fast because the B+tree maintains **subtree counters in branch pages** (enabled via `MDB_AGG_ENTRIES` and/or `MDB_AGG_KEYS`). Order-statistics queries combine those counters while descending the tree, instead of iterating records. If the DBI wasn’t created with the required counter, the corresponding call returns `MDB_INCOMPATIBLE`.

**Rank units (what “position” means)**: 
AELMDB exposes two *rank spaces*. You choose which one you want via `MDB_agg_weight`:

* **Entry-rank (`MDB_AGG_WEIGHT_ENTRIES`)** — position measured in **records** (entries).

  * Plain DB: one `(key,value)` is one entry.
  * `MDB_DUPSORT`: each duplicate value is also an entry, and iteration order is by `(key,value)`.

* **Key-rank (`MDB_AGG_WEIGHT_KEYS`)** — position measured in **distinct keys**.

  * Each distinct key contributes exactly one unit, regardless of how many values it has in `MDB_DUPSORT`.

In other words, the same DBI can be viewed as two ordered index spaces: “by every stored record” (entries), or “by distinct keys” (keys).


## Order-statistics functions 

### `mdb_agg_rank()`

Computes the **zero-based rank** of a target in the DBI’s total order, in either entry-units or key-units.
It supports an **exact** mode (must match) and a **set-range** mode (find the first record ≥ the query and return its rank), and for `MDB_DUPSORT` + entry-rank it can also report the duplicate index within the key’s value set.

```c
#define MDB_AGG_RANK_EXACT      0u  // require an exact match for the queried key/(key,value)
#define MDB_AGG_RANK_SET_RANGE  1u  // set-range: locate first record >= (key,value) and return its rank

int mdb_agg_rank(
  MDB_txn        *txn,       // IN: transaction handle
  MDB_dbi         dbi,       // IN: target DBI
  MDB_val        *key,       // IN: query key; OUT: located key (SET_RANGE mode)
  MDB_val        *data,      // IN/OUT: query/located value; must be a valid MDB_val pointer
  MDB_agg_weight  weight,    // IN: MDB_AGG_WEIGHT_ENTRIES or MDB_AGG_WEIGHT_KEYS
  unsigned        flags,     // IN: MDB_AGG_RANK_EXACT or MDB_AGG_RANK_SET_RANGE
  uint64_t       *rank,      // OUT: zero-based rank in the chosen unit
  uint64_t       *dup_index  // OUT (optional): DUPSORT+entries -> index within duplicates; keys -> 0
);
```

### `mdb_agg_select()`

Returns the record at a given **zero-based rank** (entry-rank or key-rank) without scanning.
For `MDB_DUPSORT` + key-rank, it returns the first value for that key; for entry-rank, it selects an exact `(key,value)` record and can report `dup_index`.

```c
int mdb_agg_select(
  MDB_txn        *txn,       // IN: transaction handle
  MDB_dbi         dbi,       // IN: target DBI
  MDB_agg_weight  weight,    // IN: MDB_AGG_WEIGHT_ENTRIES or MDB_AGG_WEIGHT_KEYS
  uint64_t        rank,      // IN: zero-based rank in the chosen unit
  MDB_val        *key,       // OUT: selected key
  MDB_val        *data,      // OUT: selected value (for DUPSORT+key-rank: first duplicate)
  uint64_t       *dup_index  // OUT (optional): DUPSORT+entries -> index within duplicates
);
```

### `mdb_agg_cursor_seek_rank()`

Positions an existing cursor at a given **entry rank** and returns the record at that position.
This is the “jump then iterate” primitive: you seek by rank once, then use normal cursor iteration (`MDB_NEXT`, etc.) efficiently from that point. `key` and `data` may be `NULL` when the caller only needs to reposition the cursor.

```c
int mdb_agg_cursor_seek_rank(
  MDB_cursor *mc,     // IN: cursor opened on the target DBI
  uint64_t    rank,   // IN: zero-based entry rank (entries only)
  MDB_val    *key,    // OUT: key at that rank
  MDB_val    *data    // OUT: value at that rank (optional)
);
```











<br/>

# 4) Advanced helpers (Negentropy-style windows)

**Set Reconciliation** protocols keep two replicas in sync by *exchanging compact summaries* of what each side has, then drilling down only where those summaries disagree. **Range-Based Set Reconciliation (RBSR)** does this over a *totally ordered set*: peers compute an **aggregate of a sub-range**, compare it, and if it mismatches they **split the range into subranges** and repeat recursively until the differing parts are small enough to enumerate explicitly. (see [arXiv - Range-Based Set Reconciliation](https://arxiv.org/abs/2212.13567))

For a storage backend, this creates very specific hot-path requirements:

* **Fast range aggregates**: repeatedly compute aggregates for many **overlapping subranges** (often created by successive splits), without scanning all records in each subrange. (see [Range-Based Set Reconciliation](https://aljoscha-meyer.de/assets/landing/rbsr.pdf))
* **Fast split/navigation**: given a range, quickly find “midpoints” (often by entry rank) and quickly map a queried key to a **lower-bound position** inside that same range.
* **Reuse across subranges**: reconciliation loops tend to query many subranges within the *same* outer bounds, so recomputing “where the window starts/ends” over and over is wasted work.


AELMDB’s **window subrange APIs** are built specifically for this. 
They cache the **key-bounds → entry-rank mapping**, mapping a key-range window to an **absolute entry-rank interval**, and then let a program cheaply:

1. compute a range **aggregate** (an `MDB_agg` summary, typically using `HASHSUM`) for any **relative entry-rank subrange** inside the window, and
2. compute a **window-relative lower-bound rank** for a key (and optionally value for `MDB_DUPSORT`) without re-deriving the window mapping each time.


This is the exact pattern used by Negentropy-style reconciliation loops: many aggregate calls + many lower-bound calls, all within stable outer bounds. (see [negentropy](https://github.com/hoytech/negentropy))

See also the AELMDB storage in Negentropy for Range-Based Set Reconciliation, named [AELMDBSlice](https://github.com/amparore/negentropy-aelmdb), and the extended C++ wrapper [lmdbxx](https://github.com/amparore/lmdbxx-aelmdb).



## Window subrange descriptor

`MDB_agg_window` is the cached “outer range mapping” that makes repeated subrange queries cheap. You initialize it to zero, then reuse it for subsequent queries as long as the bounds/flags stay the same.

```c
#define MDB_AGG_WINDOW_END UINT64_MAX // sentinel: rel_end means "use the full window end"

/* Mapping from a key-range window to an absolute entry-rank interval */
typedef struct MDB_agg_window {
  unsigned mv_flags;         // OUT: DBI aggregate schema (MDB_AGG_* bits) cached for this window
  unsigned mv_range_flags;   // OUT: MDB_RANGE_* flags defining inclusion/exclusion of window bounds
  uint64_t mv_total_entries; // OUT: cached total entries in DBI (used for open-ended/clamping cases)
  uint64_t mv_abs_lo;        // OUT: absolute entry-rank of the window's lower bound
  uint64_t mv_abs_hi;        // OUT: absolute entry-rank of the window's upper bound
} MDB_agg_window;
```

**Rationale:** reconciliation repeatedly queries many subranges within the same outer bounds; recomputing the outer bounds’ absolute entry-ranks every time wastes work. 
`MDB_agg_window` stores that mapping `(low_key, high_key, range_flags) → [mv_abs_lo, mv_abs_hi)`, so subsequent aggregates and lower-bound queries can run in **window-relative rank coordinates**.

**Usage rules:** zero-initialize before first use; reuse only with the same bounds and `range_flags` (not checked).



## Window subrange functions 

### `mdb_agg_window_aggregate()`

Computes aggregates for a **relative entry-rank subrange** inside a subrange window.
On first use of a zero-initialized descriptor, it computes the window’s absolute rank interval; subsequent calls reuse that cached mapping. If the window bounds change, the caller must reinitialize (zero) the descriptor before reuse.

```c
int mdb_agg_window_aggregate(
  MDB_txn        *txn,         // IN: transaction handle
  MDB_dbi         dbi,         // IN: target DBI
  const MDB_val  *low_key,     // IN: optional window lower key (NULL => open-ended)
  const MDB_val  *low_data,    // IN: optional window lower value (record-order bound for MDB_DUPSORT)
  const MDB_val  *high_key,    // IN: optional window upper key (NULL => open-ended)
  const MDB_val  *high_data,   // IN: optional window upper value (record-order bound for MDB_DUPSORT)
  unsigned        range_flags, // IN: MDB_RANGE_* controlling inclusion/exclusion of window bounds
  MDB_agg_window *window,      // IN/OUT: window subrange descriptor (must be zeroed initially)
  uint64_t        rel_begin,   // IN: relative begin entry-rank within window
  uint64_t        rel_end,     // IN: relative end entry-rank within window (or MDB_AGG_WINDOW_END)
  MDB_agg        *out          // OUT: aggregates over [rel_begin, rel_end) within the window
);
```

Window bounds follow the same inclusion/exclusion rules as `mdb_agg_range()`. 
The subrange itself is expressed in **relative entry-rank space** within the window.


### `mdb_agg_window_rank()`

Computes the **lower-bound position** of a key (and optional value for `MDB_DUPSORT`) **relative to the subrange window**.
This is typically used to place split points and to map protocol “cursor keys” into window-relative ranks during reconciliation.

```c
int mdb_agg_window_rank(
  MDB_txn        *txn,         // IN: transaction handle
  MDB_dbi         dbi,         // IN: target DBI
  const MDB_val  *low_key,     // IN: window lower key (must match window)
  const MDB_val  *low_data,    // IN: window lower value
  const MDB_val  *high_key,    // IN: window upper key
  const MDB_val  *high_data,   // IN: window upper value
  unsigned        range_flags, // IN: MDB_RANGE_* (must match window)
  MDB_agg_window *window,      // IN/OUT: window subrange descriptor
  const MDB_val  *key,         // IN: query key (lower-bound search)
  const MDB_val  *data,        // IN: optional query value (record-order for MDB_DUPSORT)
  uint64_t       *rel_rank     // OUT: relative rank within window [0, window_size]
);
```

It finds the first record ≥ `(key,data)` in the DBI’s total order, clamps that absolute rank into `[mv_abs_lo, mv_abs_hi)`, and returns the resulting **window-relative** rank.







<br/>

# Debug & validation support

Because AELMDB maintains extra on-page metadata (counts + hashsums), the source also includes **opt-in diagnostic oracles** intended for development, stress testing and validation rather than production use. These checks are deliberately independent from the maintenance path: they verify stored metadata instead of repairing it.

### `MDB_DEBUG_AGG_INTEGRITY` and `mdb_agg_check_integrity()`

When AELMDB is compiled with `MDB_DEBUG_AGG_INTEGRITY=1`, the header exposes a diagnostic API that recursively walks the complete DBI (including persistent `MDB_DUPSORT` subtrees), recomputes exact aggregates, verifies every stored branch prefix, and checks the root result against the `MDB_db` totals:

```c
#if defined(MDB_DEBUG_AGG_INTEGRITY) && MDB_DEBUG_AGG_INTEGRITY
int mdb_agg_check_integrity(MDB_txn *txn, MDB_dbi dbi);
#endif
```

### `MDB_DEBUG_UNWIND`

`MDB_DEBUG_UNWIND=1` enables a second structural oracle used by the test suite. It checks the bottom-up mutation contract: ancestor pages above the reported structural rewrite boundary must remain byte-identical until logical aggregate settlement is applied. Primary and duplicate-tree paths are checked independently.

Both mechanisms are validation tools only; neither participates in normal aggregate maintenance or error recovery.






<br/>

# 5) On-disk format and compatibility

AELMDB’s aggregate features change how **branch pages are laid out on disk** by adding per-subtree aggregate metadata. Because of that, aggregate support is **not a runtime toggle**, it must be part of the DBI’s definition from the start.

Practical implications:

* **Enable aggregate flags at DBI creation time.** The DBI’s aggregate schema (`MDB_AGG_*`) is stored/validated against page headers; mismatches are treated as corruption.
* **Changing the aggregate schema after data exists generally requires rebuilding.** If a DBI already has branch pages, switching aggregate flags typically means copying all data into a fresh DBI created with the desired flags. 
* ⚠️**Important! The AELMDB data format is different and binary incompatible with that of both LMDB and DLMDB.**



## Relationship to the counted-DB API of DLMDB

AELMDB takes inspiration from [DLMDB](https://github.com/datalevin/dlmdb/tree/main) (Datalevin’s LMDB fork), in particular its *counted B-tree* order-statistics design and selected engineering ideas. DLMDB remains a separate project with its own format and API.

DLMDB-style counted-tree functionality provides:

* a single `MDB_COUNTED` DBI flag, and
* `mdb_counted_*` APIs focused on **record-count order-statistics** (entries, rank, select). 

AELMDB generalizes this idea into a richer, reconciliation-oriented aggregate layer:

* **Multiple aggregate components**: `MDB_AGG_ENTRIES` (records), `MDB_AGG_KEYS` (distinct keys), `MDB_AGG_HASHSUM` (range aggregate accumulator).
* **Richer aggregate queries** beyond “count”: prefix and range aggregations returning a full `MDB_agg` summary (counts + aggregate where enabled).
* **Window subrange helpers** designed for **Range-Based Set Reconciliation** loops, where you repeatedly compute aggregates and lower-bound ranks inside stable outer bounds.

A simple conceptual mapping:

* `MDB_COUNTED` → `MDB_AGG_ENTRIES`
* `mdb_counted_entries()` → `mdb_agg_totals()` + read `out.mv_agg_entries`
* `mdb_counted_rank()` / `mdb_counted_select()` → `mdb_agg_rank()` / `mdb_agg_select()` with `MDB_AGG_WEIGHT_ENTRIES`
* New: `MDB_AGG_KEYS`, `MDB_AGG_HASHSUM`, window subrange anti-entropy helpers (`MDB_agg_window`, `mdb_agg_window_*`).


## Project and format versioning

AELMDB uses an independent project release line rather than reinterpreting LMDB’s upstream `MDB_VERSION_*` macros. The current release is **AELMDB 0.2.0**; the earlier implementation is retained in repository history as **AELMDB First Generation (v0.1.0 tag)**.

The source deliberately retains LMDB’s upstream `MDB_VERSION_*` values for LMDB lineage/API compatibility. AELMDB clients should identify the fork and its API release with:

```c
MDB_AELMDB_VERSION_MAJOR
MDB_AELMDB_VERSION_MINOR
MDB_AELMDB_VERSION_PATCH
MDB_AELMDB_VERSION
MDB_AELMDB_VERSION_STRING
```

For example, a consumer requiring the current API can use `#if MDB_AELMDB_VERSION < MDB_VERINT(0, 2, 0)`. AELMDB’s persistent environment format has its own internal data-version tag and also encodes the configured `MDB_HASH_SIZE`, so incompatible hash widths are rejected when an environment is opened. `MDB_AGGFORMAT_VERSION` identifies the aggregate-enabled persistent format and is intentionally separate from the project release number.

