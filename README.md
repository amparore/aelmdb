# **AELMDB**: Anti-Entropy Lightning Memory-Mapped Database

**AELMDB (Anti-Entropy LMDB)** is a fork of **[LMDB](https://github.com/LMDB/lmdb)** that augments the B+tree with **aggregate metadata stored in branch pages**. In practice, this turns LMDB’s ordered keyspace into something you can query by **position** and **summarize by range** without scanning leaf pages.

This enables:
- **Order-statistics** on the database order (fast `rank` / `select`), extending the [order statistics-B-tree](https://en.wikipedia.org/wiki/Order_statistic_tree) design used in **[DLMDB](https://github.com/datalevin/dlmdb)**
- **Fast range summaries** (counts + fixed-size **hash sums**) that are composable and efficient for large datasets
- **Anti-entropy / reconciliation-friendly primitives**, specifically targeting **Range-Based Set Reconciliation ([RBSR](https://logperiodic.com/rbsr.html))**, where you repeatedly compare and split ordered ranges using compact aggregates



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

### New DBI flags (aggregate schema)

These flags (passed to `mdb_dbi_open()`) select which aggregate components are maintained in branch pages:

* `MDB_AGG_ENTRIES`: counts logical **records** (key/value pairs). For `MDB_DUPSORT`, each duplicate counts as one record.
* `MDB_AGG_KEYS`: counts **distinct keys** in the primary tree.
* `MDB_AGG_HASHSUM`: maintain a fixed-size wraparound accumulator of _hash slices_ from each entry.
* `MDB_AGG_HASHSOURCE_FROM_KEY`: uses the **key bytes** as the hash source instead of value bytes. 
  * Only meaningful with `MDB_AGG_HASHSUM`, and **incompatible with `MDB_DUPSORT`**.



## Hash slice definition

When `MDB_AGG_HASHSUM` is enabled, AELMDB assumes that each record (entry) contains a fixed-size **hash slice**—a contiguous byte window that will be summed into the range fingerprint.
For every entry, AELMDB extracts exactly **`MDB_HASH_SIZE` bytes**:

* **From where (hash source):**
  * by default, from the entry’s **value** bytes
  * if `MDB_AGG_HASHSOURCE_FROM_KEY` is set, from the entry’s **key** bytes instead
* **From which position:** the DBI’s configured **`hash_offset`** (set once via `mdb_set_hash_offset()` on an empty DBI) selects the **start** of the `MDB_HASH_SIZE` slice:

  * `hash_offset >= 0`: start at `hash_offset` bytes from the beginning
  * `hash_offset < 0`: start from the end, where `-1` means “use the last `MDB_HASH_SIZE` bytes” (and `-k` means `(k-1)` bytes earlier)

The extracted slice is `source[start .. start + MDB_HASH_SIZE)` where `source` is `value` (default) or `key` (when `MDB_AGG_HASHSOURCE_FROM_KEY` is set).
`MDB_HASH_SIZE` defaults to **32 bytes** and is currently required to be a **multiple of 8**.


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
This DBI maintains entry counts, distinct-key counts, and a hashsum fingerprint from the defined hash slice.

```c
unsigned flags =
  MDB_CREATE |
  MDB_AGG_ENTRIES |   /* maintain entry (record) counts */
  MDB_AGG_KEYS    |   /* maintain distinct-key counts */
  MDB_AGG_HASHSUM;    /* maintain hashsum fingerprint */

MDB_dbi dbi;
mdb_dbi_open(txn, "plain_agg", flags, &dbi);

`/* hash_offset: 0 = first bytes, -1 = last bytes */`
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
### Hashsum wraparound algebra (helper layer)

AELMDB’s hashsum is a fixed-size **`MDB_HASH_SIZE`-byte accumulator**. 
Conceptually it behaves like a single unsigned integer and all operations are done **modulo `2^(8*MDB_HASH_SIZE)`** (i.e. classical wraparound arithmetic). 

The header provides small inline helpers implemented on 64-bit unsigned integers:

```c
/* acc += x (mod 2^(8*MDB_HASH_SIZE)) */
static inline void mdb_hashsum_add(uint8_t *acc, const uint8_t *x);

/* acc -= x (mod 2^(8*MDB_HASH_SIZE)) */
static inline void mdb_hashsum_sub(uint8_t *acc, const uint8_t *x);

/* out = a - b (mod 2^(8*MDB_HASH_SIZE)) */
static inline void mdb_hashsum_diff(uint8_t *out, const uint8_t *a, const uint8_t *b);

/* true iff buffer is all zeros */
static inline int  mdb_hashsum_is_zero(const uint8_t *p);
```

Since the hashsum is defined as “sum of the MDB_HASH_SIZE-byte slice selected by hash_offset”, the header provides bounds-checked slice helpers:
`mdb_hashsum_slice_ptr_bytes`/`mdb_hashsum_slice_ptr`(pointer) and `mdb_hashsum_extract_bytes`/`mdb_hashsum_extract` (copy).









<br/>

# 2) Aggregate API (totals / prefix / range)

The Aggregate API provides **fast summaries over sets of records** (entire DB, a prefix, or a bounded range) using the aggregate metadata stored in branch pages. Each function returns an `MDB_agg` structure containing the components enabled for that DBI (entries, keys, and/or hashsum). If a DBI wasn’t created with the required aggregate components, these functions return `MDB_INCOMPATIBLE`.

## Result type

```c
/* Aggregate result (returned by totals/prefix/range queries) */
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
/* Compute aggregates over the entire DBI */
int mdb_agg_totals(
  MDB_txn  *txn,   // IN: transaction handle
  MDB_dbi   dbi,   // IN: target DBI
  MDB_agg  *out    // OUT: totals for the whole DBI (mv_flags set to schema)
);
```

Returns the **full-database totals** in `*out`—the quickest way to obtain overall entry count / key count / full fingerprint.



### Prefix aggregate

The **prefix aggregate** API computes aggregate totals over the *prefix* of the DBI’s ordered records: i.e., **all records that come before a given pivot** in the DB’s total order.
It returns those totals in `MDB_agg` (entries/keys/hashsum as enabled), letting you get “counts and fingerprint up to here” without scanning.


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
  MDB_val        *data,      // IN: optional query value; OUT: located value (SET_RANGE mode)
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
This is the “jump then iterate” primitive: you seek by rank once, then use normal cursor iteration (`MDB_NEXT`, etc.) efficiently from that point.

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


AELMDB’s **cached window APIs** are built specifically for this. 
They cache the expensive part, mapping a key-range window to an **absolute entry-rank interval**, and then let a program cheaply:

1. compute a range **aggregate** (an `MDB_agg` summary, typically using `HASHSUM`) for any **relative entry-rank subrange** inside the window, and
2. compute a **window-relative lower-bound rank** for a key (and optionally value for `MDB_DUPSORT`) without re-deriving the window mapping each time.


This is the exact pattern used by Negentropy-style reconciliation loops: many aggregate calls + many lower-bound calls, all within stable outer bounds. (see [negentropy](https://github.com/hoytech/negentropy))

See also the AELMDB storage in Negentropy for Range-Based Set Reconciliation, named [AELMDBSlice](https://github.com/amparore/negentropy-aelmdb), and the extended C++ wrapper [lmdbxx](https://github.com/amparore/lmdbxx-aelmdb).



## Cached window descriptor

`MDB_agg_window` is the cached “outer range mapping” that makes repeated subrange queries cheap. You initialize it to zero, then reuse it for subsequent queries as long as the bounds/flags stay the same.

```c
#define MDB_AGG_WINDOW_END UINT64_MAX // sentinel: rel_end means "use the full window end"

/* Cached mapping from a key-range window to an absolute entry-rank interval */
typedef struct MDB_agg_window {
  unsigned mv_flags;         // OUT: DBI aggregate schema (MDB_AGG_* bits) cached for this window
  unsigned mv_range_flags;   // OUT: MDB_RANGE_* flags defining inclusion/exclusion of window bounds
  uint64_t mv_total_entries; // OUT: cached total entries in DBI (used for open-ended/clamping cases)
  uint64_t mv_abs_lo;        // OUT: absolute entry-rank of the window's lower bound
  uint64_t mv_abs_hi;        // OUT: absolute entry-rank of the window's upper bound
} MDB_agg_window;
```

**Rationale:** reconciliation repeatedly queries many subranges within the same outer bounds; recomputing the outer bounds’ absolute entry-ranks every time wastes work. `MDB_agg_window` caches that mapping`(low_key, high_key, range_flags) → [mv_abs_lo, mv_abs_hi)`, so subsequent aggregates and lower-bound queries can run in **window-relative rank coordinates**.

**Usage rules:** zero-initialize before first use; reuse only with the same bounds and `range_flags`.



## Cached window functions 

### `mdb_agg_window_aggregate()`

Computes fingeaggregates for a **relative entry-rank subrange** inside a cached window.
If the cache is empty or the bounds changed, it (re)computes the window’s absolute rank interval, then answers subrange queries efficiently.

```c
int mdb_agg_window_aggregate(
  MDB_txn        *txn,         // IN: transaction handle
  MDB_dbi         dbi,         // IN: target DBI
  const MDB_val  *low_key,     // IN: optional window lower key (NULL => open-ended)
  const MDB_val  *low_data,    // IN: optional window lower value (record-order bound for MDB_DUPSORT)
  const MDB_val  *high_key,    // IN: optional window upper key (NULL => open-ended)
  const MDB_val  *high_data,   // IN: optional window upper value (record-order bound for MDB_DUPSORT)
  unsigned        range_flags, // IN: MDB_RANGE_* controlling inclusion/exclusion of window bounds
  MDB_agg_window *window,      // IN/OUT: cached window descriptor (must be zeroed initially)
  uint64_t        rel_begin,   // IN: relative begin entry-rank within window
  uint64_t        rel_end,     // IN: relative end entry-rank within window (or MDB_AGG_WINDOW_END)
  MDB_agg        *out          // OUT: aggregates over [rel_begin, rel_end) within the window
);
```

Window bounds follow the same inclusion/exclusion rules as `mdb_agg_range()`. 
The subrange itself is expressed in **relative entry-rank space** within the window.


### `mdb_agg_window_rank()`

Computes the **lower-bound position** of a key (and optional value for `MDB_DUPSORT`) **relative to the cached window**.
This is typically used to place split points and to map protocol “cursor keys” into window-relative ranks during reconciliation.

```c
int mdb_agg_window_rank(
  MDB_txn        *txn,         // IN: transaction handle
  MDB_dbi         dbi,         // IN: target DBI
  const MDB_val  *low_key,     // IN: window lower key (must match cached window)
  const MDB_val  *low_data,    // IN: window lower value
  const MDB_val  *high_key,    // IN: window upper key
  const MDB_val  *high_data,   // IN: window upper value
  unsigned        range_flags, // IN: MDB_RANGE_* (must match cached window)
  MDB_agg_window *window,      // IN/OUT: cached window descriptor
  const MDB_val  *key,         // IN: query key (lower-bound search)
  const MDB_val  *data,        // IN: optional query value (record-order for MDB_DUPSORT)
  uint64_t       *rel_rank     // OUT: relative rank within window [0, window_size]
);
```

It finds the first record ≥ `(key,data)` in the DBI’s total order, clamps that absolute rank into `[mv_abs_lo, mv_abs_hi)`, and returns the resulting **window-relative** rank.







<br/>

# Debug & validation support

Because AELMDB maintains extra on-page metadata (counts + hashsums), it also adds extensive **opt-in debugging hooks** to detect aggregate drift early and make stress tests fail fast. These checks are intentionally expensive and are meant for development and validation rather than production.

### `MDB_AGG_CHECK` (environment flag)

`MDB_AGG_CHECK` is an **environment flag** that enables *extra aggregate integrity verification after write operations*. 

```c
mdb_env_set_flags(env, MDB_AGG_CHECK, 1);   /* enable expensive post-write agg checks */
mdb_env_set_flags(env, MDB_AGG_CHECK, 0);   /* disable */
```

### `MDB_DEBUG_AGG_INTEGRITY` (preprocessor macro) and `mdb_dbg_check_agg_db()`

When AELMDB is compiled with `MDB_DEBUG_AGG_INTEGRITY`, it exposes a dedicated API to **explicitly verify aggregate consistency** for a DBI by performing a linear scan of the entire database and checking that aggregates are consistent:

```c
#ifdef MDB_DEBUG_AGG_INTEGRITY
int mdb_dbg_check_agg_db(MDB_txn *txn, MDB_dbi dbi);
#endif
```

This is intended for unit tests that want to run a full “check aggregates now” pass on demand. 
This macro also activates sevral (slow) integrity checks.
A second macro `MDB_DEBUG_AGG_PRINT` adds important debug prints of the internal steps, again for debug purposes.







<br/>

# 5) On-disk format and compatibility

AELMDB’s aggregate features change how **branch pages are laid out on disk** by adding per-subtree aggregate metadata. Because of that, aggregate support is **not a runtime toggle**, it must be part of the DBI’s definition from the start.

Practical implications:

* **Enable aggregate flags at DBI creation time.** The DBI’s aggregate schema (`MDB_AGG_*`) is stored/validated against page headers; mismatches are treated as corruption.
* **Changing the aggregate schema after data exists generally requires rebuilding.** If a DBI already has branch pages, switching aggregate flags typically means copying all data into a fresh DBI created with the desired flags. 
* ⚠️**Important! The AELMDB data format is different and binary incompatible with that of both LMDB and DLMDB.**



## Relationship to the counted-DB API of DLMDB

AELMDB takes inspiration (and some design ideas) from [DLMDB](https://github.com/datalevin/dlmdb/tree/main) (Datalevin’s LMDB fork), which popularized *counted B-tree* order-statistics and includes improvements such as more robust interrupt handling. 
DLMDB-style *counted B-tree* functionality (as reflected in the original header you attached) provides:

* a single `MDB_COUNTED` DBI flag, and
* `mdb_counted_*` APIs focused on **record-count order-statistics** (entries, rank, select). 

AELMDB generalizes this idea into a richer, reconciliation-oriented aggregate layer:

* **Multiple aggregate components**: `MDB_AGG_ENTRIES` (records), `MDB_AGG_KEYS` (distinct keys), `MDB_AGG_HASHSUM` (range aggregate accumulator).
* **Richer aggregate queries** beyond “count”: prefix and range aggregations returning a full `MDB_agg` summary (counts + aggregate where enabled).
* **Window-cached helpers** designed for **Range-Based Set Reconciliation** loops, where you repeatedly compute aggregates and lower-bound ranks inside stable outer bounds.

A simple conceptual mapping:

* `MDB_COUNTED` → `MDB_AGG_ENTRIES`
* `mdb_counted_entries()` → `mdb_agg_totals()` + read `out.mv_agg_entries`
* `mdb_counted_rank()` / `mdb_counted_select()` → `mdb_agg_rank()` / `mdb_agg_select()` with `MDB_AGG_WEIGHT_ENTRIES`
* New: `MDB_AGG_KEYS`, `MDB_AGG_HASHSUM`, window-cached anti-entropy helpers (`MDB_agg_window`, `mdb_agg_window_*`).


## Compile-time detection

The `lmdb.h` header of AELMDB defines the macro `MDB_AELMDB_VERSION` so applications can **detect AELMDB at compile time** and conditionally enable AELMDB-specific features (distinguishing it from upstream LMDB and other forks).
