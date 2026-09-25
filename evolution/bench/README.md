# Performance benchmarks

These benchmarks are engineering tools, not product code.  They are intended
for relative comparisons on one machine/build, not as portable absolute
performance claims.

- `bench.c`: write/read/scan/delete microbenchmark.  Build-time knobs:
  `AGGF`, `HASHOFF`, `MDB_HASH_SIZE`, and `BENCH_VALUE_SIZE`.
- `bench_query.c`: aggregate totals/prefix/range/rank/select/window query
  microbenchmark.
- `run.sh quick`: short reproducible smoke benchmark.
- `run.sh full`: broader matrix intended for a performance audit.

The runner pins each timed process to one CPU when `taskset` is available and
uses a tmpfs database root when `/dev/shm` is writable.  Report medians rather
than single runs; LMDB timings are sufficiently short that scheduler and file
system noise can otherwise dominate small differences.
