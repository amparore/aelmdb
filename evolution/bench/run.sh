#!/bin/sh
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
MODE=${1:-quick}
CC=${CC:-cc}
CFLAGS=${CFLAGS:--std=c11 -O3 -DNDEBUG}
REPS=${REPS:-3}
BUILD="$ROOT/build/perf"
DBROOT=${DBROOT:-/dev/shm/aelmdb-perf}
[ -d /dev/shm ] && [ -w /dev/shm ] || DBROOT="$BUILD/db"
TASKSET=""
if command -v taskset >/dev/null 2>&1; then TASKSET="taskset -c ${BENCH_CPU:-0}"; fi
rm -rf "$BUILD" "$DBROOT"
mkdir -p "$BUILD" "$DBROOT"

stage_copy() { mkdir -p "$BUILD/$1"; cp "$ROOT/evolution/base/mdb.c" "$ROOT/evolution/base/lmdb.h" "$ROOT/evolution/base/midl.c" "$ROOT/evolution/base/midl.h" "$BUILD/$1/"; }
stage_copy base
cp -R "$BUILD/base" "$BUILD/stage01"; (cd "$BUILD/stage01" && patch -s -p1 < "$ROOT/evolution/patches/01-base.patch")
cp -R "$BUILD/stage01" "$BUILD/stage02"; (cd "$BUILD/stage02" && patch -s -p1 < "$ROOT/evolution/patches/02-bottomup.patch")
cp -R "$BUILD/stage02" "$BUILD/stage03"; (cd "$BUILD/stage03" && patch -s -p1 < "$ROOT/evolution/patches/03-aggformat.patch")
mkdir -p "$BUILD/current" "$BUILD/legacy"
cp "$ROOT/src/"* "$BUILD/current/"
cp "$ROOT/evolution/reference/aelmdb-first-generation/mdb.c" "$ROOT/evolution/reference/aelmdb-first-generation/lmdb.h" "$ROOT/evolution/reference/aelmdb-first-generation/midl.c" "$ROOT/evolution/reference/aelmdb-first-generation/midl.h" "$BUILD/legacy/"
compile_impl() {
  d=$1; h=${2:-32}; extra=${3:-}
  $CC $CFLAGS -DMDB_HASH_SIZE=$h $extra -I"$d" -c "$d/mdb.c" -o "$d/mdb.o"
  $CC $CFLAGS -I"$d" -c "$d/midl.c" -o "$d/midl.o"
}
for s in base stage01 stage02 stage03 current legacy; do compile_impl "$BUILD/$s" 32; done
compile_rw() {
  name=$1; d=$2; flags=$3; hashoff=$4; vsize=${5:-48}
  $CC $CFLAGS -DMDB_HASH_SIZE=32 -DAGGF=$flags -DHASHOFF=$hashoff -DBENCH_VALUE_SIZE=$vsize -I"$d" \
    "$ROOT/evolution/bench/bench.c" "$d/mdb.o" "$d/midl.o" -lpthread -o "$BUILD/$name"
}
for s in base stage01 stage02 stage03 current; do compile_rw "${s}_plain" "$BUILD/$s" 0 0; done
compile_rw current_all "$BUILD/current" 0x380 1
compile_rw legacy_all "$BUILD/legacy" 0x380 1

N=${BENCH_N:-100000}; DN=${BENCH_DUP_N:-30000}; D=${BENCH_DUPS:-8}
[ "$MODE" = full ] && N=${BENCH_N:-300000} && DN=${BENCH_DUP_N:-80000}
TSV="$BUILD/rw.tsv"
printf 'workload\tvariant\trep\tput\tget\tscan\tdel\n' > "$TSV"
run_rw() {
  wl=$1; v=$2; r=$3; n=$4; d=$5; db="$DBROOT/${wl}_${v}_${r}"
  line=$($TASKSET "$BUILD/$v" "$n" "$d" "$db")
  vals=$(printf '%s\n' "$line" | sed -E 's/put=([0-9.]+) get=([0-9.]+) scan=([0-9.]+) del=([0-9.]+)/\1\t\2\t\3\t\4/')
  printf '%s\t%s\t%s\t%s\n' "$wl" "$v" "$r" "$vals" >> "$TSV"
  rm -rf "$db"
}
variants="base_plain stage01_plain stage02_plain stage03_plain current_plain current_all legacy_all"
r=1; while [ $r -le "$REPS" ]; do for v in $variants; do run_rw plain "$v" "$r" "$N" 0; done; r=$((r+1)); done
if [ "$MODE" = full ]; then
  r=1; while [ $r -le "$REPS" ]; do for v in $variants; do run_rw dup8 "$v" "$r" "$DN" "$D"; done; r=$((r+1)); done
fi

# Query comparison at H=32.
for s in current legacy; do
  $CC $CFLAGS -DMDB_HASH_SIZE=32 -I"$BUILD/$s" "$ROOT/evolution/bench/bench_query.c" \
    "$BUILD/$s/mdb.o" "$BUILD/$s/midl.o" -lpthread -o "$BUILD/query_$s"
done
QTSV="$BUILD/query.tsv"
printf 'variant\trep\toutput\n' > "$QTSV"
r=1; while [ $r -le "$REPS" ]; do for s in current legacy; do db="$DBROOT/q_${s}_$r"; out=$($TASKSET "$BUILD/query_$s" "${BENCH_QUERY_N:-100000}" "${BENCH_QUERIES:-50000}" "$db"); printf '%s\t%s\t%s\n' "$s" "$r" "$out" >> "$QTSV"; rm -rf "$db"; done; r=$((r+1)); done

if [ "$MODE" = full ]; then
  # Aggregate-component cost at H=32.
  for spec in entries:0x80:0 keys:0x180:0 hash:0x280:1; do
    name=$(printf '%s' "$spec" | cut -d: -f1); flags=$(printf '%s' "$spec" | cut -d: -f2); hoff=$(printf '%s' "$spec" | cut -d: -f3)
    compile_rw "current_$name" "$BUILD/current" "$flags" "$hoff"
  done
  FTSV="$BUILD/flags.tsv"; printf 'workload\tvariant\trep\tput\tget\tscan\tdel\n' > "$FTSV"
  oldtsv=$TSV; TSV=$FTSV
  r=1; while [ $r -le "$REPS" ]; do for v in current_plain current_entries current_keys current_hash current_all; do run_rw plain "$v" "$r" "$N" 0; run_rw dup8 "$v" "$r" "$DN" "$D"; done; r=$((r+1)); done
  TSV=$oldtsv

  # Optimized limbs versus generic byte arithmetic, same 32-byte schema.
  mkdir -p "$BUILD/generic32"; cp "$ROOT/src/"* "$BUILD/generic32/"
  compile_impl "$BUILD/generic32" 32 -DMDB_AGG_GENERIC_HASH_ARITH=1
  compile_rw current_all_generic "$BUILD/generic32" 0x380 1
  BTSV="$BUILD/hash-backend.tsv"; printf 'workload\tbackend\trep\tput\tget\tscan\tdel\n' > "$BTSV"
  oldtsv=$TSV; TSV=$BTSV
  r=1; while [ $r -le "$REPS" ]; do
    run_rw plain current_all "$r" "$N" 0; sed -i '$s/current_all/optimized/' "$BTSV"
    run_rw plain current_all_generic "$r" "$N" 0; sed -i '$s/current_all_generic/generic/' "$BTSV"
    run_rw dup8 current_all "$r" "$DN" "$D"; sed -i '$s/current_all/optimized/' "$BTSV"
    run_rw dup8 current_all_generic "$r" "$DN" "$D"; sed -i '$s/current_all_generic/generic/' "$BTSV"
    r=$((r+1))
  done
  TSV=$oldtsv

  # Width scaling uses a fixed 256-byte value so all widths hash the same payload size.
  WTSV="$BUILD/hash-width.tsv"; printf 'workload\thash\trep\tput\tget\tscan\tdel\n' > "$WTSV"
  for h in 8 32 64 256; do
    d="$BUILD/h$h"; mkdir -p "$d"; cp "$ROOT/src/"* "$d/"; compile_impl "$d" "$h"
    $CC $CFLAGS -DMDB_HASH_SIZE=$h -DAGGF=0x380 -DHASHOFF=1 -DBENCH_VALUE_SIZE=256 -I"$d" \
      "$ROOT/evolution/bench/bench.c" "$d/mdb.o" "$d/midl.o" -lpthread -o "$BUILD/all_h$h"
  done
  r=1; while [ $r -le "$REPS" ]; do for h in 8 32 64 256; do
    for wl in plain dup8; do if [ "$wl" = plain ]; then nn=$N; dd=0; else nn=$DN; dd=$D; fi; db="$DBROOT/w_${wl}_${h}_${r}"; line=$($TASKSET "$BUILD/all_h$h" "$nn" "$dd" "$db"); vals=$(printf '%s\n' "$line" | sed -E 's/put=([0-9.]+) get=([0-9.]+) scan=([0-9.]+) del=([0-9.]+)/\1\t\2\t\3\t\4/'); printf '%s\t%s\t%s\t%s\n' "$wl" "$h" "$r" "$vals" >> "$WTSV"; rm -rf "$db"; done
  done; r=$((r+1)); done

  # DUPSORT cardinality scaling: fixed number of primary keys.
  DTSV="$BUILD/dupsort-scale.tsv"; printf 'variant\tdups\trep\tkeys\tput\tget\tscan\tdel\n' > "$DTSV"
  SCALE_KEYS=${BENCH_SCALE_KEYS:-30000}
  r=1; while [ $r -le "$REPS" ]; do for dd in 1 2 4 8 16 32; do for v in current_plain current_all; do db="$DBROOT/s_${v}_${dd}_${r}"; line=$($TASKSET "$BUILD/$v" "$SCALE_KEYS" "$dd" "$db"); vals=$(printf '%s\n' "$line" | sed -E 's/put=([0-9.]+) get=([0-9.]+) scan=([0-9.]+) del=([0-9.]+)/\1\t\2\t\3\t\4/'); printf '%s\t%s\t%s\t%s\t%s\n' "$v" "$dd" "$r" "$SCALE_KEYS" "$vals" >> "$DTSV"; rm -rf "$db"; done; done; r=$((r+1)); done
fi

printf 'performance results: %s\n' "$BUILD"
cat "$TSV"
cat "$QTSV"
