#!/bin/sh
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
OUT="$ROOT/build/tests"
CC=${CC:-cc}
CFLAGS=${CFLAGS:--std=c11 -O2 -Wall -Wextra -Wno-unused-parameter -D_GNU_SOURCE}
HASHES=${HASHES:-32}
EXTRA_CFLAGS=${EXTRA_CFLAGS:-}
CORE_TAG=${CORE_TAG:-core}
mkdir -p "$OUT"

stage_product() {
  dir=$1
  rm -rf "$dir"; mkdir -p "$dir"
  cp "$ROOT/src/"*.c "$ROOT/src/"*.h "$dir/"
}

ensure_core() {
  h=$1; core="$OUT/$CORE_TAG-h$h"
  if [ ! -f "$core/mdb.o" ]; then
    stage_product "$core"
    (cd "$core" && $CC $CFLAGS $EXTRA_CFLAGS -DMDB_HASH_SIZE=$h -DMDB_DEBUG_AGG_INTEGRITY=1 -DMDB_DEBUG_UNWIND=1 -c mdb.c -o mdb.o && $CC $CFLAGS $EXTRA_CFLAGS -c midl.c -o midl.o)
  fi
}

run_api_test() {
  src=$1; name=$2; h=${3:-32}; d="$OUT/$name-h$h"; core="$OUT/$CORE_TAG-h$h"
  ensure_core "$h"
  rm -rf "$d"; mkdir -p "$d/run"
  cp "$src" "$d/test.c"; cp "$ROOT/src/lmdb.h" "$d/"
  [ -f "$ROOT/tests/api/agg_test_compat.h" ] && cp "$ROOT/tests/api/agg_test_compat.h" "$d/" || true
  (cd "$d" && $CC $CFLAGS $EXTRA_CFLAGS -DMDB_HASH_SIZE=$h -DMDB_DEBUG_AGG_INTEGRITY=1 test.c "$core/mdb.o" "$core/midl.o" -lpthread -o test && cd run && ../test)
}

case ${1:-all} in
  semantics)
    for h in $HASHES; do
      for t in aggregate_algebra local_fold split_maintenance rebalance_maintenance dupsort_maintenance; do
        d="$OUT/sem-$t-h$h"; stage_product "$d"; cp "$ROOT/tests/semantics/$t.c" "$d/test.c"; cp "$ROOT/src/mdb.c" "$d/aelmdb_mdb.c"
        (cd "$d" && $CC $CFLAGS $EXTRA_CFLAGS -Wno-unused-function -DMDB_HASH_SIZE=$h -DMDB_DEBUG_AGG_INTEGRITY=1 -DMDB_DEBUG_UNWIND=1 test.c midl.c -lpthread -o test && mkdir -p run && cd run && ../test)
      done
      d="$OUT/sem-integration-h$h"; stage_product "$d"; cp "$ROOT/tests/semantics/integration_maintenance.c" "$d/test.c"; cp "$ROOT/tests/semantics/dupsort_maintenance.c" "$d/test_dupsort_maintenance.c"; cp "$ROOT/src/mdb.c" "$d/aelmdb_mdb.c"
      (cd "$d" && $CC $CFLAGS $EXTRA_CFLAGS -Wno-unused-function -DMDB_HASH_SIZE=$h -DMDB_DEBUG_AGG_INTEGRITY=1 -DMDB_DEBUG_UNWIND=1 test.c midl.c -lpthread -o test && mkdir -p run && cd run && ../test)
      for t in aggregate_queries query_absent current_subpage appenddup_order dupset_bounds; do run_api_test "$ROOT/tests/semantics/$t.c" "sem-$t" "$h"; done
    done
    ;;
  api)
    for h in $HASHES; do
      for t in mtest_unit mtest_unit_keyhash mtest_adv; do run_api_test "$ROOT/tests/api/$t.c" "api-$t" "$h"; done
    done
    ;;
  format)
    # These are layer-contract tests: run them on the reconstructed 03 state,
    # exactly where aggregate representation exists without maintenance.
    for h in $HASHES; do
      for t in schema opaque_prefix; do
        d="$OUT/format-$t-h$h"; rm -rf "$d"; mkdir -p "$d"
        cp "$ROOT/evolution/base/mdb.c" "$ROOT/evolution/base/lmdb.h" "$ROOT/evolution/base/midl.c" "$ROOT/evolution/base/midl.h" "$d/"
        for p in 01-base 02-bottomup 03-aggformat; do (cd "$d" && patch -s -p1 < "$ROOT/evolution/patches/$p.patch"); done
        cp "$ROOT/tests/format/$t.c" "$d/test.c"; cp "$ROOT/tests/format/f_file.h" "$d/"
        (cd "$d" && $CC $CFLAGS $EXTRA_CFLAGS -Wno-unused-function -DMDB_HASH_SIZE=$h test.c midl.c -lpthread -o test && mkdir -p run && cd run && ../test)
      done
    done
    ;;
  regress)
    for t in even_key_padding cursor_sync cursor_hazards key_aliasing empty_value; do run_api_test "$ROOT/tests/regress/$t.c" "regress-$t" 32; done
    ;;
  lmdb)
    for t in "$ROOT"/tests/lmdb/mtest*.c; do n=$(basename "$t" .c); run_api_test "$t" "lmdb-$n" 32; done
    ;;
  differential)
    for t in rebalance_stress large_keys operation_stream; do run_api_test "$ROOT/tests/differential/$t.c" "diff-$t" 32; done
    ;;
  compare)
    exec "$ROOT/tests/compare/run_fixed_signatures.sh"
    ;;
  generic-hash)
    CORE_TAG=generic HASHES="7 13 31" EXTRA_CFLAGS="-DMDB_AGG_GENERIC_HASH_ARITH=1" "$0" semantics
    ;;
  sanitize)
    SAN_FLAGS="-O1 -g -fno-omit-frame-pointer -fsanitize=address,undefined"
    export ASAN_OPTIONS=${ASAN_OPTIONS:-detect_leaks=0}
    export UBSAN_OPTIONS=${UBSAN_OPTIONS:-print_stacktrace=1:halt_on_error=1}
    for h in $HASHES; do
      for t in split_maintenance rebalance_maintenance; do
        d="$OUT/san-$t-h$h"; stage_product "$d"; cp "$ROOT/tests/semantics/$t.c" "$d/test.c"; cp "$ROOT/src/mdb.c" "$d/aelmdb_mdb.c"
        (cd "$d" && $CC $CFLAGS $EXTRA_CFLAGS $SAN_FLAGS -Wno-unused-function -DMDB_HASH_SIZE=$h -DMDB_DEBUG_AGG_INTEGRITY=1 -DMDB_DEBUG_UNWIND=1 test.c midl.c -lpthread -o test && mkdir -p run && cd run && ../test)
      done
      CORE_TAG=sanitize EXTRA_CFLAGS="$EXTRA_CFLAGS $SAN_FLAGS" run_api_test "$ROOT/tests/semantics/aggregate_queries.c" "san-aggregate_queries" "$h"
      CORE_TAG=sanitize EXTRA_CFLAGS="$EXTRA_CFLAGS $SAN_FLAGS" run_api_test "$ROOT/tests/semantics/current_subpage.c" "san-current_subpage" "$h"
      CORE_TAG=sanitize EXTRA_CFLAGS="$EXTRA_CFLAGS $SAN_FLAGS" run_api_test "$ROOT/tests/semantics/dupset_bounds.c" "san-dupset_bounds" "$h"
      CORE_TAG=sanitize EXTRA_CFLAGS="$EXTRA_CFLAGS $SAN_FLAGS" run_api_test "$ROOT/tests/regress/empty_value.c" "san-empty_value" "$h"
    done
    ;;
  all)
    "$0" semantics
    "$0" api
    "$0" format
    "$0" regress
    "$0" compare
    ;;
  *) echo "usage: $0 {semantics|api|format|regress|lmdb|differential|compare|generic-hash|sanitize|all}" >&2; exit 2 ;;
esac
