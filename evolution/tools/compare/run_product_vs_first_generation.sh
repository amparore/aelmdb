#!/bin/sh
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/../../.." && pwd)
OUT=${OUT:-$ROOT/build/evolution-compare}
CC=${CC:-cc}
CFLAGS=${CFLAGS:--std=c11 -O2 -Wall -Wextra -Wno-unused-parameter -D_GNU_SOURCE}
HASHES=${HASHES:-"8 32 64 256"}
MODES=${MODES:-"query growth stress"}
TEST="$ROOT/evolution/tools/compare/agg_compare_shared.c"
rm -rf "$OUT"; mkdir -p "$OUT"
printf 'impl\tmode\thash\tstatus\tsignature_or_failure\n' > "$OUT/results.tsv"

build_core() {
  impl=$1; h=$2; d="$OUT/core-$impl-$h"; mkdir -p "$d"
  case "$impl" in
    current)
      cp "$ROOT/src/"*.c "$ROOT/src/"*.h "$d/" ;;
    first-generation)
      cp "$ROOT/evolution/reference/aelmdb-first-generation/"*.c "$ROOT/evolution/reference/aelmdb-first-generation/"*.h "$d/" ;;
  esac
  (cd "$d" && $CC $CFLAGS -DMDB_HASH_SIZE=$h -c mdb.c -o mdb.o && $CC $CFLAGS -c midl.c -o midl.o)
}

run_one() {
  impl=$1; h=$2; mode=$3; core="$OUT/core-$impl-$h"; d="$OUT/$impl-$mode-$h"; mkdir -p "$d"
  cp "$TEST" "$d/test.c"; cp "$core/lmdb.h" "$d/"
  case "$mode" in
    query) def= ;;
    growth) def=-DAGG_COMPARE_STRESS_GROWTH=1 ;;
    stress) def=-DAGG_COMPARE_STRESS_MUTATIONS=1 ;;
    *) echo "unsupported mode $mode" >&2; exit 2 ;;
  esac
  if (cd "$d" && $CC $CFLAGS -DMDB_HASH_SIZE=$h -DTEST_IMPL_NAME=\"$impl\" $def test.c "$core/mdb.o" "$core/midl.o" -lpthread -o test); then
    set +e; result=$(cd "$d" && ./test 2>&1); rc=$?; set -e
    if [ "$rc" -eq 0 ]; then
      sig=$(printf '%s\n' "$result" | sed -n 's/.*signature=\([0-9a-fA-F]*\).*/\1/p' | tail -n 1)
      printf '%s\t%s\t%s\tPASS\t%s\n' "$impl" "$mode" "$h" "$sig" >> "$OUT/results.tsv"
    else
      printf '%s\t%s\t%s\tFAIL(%s)\t%s\n' "$impl" "$mode" "$h" "$rc" "$(printf '%s\n' "$result" | tail -1)" >> "$OUT/results.tsv"
    fi
  else
    printf '%s\t%s\t%s\tBUILD_FAIL\t-\n' "$impl" "$mode" "$h" >> "$OUT/results.tsv"
  fi
}

for h in $HASHES; do
  for impl in current first-generation; do build_core "$impl" "$h"; done
  for mode in $MODES; do
    for impl in current first-generation; do run_one "$impl" "$h" "$mode"; done
  done
done
cat "$OUT/results.tsv"
