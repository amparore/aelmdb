#!/bin/sh
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
OUT="$ROOT/build/test-compare"
CC=${CC:-cc}
CFLAGS=${CFLAGS:--std=c11 -O2 -Wall -Wextra -Wno-unused-parameter -D_GNU_SOURCE}
HASHES=${HASHES:-"8 32 64 256"}
MODES=${MODES:-"query growth stress"}
rm -rf "$OUT"; mkdir -p "$OUT"
printf 'mode\thash\tsignature\n' > "$OUT/actual.tsv"
for h in $HASHES; do
  core="$OUT/core-$h"; mkdir -p "$core"
  cp "$ROOT/src/"*.c "$ROOT/src/"*.h "$core/"
  (cd "$core" && $CC $CFLAGS -DMDB_HASH_SIZE=$h -c mdb.c -o mdb.o && $CC $CFLAGS -c midl.c -o midl.o)
  for mode in $MODES; do
    d="$OUT/$mode-$h"; mkdir -p "$d"
    cp "$ROOT/tests/compare/aggregate_signatures.c" "$d/test.c"
    cp "$ROOT/src/lmdb.h" "$d/"
    mode_def=
    case "$mode" in
      query) mode_def= ;;
      growth) mode_def=-DAGG_COMPARE_STRESS_GROWTH=1 ;;
      stress) mode_def=-DAGG_COMPARE_STRESS_MUTATIONS=1 ;;
      *) echo "unknown mode: $mode" >&2; exit 2 ;;
    esac
    (cd "$d" && $CC $CFLAGS -DMDB_HASH_SIZE=$h -DTEST_IMPL_NAME=\"AELMDB\" $mode_def test.c "$core/mdb.o" "$core/midl.o" -lpthread -o test)
    result=$(cd "$d" && ./test)
    printf '%s\n' "$result" > "$d/run.log"
    sig=$(printf '%s\n' "$result" | sed -n 's/.*signature=\([0-9a-fA-F]*\).*/\1/p' | tail -n 1)
    test -n "$sig"
    printf '%s\t%s\t%s\n' "$mode" "$h" "$sig" >> "$OUT/actual.tsv"
  done
done
# Filter expected rows to exactly the requested matrix, preserving actual order.
{
  printf 'mode\thash\tsignature\n'
  tail -n +2 "$OUT/actual.tsv" | while read -r mode h sig; do
    awk -F '\t' -v m="$mode" -v h="$h" 'NR>1 && $1==m && $2==h {print; found=1} END{if(!found) exit 1}' "$ROOT/tests/compare/expected.tsv"
  done
} > "$OUT/expected.tsv"
diff -u "$OUT/expected.tsv" "$OUT/actual.tsv"
echo "test-compare: PASS"
