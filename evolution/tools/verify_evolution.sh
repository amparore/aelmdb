#!/bin/sh
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
TMP="$ROOT/build/evolution-replay"
rm -rf "$TMP"
mkdir -p "$TMP"
cp "$ROOT/evolution/base/mdb.c" "$ROOT/evolution/base/lmdb.h" \
   "$ROOT/evolution/base/midl.c" "$ROOT/evolution/base/midl.h" "$TMP/"
for p in 01-base 02-bottomup 03-aggformat 04-aggregates; do
    (cd "$TMP" && patch -s -p1 < "$ROOT/evolution/patches/$p.patch")
done
find "$ROOT/src" -maxdepth 1 -type f -printf '%f\n' | sort > "$TMP/src.files"
find "$TMP" -maxdepth 1 -type f ! -name 'src.files' ! -name 'replay.files' -printf '%f\n' | sort > "$TMP/replay.files"
diff -u "$TMP/src.files" "$TMP/replay.files"
while IFS= read -r f; do
    cmp "$ROOT/src/$f" "$TMP/$f"
done < "$TMP/src.files"
echo "verify-evolution: PASS (four patches reproduce src/ byte-for-byte)"
