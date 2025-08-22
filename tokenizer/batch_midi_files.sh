#!/usr/bin/env bash
set -euo pipefail

# ====== Config (override via env or args) ======
SRC_DIR="${1:-/path/to/midis}"              # input tree
OUT_DIR="${2:-/path/to/output_archives}"    # where .tar.gzs go
NUM_ARCHIVES="${NUM_ARCHIVES:-10000}"       # total archives to produce
GLOB="${GLOB:-*.mid}"                       # which files to pack
PREFIX="${PREFIX:-bundle}"                  # archive name prefix
PARALLEL_JOBS="${PARALLEL_JOBS:-$(nproc)}"  # how many archives to build at once
COMPRESS_THREADS="${COMPRESS_THREADS:-$(nproc)}"  # threads for pigz

# ====== Pre-flight ======
mkdir -p "$OUT_DIR"
case "$OUT_DIR" in "$SRC_DIR"|"$SRC_DIR"/*) echo "ERROR: OUT_DIR must not be inside SRC_DIR"; exit 1;; esac

if ! tar --version 2>/dev/null | grep -qi 'gnu tar'; then
  echo "ERROR: Requires GNU tar (for --null / --files-from)"; exit 1
fi

# Prefer pigz; fall back to gzip (single-core)
COMP="gzip"
if command -v pigz >/dev/null 2>&1; then
  COMP="pigz -p ${COMPRESS_THREADS}"
fi

echo "Scanning $SRC_DIR for '$GLOB'..."
ALL_LIST="$(mktemp)"
find "$SRC_DIR" -type f -name "$GLOB" -print0 > "$ALL_LIST"
TOTAL=$(tr -cd '\0' < "$ALL_LIST" | wc -c)
if (( TOTAL == 0 )); then
  echo "No files found."
  rm -f "$ALL_LIST"
  exit 0
fi

# Compute batch size to produce exactly NUM_ARCHIVES (or fewer if not enough files)
BATCH_SIZE=$(( (TOTAL + NUM_ARCHIVES - 1) / NUM_ARCHIVES ))
(( BATCH_SIZE < 1 )) && BATCH_SIZE=1

echo "Total files:   $TOTAL"
echo "Archives goal: $NUM_ARCHIVES"
echo "Batch size:    $BATCH_SIZE files/archive"
echo "Parallel jobs: $PARALLEL_JOBS"
echo "Compressor:    $COMP"
echo

# ====== Split file list into per-archive manifests (null-delimited) ======
# We'll generate manifest-000001.list (NUL-delimited), then process them in parallel.
MAN_DIR="$(mktemp -d)"
idx=0
count=0
cur="$MAN_DIR/manifest-$(printf '%06d' $((idx+1))).list"
: > "$cur"  # create first file

# Walk the NUL-delimited list and split
# shellcheck disable=SC2034
while IFS= read -r -d '' f; do
  printf '%s\0' "$f" >> "$cur"
  count=$((count+1))
  if (( count == BATCH_SIZE )); then
    idx=$((idx+1))
    count=0
    cur="$MAN_DIR/manifest-$(printf '%06d' $((idx+1))).list"
    : > "$cur"
  fi
done < "$ALL_LIST"
# If last manifest ended exactly at boundary, remove the empty trailing file
[[ -s "$cur" ]] || rm -f "$cur"

# ====== Build archives in parallel ======
export OUT_DIR PREFIX COMP REMOVE_FILES
build_one() {
  local manifest="$1"
  local base
  base=$(basename "$manifest" .list)   # manifest-000123
  local num="${base##*-}"              # 000123
  local archive="${OUT_DIR}/${PREFIX}-${num}.tar.gz"

  if [[ -f "$archive" ]]; then
    echo "SKIP (exists): $archive"
    return 0
  fi

  echo "Creating: $archive"
  tar -I "$COMP" -cf "$archive" --null --files-from="$manifest"
}
export -f build_one

# Prefer GNU parallel; otherwise use xargs -P
if command -v parallel >/dev/null 2>&1; then
  find "$MAN_DIR" -type f -name 'manifest-*.list' -print0 \
    | parallel --null -P "$PARALLEL_JOBS" build_one {}
else
  find "$MAN_DIR" -type f -name 'manifest-*.list' -print0 \
    | xargs -0 -I{} -P "$PARALLEL_JOBS" bash -c 'build_one "$@"' _ {}
fi

# ====== Cleanup ======
rm -f "$ALL_LIST"
rm -rf "$MAN_DIR"

# Tidy empty dirs left after deletions (only matters if REMOVE_FILES=1)
if (( REMOVE_FILES )); then
  echo "Cleaning up empty directories under $SRC_DIR..."
  find "$SRC_DIR" -type d -empty -delete || true
fi

echo "Done."
