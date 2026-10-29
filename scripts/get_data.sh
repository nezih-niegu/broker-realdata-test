#!/usr/bin/env bash
# Fetch Binance tick files from github.com/Nucs/cryptocurrency-ticks-data (Unlicense) without
# downloading the whole 5 GB repository: a blob-less shallow clone, then checkout of only the
# requested files.
#
# Usage: scripts/get_data.sh [SYMBOL] [DATE_REGEX]
#   default: BTCUSDT and 2019-01-01 .. 2019-01-20 (the files used in RESULTS.md)
# Note: the file labelled 2019-01-01 starts at 2018-12-31 00:00 UTC; the source README warns
#       about timestamp offsets. Nothing in the test depends on calendar time.
# Do NOT run `git ls-tree -l` in the clone: listing sizes makes git download every file.
set -euo pipefail
SYMBOL="${1:-BTCUSDT}"
REGEX="${2:-2019-01-(0[1-9]|1[0-9]|20)}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TMP="${ROOT}/.cache/nucs"
mkdir -p "${ROOT}/data/${SYMBOL}" "$(dirname "$TMP")"
if [ ! -d "$TMP/.git" ]; then
  GIT_LFS_SKIP_SMUDGE=1 git -c gc.auto=0 clone -q --depth 1 --filter=blob:none --no-checkout \
    https://github.com/Nucs/cryptocurrency-ticks-data "$TMP"
fi
cd "$TMP"
FILES=$(git -c gc.auto=0 ls-tree -r --name-only HEAD | grep -E "data/${SYMBOL}/${SYMBOL}\.${REGEX}\.csv\.zip" || true)
if [ -z "$FILES" ]; then echo "no files match ${SYMBOL} ${REGEX}"; exit 1; fi
git -c gc.auto=0 checkout HEAD -- $FILES
cp $FILES "${ROOT}/data/${SYMBOL}/"
echo "copied $(echo "$FILES" | wc -l) files to data/${SYMBOL}/"
