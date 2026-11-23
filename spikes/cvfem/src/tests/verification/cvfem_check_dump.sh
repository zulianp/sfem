#!/usr/bin/env bash
# Validate the whole native-double dump before comparing or accepting equal bytes.
set -euo pipefail
file=${1:?usage: cvfem_check_dump.sh FILE EXPECTED_VALUES}
expected=${2:?expected value count is required}
[[ $expected =~ ^[1-9][0-9]*$ ]] || exit 1
if [[ $(wc -c < "$file") -ne $((8 * expected)) ]]; then
    echo "FAIL: wrong dump size: $file" >&2
    exit 1
fi
LC_ALL=C od -An -v -t f8 "$file" | LC_ALL=C awk -v expected="$expected" '
    toupper($0) ~ /NAN|INF/ { bad=1 }
    { for (i=1; i<=NF; ++i) { print $i; count++ } }
    END { exit bad || count != expected }'
