#!/usr/bin/env bash
set -euo pipefail
probe=${1:?usage: cvfem_storage_width.sh /absolute/path/to/cvfem_storage_width_probe}
work=${WIDTH_PROBE_LOG_DIR:-$(mktemp -d)}
mkdir -p "$work"
case "$work" in /*) ;; *) work="$PWD/$work" ;; esac
case "$probe" in /*) ;; *) probe="$PWD/$probe" ;; esac
cd "$work"
n=${WIDTH_PROBE_N:-16}
pack=${WIDTH_PROBE_PACK:-64}
samples=${WIDTH_PROBE_SAMPLES:-4}
repeat=${WIDTH_PROBE_REPEAT:-2}
export LC_ALL=C OMP_DYNAMIC=false
for threads in ${WIDTH_PROBE_THREADS:-1 2}; do
    if ! OMP_NUM_THREADS=$threads "$probe" --self-test > "$work/self-T${threads}.log" 2>&1; then
        cat "$work/self-T${threads}.log"; exit 1
    fi
    if ! OMP_NUM_THREADS=$threads "$probe" --n "$n" --pack-size "$pack" \
          --samples "$samples" --repeat "$repeat" --warmup 3 > "$work/timed-T${threads}.log" 2>&1; then
        cat "$work/timed-T${threads}.log"; exit 1
    fi
    awk -F, -v expected="$samples" -v expected_threads="$threads" '
        $1=="metadata" {
            split($4,a,"="); if (a[2]+0!=expected_threads) bad=1
            metadata++
        }
        $1=="validation" {
            if ($3!="graph_bit_equal=1") bad=1
            split($4,a,"="); if (a[2]+0>2e-7 || a[2] ~ /nan|inf/) bad=1
            validations++
        }
        $1 ~ /^[0-9]+$/ && ($2=="graph" || $2=="geometry") {
            if ($1>=expected || seen[$1,$2]++ || $3!=($1%2 ? "narrow" : "wide") ||
                $4+0<=0 || $5+0<=0 || $6+0<=0 || $0 ~ /nan|inf/) bad=1
            counts[$2]++
        }
        $1=="probe_complete" && $2==1 { complete++ }
        END { if (bad || metadata!=1 || validations!=4 || complete!=1 || counts["graph"]!=expected || counts["geometry"]!=expected) exit 1 }
    ' "$work/timed-T${threads}.log"
    cat "$work/self-T${threads}.log" "$work/timed-T${threads}.log"
done
for option in --n --pack-size --samples --repeat; do
    if "$probe" "$option" 0 > "$work/reject-${option#--}.log" 2>&1; then
        echo "storage width: accepted $option 0" >&2; exit 1
    fi
    grep -q 'invalid probe size' "$work/reject-${option#--}.log"
done
printf 'storage_width_check: PASSED; logs=%s\n' "$work"
