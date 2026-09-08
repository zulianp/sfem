#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

# Experiment settings live here; environment overrides also work with sbatch.
# One resolution/mode per job is the default. Space-separated lists run sequentially.
: "${TORSION_RESOLUTIONS:=coarse}"
: "${TORSION_MODES:=per_qp}"
: "${TORSION_CASES:=fp64 fp32 fp16 fp16_tensor fp16_element_prony}"
: "${TORSION_POSTPROCESS:=1}"
# Leave empty for the YAML's T=30 s. Set explicitly for a short debug pilot.
: "${TORSION_END_TIME:=}"
: "${TORSION_PYTHON:=python3}"
: "${PRONY_EXE:=$ROOT_DIR/spikes/prony-series/build/prony_visco_torsion}"
: "${TORSION_OUT_BASE:=$ROOT_DIR/build_torsion_runs/study_${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}}"
export SFEM_HISTORY_CHECK="${SFEM_HISTORY_CHECK:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MPLBACKEND=Agg

read -r -a resolutions <<< "$TORSION_RESOLUTIONS"
read -r -a modes <<< "$TORSION_MODES"
read -r -a cases <<< "$TORSION_CASES"
if [[ ${#resolutions[@]} -eq 0 || ${#modes[@]} -eq 0 || ${#cases[@]} -eq 0 ]]; then
    echo "[error] Resolution, mode and case lists must not be empty." >&2
    exit 1
fi
for resolution in "${resolutions[@]}"; do
    case "$resolution" in coarse|medium|fine) ;; *) echo "Invalid resolution: $resolution" >&2; exit 1 ;; esac
done
for mode in "${modes[@]}"; do
    case "$mode" in per_qp|per_elem) ;; *) echo "Invalid history mode: $mode" >&2; exit 1 ;; esac
done
case "$TORSION_POSTPROCESS" in 0|1) ;; *) echo "TORSION_POSTPROCESS must be 0 or 1" >&2; exit 1 ;; esac
if [[ -e "$TORSION_OUT_BASE" ]]; then
    echo "[error] Output already exists: $TORSION_OUT_BASE; choose an unused directory." >&2
    exit 1
fi
"$TORSION_PYTHON" -c 'import sys, numpy, pandas, matplotlib, yaml; print("Python dependencies OK:", sys.executable)'
if [[ ! -x "$PRONY_EXE" ]]; then
    echo "[error] Build torsion first; executable missing or not executable: $PRONY_EXE" >&2
    exit 1
fi
# On CSCS this also catches missing runtime libraries inside the compute-node uenv.
if [[ "$(uname -s)" == Linux ]]; then
    dependencies="$(ldd "$PRONY_EXE")"
    if [[ "$dependencies" == *"not found"* ]]; then
        echo "$dependencies" >&2
        exit 1
    fi
fi
args=(--run-only)
if [[ -n "$TORSION_END_TIME" ]]; then args+=(--end-time "$TORSION_END_TIME"); fi
runner="$ROOT_DIR/scripts/run_torsion_history_compare.py"
for resolution in "${resolutions[@]}"; do
    for mode in "${modes[@]}"; do
        out="$TORSION_OUT_BASE/${resolution}_${mode}"
        echo "[study] resolution=$resolution mode=$mode cases=$TORSION_CASES out=$out"
        "$TORSION_PYTHON" "$runner" --exe "$PRONY_EXE" --out "$out" \
            --resolution "$resolution" --history-mode "$mode" --cases "${cases[@]}" \
            "${args[@]}"
        if [[ "$TORSION_POSTPROCESS" == 1 && " ${cases[*]} " == *" fp64 "* && ${#cases[@]} -gt 1 ]]; then
            "$TORSION_PYTHON" "$runner" --compare-only --out "$out"
        else
            echo "[info] Raw results retained. Compare later with --compare-only --runs ..."
        fi
    done
done
echo "[done] Study results: $TORSION_OUT_BASE"
