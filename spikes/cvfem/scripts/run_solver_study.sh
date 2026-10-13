#!/usr/bin/env bash
set -euo pipefail

if [[ $# != 2 ]]; then
    echo "usage: $0 <cvfem-build-directory> <new-results-directory>" >&2
    exit 2
fi
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
build_dir=$1
results_dir=$2
PY=${PYTHON:-python3}
export PYTHONPATH="$script_dir/../../../python${PYTHONPATH:+:$PYTHONPATH}"
if [[ -e "$results_dir" ]]; then
    echo "results directory already exists: $results_dir" >&2
    exit 2
fi
mkdir -p -- "$results_dir"
date -u
hostname
cmake --build "$build_dir" --target cvfem_hex8_ns_ssgmg -j "${SLURM_CPUS_PER_TASK:-4}"
binary=$(cd -- "$build_dir" && pwd)/cvfem_hex8_ns_ssgmg
gate_script=$script_dir/../src/tests/verification/cvfem_solver_iteration_gate.py
provenance_args=()
revisions_file=${SFEM_SOURCE_REVISIONS:-$script_dir/../../../../SOURCE-REVISIONS}
if [[ -f "$revisions_file" ]]; then
    provenance_args=(--provenance-file "$revisions_file")
fi
"$PY" "$gate_script" "$binary" --out "$results_dir/gate"
"$PY" "$script_dir/solver_iteration_study.py" \
    --binary "$binary" --out "$results_dir/study" "${provenance_args[@]}" \
    --sizes 4,8,12 --reynolds 1,10 --warps 0,0.025 --level 2 --threads 1 --timeout 120
