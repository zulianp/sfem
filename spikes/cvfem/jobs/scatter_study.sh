#!/usr/bin/env bash
# Run inside one CPU allocation; keep raw paired samples and their provenance.
set -euo pipefail
bench=${1:?usage: scatter_study.sh /absolute/path/to/bench /new/output/directory}
out=${2:?missing output directory}
bench=$(cd "$(dirname "$bench")" && pwd)/$(basename "$bench")
mkdir "$out"
out=$(cd "$out" && pwd)
cd "$out"
export OMP_PROC_BIND=close OMP_PLACES=cores OMP_DYNAMIC=FALSE
date -u > "$out/environment.txt"
hostname >> "$out/environment.txt"
env | sort | grep -E '^(OMP_|SLURM_(JOB_ID|CPUS_PER_TASK|CPU_BIND|JOB_NODELIST)=)' >> "$out/environment.txt"
sha256sum "$bench" > "$out/binary.sha256"
common=(--n "${N:-128}" --pack-size "${PACK:-1024}" --repeat "${REPEAT:-10}" --warmup 3
        --scatter-study --scatter-samples "${SAMPLES:-12}" --rhie-chow)
for threads in ${SCATTER_THREADS:-1 72}; do
    export OMP_NUM_THREADS=$threads
    for scheme in fo darwish; do
        layouts=(packed atomic)
        if [[ $scheme == darwish ]]; then layouts=(atomic packed); fi
        for layout in "${layouts[@]}"; do
            case_args=("${common[@]}" --layout "$layout")
            [[ $scheme == fo ]] || case_args+=(--conv-ho 3)
            log="$out/$layout-$scheme-t$threads.log"
            printf '%q ' "$bench" "${case_args[@]}" > "$log"
            printf '\n' >> "$log"
            "$bench" "${case_args[@]}" >> "$log" 2>&1 || { cat "$log"; exit 1; }
            grep -q '^scatter_study PASS$' "$log"
            echo "PASSED: $log"
        done
    done
done
sha256sum "$out"/*.log > "$out/logs.sha256"
