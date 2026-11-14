#!/usr/bin/env bash
#SBATCH --job-name=bench-mamal
#SBATCH --account=c40
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --gpus-per-node=1
#SBATCH --time=10:00:00
#SBATCH --output=slurm-bench-mamal-%j.out
#SBATCH --error=slurm-bench-mamal-%j.err
#SBATCH --exclusive
#SBATCH --partition=normal
#SBATCH --uenv-passthrough=use

set -uo pipefail


echo "#---------------#"
date
echo "#---------------#"

export PATH=/capstor/scratch/cscs/zulianp/sfem/build:$PATH

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-72}"
export OMP_PROC_BIND="${OMP_PROC_BIND:-true}"
export OMP_PLACES="${OMP_PLACES:-cores}"
export MPICH_GPU_SUPPORT_ENABLED="${MPICH_GPU_SUPPORT_ENABLED:-1}"

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
BUILD_DIR="${BUILD_DIR:-${SUBMIT_DIR}}"
BIN="${SFEM_BIN:-/capstor/scratch/cscs/zulianp/sfem/build/sfem_MGSDFContactTest}"
OUT_DIR="${OUT_DIR:-bench_MaMAL_${SLURM_JOB_ID:-local}}"
CSV="${CSV:-${OUT_DIR}/bench_MaMAL.csv}"

SCENARIOS="${SCENARIOS:-sphere multisphere}"
EXECUTION_SPACES="${EXECUTION_SPACES:-device}"
LINE_SEARCH_MODES="${LINE_SEARCH_MODES:-1 0}"
NL_OBSTACLE_MODES="${NL_OBSTACLE_MODES:-1}"
SIZES="${SIZES:-1:8 2:8 4:8 6:8 8:8 9:8 10:8}"
# SIZES="${SIZES:-9:8 10:8}"
MAX_NODES="${MAX_NODES:-207619281}"

export SFEM_MAX_COARSE_IT="${SFEM_MAX_COARSE_IT:-400000}"
# sfem_MGSDFContactTest runs test_contact alone: its solve's statistics (count_iter, count_mg_cycles,
# relative_residual) are then the stats file's, which the binary's later tests would overwrite.
export SFEM_MGSDF_CONTACT_ONLY="${SFEM_MGSDF_CONTACT_ONLY:-1}"
export SFEM_ATOL="${SFEM_ATOL:-1e-8}"

case "${OUT_DIR}" in
    /*) ;;
    *) OUT_DIR="${SUBMIT_DIR}/${OUT_DIR}" ;;
esac

case "${CSV}" in
    /*) ;;
    *) CSV="${SUBMIT_DIR}/${CSV}" ;;
esac

mkdir -p "${OUT_DIR}/logs" "${OUT_DIR}/traces" "${OUT_DIR}/stats"
cd "${BUILD_DIR}" || exit 1

if [[ ! -x "${BIN}" ]]; then
    printf 'error: %s is not executable in %s\n' "${BIN}" "${BUILD_DIR}" >&2
    exit 1
fi

# Trace CSV: name,calls,total,avg
# Stats CSV: count_iter,count_mg_cycles,...,omega,relative_residual,rate
printf 'scenario,execution_space,line_search,nl_obstacle,base_resolution,element_refine_level,nodes,status,elapsed_s,test_contact_s,count_iter,count_mg_cycles,relative_residual,trace_file,stats_file,log_file\n' > "${CSV}"

# Read a named column from a headered CSV for the matching row / last data row.
csv_col() {
    local file="$1"
    local key_col="$2"
    local key_val="$3"
    local out_col="$4"
    local mode="${5:-first}" # first | last

    awk -F, -v key_col="${key_col}" -v key_val="${key_val}" -v out_col="${out_col}" -v mode="${mode}" '
        NR == 1 {
            for (i = 1; i <= NF; i++) {
                gsub(/\r/, "", $i)
                h[$i] = i
            }
            if (!(out_col in h)) exit 0
            if (key_col != "" && !(key_col in h)) exit 0
            next
        }
        {
            gsub(/\r/, "", $0)
            if (key_col == "" || $(h[key_col]) == key_val) {
                val = $(h[out_col])
                if (mode == "first") {
                    print val
                    exit
                }
                last = val
            }
        }
        END {
            if (mode == "last" && last != "") print last
        }
    ' "${file}"
}

run_one() {
    local scenario="$1"
    local execution_space="$2"
    local line_search="$3"
    local nl_obstacle="$4"
    local base_resolution="$5"
    local refine_level="$6"
    local nodes="$7"

    local tag="${scenario}_${execution_space}_ls${line_search}_nlobs${nl_obstacle}_b${base_resolution}_l${refine_level}"
    local log_file="${OUT_DIR}/logs/${tag}.log"
    local trace_file="${OUT_DIR}/traces/${tag}.csv"
    local stats_file="${OUT_DIR}/stats/${tag}.csv"
    local start_s end_s elapsed_s status
    local test_contact_s count_iter count_mg_cycles relative_residual

    printf '[%s] nodes=%s\n' "${tag}" "${nodes}"
    start_s=$(date +%s)

    SFEM_ENABLE_OUTPUT=0 \
    SFEM_USE_SPMG=1 \
    SFEM_CONTACT_CASE="${scenario}" \
    SFEM_EXECUTION_SPACE="${execution_space}" \
    SFEM_ENABLE_LINE_SEARCH="${line_search}" \
    SFEM_ENABLE_NL_OBSTACLE="${nl_obstacle}" \
    SFEM_BASE_RESOLUTION="${base_resolution}" \
    SFEM_ELEMENT_REFINE_LEVEL="${refine_level}" \
    SMESH_TRACE_FILE="${trace_file}" \
    SFEM_SHIFTED_PENALTY_MULTIGRID_STATS_PATH="${stats_file}" \
        srun --nodes=1 --ntasks=1 --cpus-per-task="${OMP_NUM_THREADS}" --cpu-bind=cores "${BIN}" > "${log_file}" 2>&1

    status=$?
    end_s=$(date +%s)
    elapsed_s=$((end_s - start_s))

    test_contact_s=""
    if [[ -s "${trace_file}" ]]; then
        test_contact_s=$(csv_col "${trace_file}" name test_contact total first)
    fi

    count_iter=""
    count_mg_cycles=""
    relative_residual=""
    if [[ -s "${stats_file}" ]]; then
        count_iter=$(csv_col "${stats_file}" "" "" count_iter last)
        count_mg_cycles=$(csv_col "${stats_file}" "" "" count_mg_cycles last)
        relative_residual=$(csv_col "${stats_file}" "" "" relative_residual last)
    fi

    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
        "${scenario}" \
        "${execution_space}" \
        "${line_search}" \
        "${nl_obstacle}" \
        "${base_resolution}" \
        "${refine_level}" \
        "${nodes}" \
        "${status}" \
        "${elapsed_s}" \
        "${test_contact_s}" \
        "${count_iter}" \
        "${count_mg_cycles}" \
        "${relative_residual}" \
        "${trace_file}" \
        "${stats_file}" \
        "${log_file}" >> "${CSV}"
}

for size in ${SIZES}; do
    base_resolution="${size%%:*}"
    refine_level="${size##*:}"
    n=$((base_resolution * refine_level))
    nodes=$(((20 * n + 1) * (n + 1) * (20 * n + 1)))

    if ((nodes > MAX_NODES)); then
        continue
    fi

    for scenario in ${SCENARIOS}; do
        for execution_space in ${EXECUTION_SPACES}; do
            for line_search in ${LINE_SEARCH_MODES}; do
                for nl_obstacle in ${NL_OBSTACLE_MODES}; do
                    run_one "${scenario}" "${execution_space}" "${line_search}" "${nl_obstacle}" "${base_resolution}" "${refine_level}" "${nodes}"
                done
            done
        done
    done
done

printf 'results: %s\n' "${CSV}"

echo "#---------------#"
date
echo "#---------------#"