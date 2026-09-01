#!/usr/bin/env bash
#SBATCH --account=c40
#SBATCH --job-name=cvfem-perf
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --time=00:30:00
#SBATCH --exclusive
#SBATCH --hint=nomultithread
#SBATCH --uenv-passthrough=ignore
#SBATCH --output=cvfem_perf_%j.out
#SBATCH --error=cvfem_perf_%j.err

# Linux perf profiling for the CVFEM TET4 Navier-Stokes upwind benchmark on CSCS ALPS.
#
# Submit:
#   sbatch profile_perf_alps.sbatch
#
# If submitting from an already-loaded uenv manually, this script ignores that
# environment and uses UENV_IMAGE for the job step below.
#
# Common overrides:
#   N=96 REPEAT=10 MODES="jac-action bsr-apply" sbatch profile_perf_alps.sbatch
#   PERF_RECORD=1 RECORD_MODES="jac-action" sbatch profile_perf_alps.sbatch
#   FORCE_BUILD=1 sbatch profile_perf_alps.sbatch
#
# Output:
#   perf_alps_<timestamp>/perf_<mode>.csv       perf stat counters
#   perf_alps_<timestamp>/run_<mode>.stdout.txt benchmark output
#   perf_alps_<timestamp>/run_<mode>.stderr.txt benchmark stderr / launcher output
#   perf_alps_<timestamp>/dof_rate_summary.tsv  worst/best/median/average unique-DOF rates
#   perf_alps_<timestamp>/perf_<mode>.data      optional perf record samples

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/cvfem_tet4_ns_upwind_bench.cpp" ]]; then
    SCRIPT_DIR="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
elif [[ -n "${CVFEM_DIR:-}" && -f "${CVFEM_DIR}/cvfem_tet4_ns_upwind_bench.cpp" ]]; then
    SCRIPT_DIR="$(cd "${CVFEM_DIR}" && pwd)"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi

if [[ ! -f "${SCRIPT_DIR}/cvfem_tet4_ns_upwind_bench.cpp" ]]; then
    echo "error: could not locate the CVFEM spike directory" >&2
    echo "       submit from spikes/cvfem or set CVFEM_DIR to the real spikes/cvfem path" >&2
    exit 1
fi
cd "${SCRIPT_DIR}"

FORCE_BUILD="${FORCE_BUILD:-0}"
BUILD_DIR="${BUILD_DIR:-${SCRIPT_DIR}/build_alps}"
BENCH="${BENCH:-${BUILD_DIR}/cvfem_tet4_ns_upwind_bench}"
SFEM_DIR="${SFEM_DIR:-${SCRATCH:-}/installations/sfem/lib/cmake}"
CXX="${CXX:-g++}"
BUILD_JOBS="${BUILD_JOBS:-${SLURM_CPUS_PER_TASK:-72}}"

N="${N:-64}"
REPEAT="${REPEAT:-}"
WARMUP="${WARMUP:-}"
LAYOUT="${LAYOUT:-packed}"
KERNEL="${KERNEL:-sympy_row_simd}"
PACK_SIZE="${PACK_SIZE:-}"
NO_SFC="${NO_SFC:-0}"

VERIFY="${VERIFY:-1}"
VERIFY_N="${VERIFY_N:-4}"

OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-72}}"
OMP_PROC_BIND="${OMP_PROC_BIND:-close}"
OMP_PLACES="${OMP_PLACES:-cores}"
CPU_BIND="${CPU_BIND:-cores}"

USE_SRUN="${USE_SRUN:-auto}"
USE_UENV="${USE_UENV:-1}"
UENV_IMAGE="${UENV_IMAGE:-prgenv-gnu/24.7:v3}"
UENV_VIEW="${UENV_VIEW:-default}"

PERF_STAT_REPEAT="${PERF_STAT_REPEAT:-}"
PERF_EVENTS="${PERF_EVENTS:-cycles,instructions,branches,branch-misses,cache-references,cache-misses,L1-dcache-loads,L1-dcache-load-misses,LLC-loads,LLC-load-misses,dTLB-loads,dTLB-load-misses}"
PERF_RECORD="${PERF_RECORD:-0}"
RECORD_MODES="${RECORD_MODES:-jac-action bsr-apply}"
PERF_FREQ="${PERF_FREQ:-199}"
CALL_GRAPH="${CALL_GRAPH:-none}"
PERF_REPORT="${PERF_REPORT:-0}"

if [[ -z "${MODES:-}" ]]; then
    if [[ "${PERF_RECORD}" == "1" ]]; then
        MODES="${RECORD_MODES}"
    else
        MODES="residual jac-action bsr-apply assemble"
    fi
fi

if [[ -z "${REPEAT}" ]]; then
    if [[ "${PERF_RECORD}" == "1" ]]; then
        REPEAT=5
    else
        REPEAT=20
    fi
fi

if [[ -z "${WARMUP}" ]]; then
    if [[ "${PERF_RECORD}" == "1" ]]; then
        WARMUP=2
    else
        WARMUP=5
    fi
fi

if [[ -z "${PERF_STAT_REPEAT}" ]]; then
    if [[ "${PERF_RECORD}" == "1" ]]; then
        PERF_STAT_REPEAT=1
    else
        PERF_STAT_REPEAT=3
    fi
fi

stamp="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${OUT_DIR:-${SCRIPT_DIR}/perf_alps_${stamp}}"
mkdir -p "${OUT_DIR}"
SUMMARY_TSV="${OUT_DIR}/dof_rate_summary.tsv"

export OMP_NUM_THREADS
export OMP_PROC_BIND
export OMP_PLACES

die() {
    echo "error: $*" >&2
    exit 1
}

need_cmd() {
    command -v "$1" >/dev/null 2>&1 || die "missing required command: $1"
}

quote_cmd() {
    printf "%q " "$@"
}

use_srun_now() {
    if [[ "${USE_SRUN}" == "0" ]]; then
        return 1
    fi
    if [[ "${USE_SRUN}" == "1" ]]; then
        return 0
    fi
    [[ -n "${SLURM_JOB_ID:-}" ]] && command -v srun >/dev/null 2>&1
}

run_step() {
    if use_srun_now; then
        local -a srun_cmd=(srun -u --ntasks=1 --cpus-per-task="${OMP_NUM_THREADS}" --cpu-bind="${CPU_BIND}")
        if [[ "${USE_UENV}" != "0" ]]; then
            srun_cmd+=(--uenv="${UENV_IMAGE}" --view="${UENV_VIEW}")
        fi
        local cmdline
        cmdline="$(quote_cmd "$@")"
        "${srun_cmd[@]}" bash -lc "${cmdline}"
    else
        "$@"
    fi
}

mode_extra_arg() {
    case "$1" in
        residual) ;;
        jac-action) printf "%s" "--jac-action" ;;
        bsr-apply) printf "%s" "--bsr-apply" ;;
        assemble) printf "%s" "--assemble" ;;
        *) die "unknown mode '$1' (expected residual, jac-action, bsr-apply, assemble)" ;;
    esac
}

bench_base_args() {
    local -a args=(--n "${N}" --repeat "${REPEAT}" --warmup "${WARMUP}" --layout "${LAYOUT}" --kernel "${KERNEL}")
    if [[ -n "${PACK_SIZE}" ]]; then
        args+=(--pack-size "${PACK_SIZE}")
    fi
    if [[ "${NO_SFC}" == "1" ]]; then
        args+=(--no-sfc)
    fi
    printf "%s\n" "${args[@]}"
}

contains_word() {
    local needle="$1"
    shift
    local word
    for word in "$@"; do
        [[ "${word}" == "${needle}" ]] && return 0
    done
    return 1
}

summarize_dof_rates() {
    local driver="$1"
    local stdout_log="$2"
    awk -v driver="${driver}" '
        /MDOF\/s_unique_mesh_dofs:/ {
            v = $2 + 0.0
            values[++n] = v
            sum += v
        }
        END {
            if (n == 0) {
                printf "%s\t0\tnan\tnan\tnan\tnan\n", driver
                exit
            }
            for (i = 1; i <= n; ++i) {
                v = values[i]
                j = i - 1
                while (j >= 1 && sorted[j] > v) {
                    sorted[j + 1] = sorted[j]
                    --j
                }
                sorted[j + 1] = v
            }
            if (n % 2) {
                median = sorted[(n + 1) / 2]
            } else {
                median = 0.5 * (sorted[n / 2] + sorted[n / 2 + 1])
            }
            printf "%s\t%d\t%.3f\t%.3f\t%.3f\t%.3f\n", driver, n, sorted[1], sorted[n], median, sum / n
        }
    ' "${stdout_log}"
}

print_summary_table() {
    if command -v column >/dev/null 2>&1; then
        column -t -s $'\t' "${SUMMARY_TSV}"
    else
        awk -F '\t' '{ printf "%-17s %5s %14s %14s %14s %14s\n", $1, $2, $3, $4, $5, $6 }' "${SUMMARY_TSV}"
    fi
}

build_bench() {
    [[ -n "${SFEM_DIR}" ]] || die "SFEM_DIR is empty; set SFEM_DIR or SCRATCH before building"
    run_step cmake -S "${SCRIPT_DIR}" -B "${BUILD_DIR}" \
        -DCMAKE_CXX_COMPILER="${CXX}" \
        -DSFEM_DIR="${SFEM_DIR}" \
        -DCMAKE_BUILD_TYPE=Release
    run_step cmake --build "${BUILD_DIR}" --target cvfem_tet4_ns_upwind_bench -j "${BUILD_JOBS}"
}

if [[ "${FORCE_BUILD}" != "0" && "${FORCE_BUILD}" != "1" ]]; then
    die "invalid FORCE_BUILD='${FORCE_BUILD}' (expected 0 or 1)"
fi

if [[ "${FORCE_BUILD}" == "1" || ! -x "${BENCH}" ]]; then
    build_bench
fi

[[ -x "${BENCH}" ]] || die "benchmark not found or not executable after build: ${BENCH}. Set SFEM_DIR or BENCH=/path/to/cvfem_tet4_ns_upwind_bench"

if [[ "${VERIFY}" == "1" ]]; then
    echo "==> verifying matrix-free Jacobian action on n=${VERIFY_N}"
    run_step "${BENCH}" --n "${VERIFY_N}" --repeat 1 --warmup 0 --layout packed --kernel "${KERNEL}" --verify-jac \
        >"${OUT_DIR}/verify.stdout.txt" 2>"${OUT_DIR}/verify.stderr.txt"
fi

if ! run_step perf stat true >/dev/null 2>"${OUT_DIR}/perf_probe.stderr.txt"; then
    die "perf stat is unavailable in this allocation. Check perf_event_paranoid / site permissions; see ${OUT_DIR}/perf_probe.stderr.txt"
fi

echo "CVFEM perf run"
echo "  work_dir:       ${SCRIPT_DIR}"
echo "  bench:          ${BENCH}"
echo "  out_dir:        ${OUT_DIR}"
echo "  n/repeat/warm:  ${N}/${REPEAT}/${WARMUP}"
echo "  layout/kernel:  ${LAYOUT}/${KERNEL}"
echo "  modes:          ${MODES}"
echo "  threads:        ${OMP_NUM_THREADS}, bind=${CPU_BIND}, OMP_PROC_BIND=${OMP_PROC_BIND}, OMP_PLACES=${OMP_PLACES}"
echo "  perf repeats:   ${PERF_STAT_REPEAT}"
echo "  events:         ${PERF_EVENTS}"
echo

readarray -t base_args < <(bench_base_args)
read -r -a mode_list <<<"${MODES}"
read -r -a record_mode_list <<<"${RECORD_MODES}"
printf "driver\truns\tDOF_worst\tDOF_best\tDOF_median\tDOF_average\n" >"${SUMMARY_TSV}"

for mode in "${mode_list[@]}"; do
    extra="$(mode_extra_arg "${mode}")"
    label="${mode//-/_}"
    stdout_log="${OUT_DIR}/run_${label}.stdout.txt"
    stderr_log="${OUT_DIR}/run_${label}.stderr.txt"
    stat_csv="${OUT_DIR}/perf_${label}.csv"

    cmd=("${BENCH}" "${base_args[@]}")
    if [[ -n "${extra}" ]]; then
        cmd+=("${extra}")
    fi

    echo "==> perf stat: ${mode}"
    run_step perf stat -x, -r "${PERF_STAT_REPEAT}" -e "${PERF_EVENTS}" -o "${stat_csv}" -- "${cmd[@]}" \
        >"${stdout_log}" 2>"${stderr_log}"

    echo "    counters: ${stat_csv}"
    echo "    stdout:   ${stdout_log}"
    grep -E "operation:|seconds_per_|MELEM|MDOF|GFLOP|GB/s|elements:|nodes:|nnz:" "${stdout_log}" || true
    summarize_dof_rates "${mode}" "${stdout_log}" >>"${SUMMARY_TSV}"

    if [[ "${PERF_RECORD}" == "1" ]] && contains_word "${mode}" "${record_mode_list[@]}"; then
        record_data="${OUT_DIR}/perf_${label}.data"
        record_report="${OUT_DIR}/perf_${label}.report.txt"
        record_cmd=(perf record -F "${PERF_FREQ}" -o "${record_data}")
        if [[ "${CALL_GRAPH}" != "none" ]]; then
            record_cmd+=(-g --call-graph "${CALL_GRAPH}")
        fi
        echo "==> perf record: ${mode}"
        run_step "${record_cmd[@]}" -- "${cmd[@]}" \
            >"${OUT_DIR}/record_${label}.stdout.txt" 2>"${OUT_DIR}/record_${label}.stderr.txt"
        echo "    samples:  ${record_data}"
        if [[ "${PERF_REPORT}" == "1" ]]; then
            run_step perf report --stdio --sort overhead,symbol,dso -i "${record_data}" \
                >"${record_report}" 2>"${OUT_DIR}/report_${label}.stderr.txt" || true
            echo "    report:   ${record_report}"
        fi
    fi
done

echo
echo "Unique mesh DOF-rate summary, MDOF/s:"
print_summary_table
echo
echo "Done. Counter CSVs and benchmark logs are in:"
echo "  ${OUT_DIR}"
