#!/usr/bin/env bash
#SBATCH --account=c40
#SBATCH --job-name=cvfem-hex8-bench
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --time=01:00:00
#SBATCH --exclusive
#SBATCH --hint=nomultithread
#SBATCH --uenv-passthrough=ignore
#SBATCH --output=cvfem_hex8_bench_%j.out
#SBATCH --error=cvfem_hex8_bench_%j.err

# CVFEM HEX8 Navier-Stokes benchmark sweep on one CSCS ALPS Grace socket.
#
# Produces a single CSV that plot_cvfem_bench.py and report_cvfem_bench.py read.
# One Grace socket is 72 Neoverse V2 cores on ~500 GB/s of LPDDR5X, so the
# assembly is expected to be even more bandwidth-bound here than on a laptop.
#
# Submit:
#   sbatch bench_hex8_alps.sbatch
#
# Common overrides:
#   SIZES="48 64 96 128 160 192" THREADS="72" sbatch bench_hex8_alps.sbatch
#   SWEEPS="layout threads" sbatch bench_hex8_alps.sbatch
#   TRIALS=5 sbatch bench_hex8_alps.sbatch          # best-of-N per configuration
#   FORCE_BUILD=1 sbatch bench_hex8_alps.sbatch
#
# Output:
#   bench_alps_<timestamp>/cvfem_hex8_bench.csv   every run, one row each
#   bench_alps_<timestamp>/verify.txt             correctness gate
#   bench_alps_<timestamp>/env.txt                machine / toolchain provenance

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/cvfem_hex8_ns_upwind_bench.cpp" ]]; then
    SCRIPT_DIR="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
elif [[ -n "${CVFEM_DIR:-}" && -f "${CVFEM_DIR}/cvfem_hex8_ns_upwind_bench.cpp" ]]; then
    SCRIPT_DIR="$(cd "${CVFEM_DIR}" && pwd)"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
[[ -f "${SCRIPT_DIR}/cvfem_hex8_ns_upwind_bench.cpp" ]] || {
    echo "error: could not locate spikes/cvfem; submit from there or set CVFEM_DIR" >&2; exit 1; }
cd "${SCRIPT_DIR}"

# ---------------------------------------------------------------- configuration
FORCE_BUILD="${FORCE_BUILD:-0}"
BUILD_DIR="${BUILD_DIR:-${SCRIPT_DIR}/build_alps}"
BENCH="${BENCH:-${BUILD_DIR}/cvfem_hex8_ns_upwind_bench}"
SFEM_DIR="${SFEM_DIR:-${SCRATCH:-}/installations/sfem/lib/cmake}"
CXX="${CXX:-g++}"
BUILD_JOBS="${BUILD_JOBS:-${SLURM_CPUS_PER_TASK:-72}}"

NCORES="${NCORES:-${SLURM_CPUS_PER_TASK:-72}}"
TAG="${TAG:-grace}"
TRIALS="${TRIALS:-3}"          # runs per configuration; the best is what lands in the csv
REPEAT="${REPEAT:-20}"
WARMUP="${WARMUP:-3}"

# which sweeps to run
SWEEPS="${SWEEPS:-layout packsize threads size kernel breakdown}"

# cube_n. The old default stopped at 96, which is below saturation on a GH200: at n=64
# the mesh has 262k elements against 132 SMs, roughly one element per resident thread,
# so throughput measured there is still climbing with problem size. The range now runs
# to 192 (7.2 M nodes, ~25 GiB BSR) so the plateau is inside it. Assembled runs at the
# top of the range are matrix-size bound, not compute bound -- drop SIZES if the node
# has less memory.
SIZES="${SIZES:-48 64 96 128 160 192}"
MAIN_N="${MAIN_N:-64}"                        # the size used by the non-size sweeps
THREADS="${THREADS:-1 2 4 9 18 36 72}"
PACK_SIZES="${PACK_SIZES:-128 256 384 512 768 1024 2048 4096}"
LAYOUTS="${LAYOUTS:-atomic packed colored store}"
KERNELS="${KERNELS:-sumfact sympy sympy_row current}"
OPS="${OPS:-residual jac-action assemble bsr-apply}"

# best pack size per layout for the sweeps that do not vary it; retune from the
# packsize sweep once you have Grace numbers.
PACK_DEFAULT_atomic=0
PACK_DEFAULT_packed="${PACK_DEFAULT_packed:-2048}"
PACK_DEFAULT_colored="${PACK_DEFAULT_colored:-512}"
PACK_DEFAULT_store="${PACK_DEFAULT_store:-384}"

USE_SRUN="${USE_SRUN:-auto}"
USE_UENV="${USE_UENV:-1}"
UENV_IMAGE="${UENV_IMAGE:-prgenv-gnu/24.11:v1}"
UENV_VIEW="${UENV_VIEW:-default}"
CPU_BIND="${CPU_BIND:-cores}"
export OMP_PROC_BIND="${OMP_PROC_BIND:-close}"
export OMP_PLACES="${OMP_PLACES:-cores}"

stamp="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${OUT_DIR:-${SCRIPT_DIR}/bench_alps_${stamp}}"
mkdir -p "${OUT_DIR}"
CSV="${OUT_DIR}/cvfem_hex8_bench.csv"

die() { echo "error: $*" >&2; exit 1; }

use_srun_now() {
    [[ "${USE_SRUN}" == "0" ]] && return 1
    [[ "${USE_SRUN}" == "1" ]] && return 0
    [[ -n "${SLURM_JOB_ID:-}" ]] && command -v srun >/dev/null 2>&1
}

# run one step on the allocated socket, with the thread count it asks for
run_step() {
    local nthreads="$1"; shift
    if use_srun_now; then
        local -a srun_cmd=(srun -u --ntasks=1 --cpus-per-task="${NCORES}" --cpu-bind="${CPU_BIND}")
        [[ "${USE_UENV}" != "0" ]] && srun_cmd+=(--uenv="${UENV_IMAGE}" --view="${UENV_VIEW}")
        local cmdline; cmdline="$(printf "%q " "$@")"
        OMP_NUM_THREADS="${nthreads}" "${srun_cmd[@]}" bash -lc "OMP_NUM_THREADS=${nthreads} ${cmdline}"
    else
        OMP_NUM_THREADS="${nthreads}" "$@"
    fi
}

op_flag() {
    case "$1" in
        residual)   printf "" ;;
        jac-action) printf -- "--jac-action" ;;
        assemble)   printf -- "--assemble" ;;
        bsr-apply)  printf -- "--bsr-apply" ;;
        *) die "unknown operation '$1'" ;;
    esac
}

pack_default_for() {
    case "$1" in
        atomic)  printf "%s" "${PACK_DEFAULT_atomic}" ;;
        packed)  printf "%s" "${PACK_DEFAULT_packed}" ;;
        colored) printf "%s" "${PACK_DEFAULT_colored}" ;;
        store)   printf "%s" "${PACK_DEFAULT_store}" ;;
        *) printf "512" ;;
    esac
}

# --tag encodes the sweep so the analysis scripts can slice the csv
# usage: measure <sweep> <threads> <n> <layout> <kernel> <op> <pack_size> [extra args...]
measure() {
    local sweep="$1" nthreads="$2" n="$3" layout="$4" kernel="$5" op="$6" ps="$7"; shift 7
    local -a extra=("$@")
    local flag; flag="$(op_flag "${op}")"
    local -a args=("${BENCH}" --n "${n}" --repeat "${REPEAT}" --warmup "${WARMUP}"
                   --layout "${layout}" --kernel "${kernel}"
                   --csv "${CSV}" --tag "${TAG}:${sweep}")
    [[ -n "${flag}" ]] && args+=("${flag}")
    [[ "${layout}" != "atomic" && "${ps}" != "0" ]] && args+=(--pack-size "${ps}")
    args+=("${extra[@]}")

    # One run per call. The trial loop is the *outermost* loop of the sweep, so
    # repeated trials of a configuration are spread across the whole job rather
    # than run back to back: any drift over the job then hits every configuration
    # equally instead of penalising whichever one happened to run during it.
    run_step "${nthreads}" "${args[@]}" >>"${OUT_DIR}/raw_${sweep}.txt" 2>&1 \
        || echo "  (failed: ${layout}/${kernel}/${op}/n=${n}/ps=${ps}/t=${nthreads})"
}

has_sweep() { [[ " ${SWEEPS} " == *" $1 "* ]]; }

# ---------------------------------------------------------------------- build
if [[ "${FORCE_BUILD}" == "1" || ! -x "${BENCH}" ]]; then
    [[ -n "${SFEM_DIR}" ]] || die "SFEM_DIR is empty; set SFEM_DIR or SCRATCH"
    run_step 1 cmake -S "${SCRIPT_DIR}" -B "${BUILD_DIR}" \
        -DCMAKE_CXX_COMPILER="${CXX}" -DSFEM_DIR="${SFEM_DIR}" -DCMAKE_BUILD_TYPE=Release
    run_step 1 cmake --build "${BUILD_DIR}" --target cvfem_hex8_ns_upwind_bench -j "${BUILD_JOBS}"
fi
[[ -x "${BENCH}" ]] || die "benchmark not found after build: ${BENCH}"

# ------------------------------------------------------------------ provenance
{
    echo "date:        $(date -Is)"
    echo "host:        $(hostname)"
    echo "slurm_job:   ${SLURM_JOB_ID:-none}"
    echo "cores:       ${NCORES}"
    echo "uenv:        ${UENV_IMAGE} (${UENV_VIEW})"
    echo "compiler:    $(${CXX} --version 2>/dev/null | head -1)"
    echo "sfem_dir:    ${SFEM_DIR}"
    echo "git:         $(git -C "${SCRIPT_DIR}" rev-parse --short HEAD 2>/dev/null || echo n/a)"
    echo
    echo "--- lscpu ---"; lscpu 2>/dev/null || true
    echo
    echo "--- meminfo ---"; head -5 /proc/meminfo 2>/dev/null || true
} >"${OUT_DIR}/env.txt" 2>&1

# ------------------------------------------------------------------- verify
echo "==> verifying (n=16, every layout)"
{
    for l in atomic packed colored store; do
        echo "--- layout ${l}"
        run_step "${NCORES}" "${BENCH}" --n 16 --repeat 1 --warmup 0 --layout "${l}" \
            --kernel sumfact --assemble --verify-jac --pack-size 256 2>&1 | grep -E "verify_|error" || true
    done
    echo "--- residual (all kernels)"
    run_step "${NCORES}" "${BENCH}" --n 16 --repeat 1 --warmup 0 --layout packed --verify --pack-size 256 2>&1 \
        | grep -E "verify_|error" || true
} >"${OUT_DIR}/verify.txt" 2>&1
grep -q "mismatch" "${OUT_DIR}/verify.txt" && die "verification failed; see ${OUT_DIR}/verify.txt"
echo "    ok -> ${OUT_DIR}/verify.txt"

# -------------------------------------------------------------------- sweeps
echo "==> sweeps: ${SWEEPS}  (${TRIALS} interleaved passes)"

for pass in $(seq 1 "${TRIALS}"); do
echo "== pass ${pass}/${TRIALS}"

# 1. every layout x every operation at full width
if has_sweep layout; then
    echo "-- layout x operation"
    for op in ${OPS}; do
        for l in ${LAYOUTS}; do
            [[ "${op}" == "bsr-apply" && "${l}" != "packed" ]] && continue   # spmv is layout-independent
            measure layout "${NCORES}" "${MAIN_N}" "${l}" sumfact "${op}" "$(pack_default_for "${l}")"
        done
    done
fi

# 2. pack size, per layout and operation
if has_sweep packsize; then
    echo "-- pack size"
    for op in residual jac-action assemble; do
        for l in packed colored store; do
            [[ "${l}" == "store" && "${op}" != "assemble" ]] && continue     # store falls back to packed
            for ps in ${PACK_SIZES}; do
                measure packsize "${NCORES}" "${MAIN_N}" "${l}" sumfact "${op}" "${ps}"
            done
        done
    done
fi

# 3. thread scaling
if has_sweep threads; then
    echo "-- thread scaling"
    for t in ${THREADS}; do
        for op in residual assemble; do
            for l in atomic packed colored; do
                measure threads "${t}" "${MAIN_N}" "${l}" sumfact "${op}" "$(pack_default_for "${l}")"
            done
        done
    done
fi

# 4. problem size
if has_sweep size; then
    echo "-- problem size"
    for n in ${SIZES}; do
        for op in residual assemble; do
            for l in atomic packed colored store; do
                [[ "${l}" == "store" && "${op}" != "assemble" ]] && continue
                measure size "${NCORES}" "${n}" "${l}" sumfact "${op}" "$(pack_default_for "${l}")"
            done
        done
    done
fi

# 5. element kernel variants
if has_sweep kernel; then
    echo "-- kernel variants"
    for k in ${KERNELS}; do
        for op in residual assemble; do
            measure kernel "${NCORES}" "${MAIN_N}" colored "${k}" "${op}" "${PACK_DEFAULT_colored}"
            measure kernel "${NCORES}" "${MAIN_N}" packed  "${k}" "${op}" "${PACK_DEFAULT_packed}"
        done
    done
fi

done   # trial passes

# 6. phase breakdown, once: the timers perturb the run slightly, so these rows are
#    for the shape of the budget, not for the headline rate.
if has_sweep breakdown; then
    echo "-- phase breakdown"
    for l in packed colored store; do
        measure breakdown "${NCORES}" "${MAIN_N}" "${l}" sumfact assemble "$(pack_default_for "${l}")" --breakdown
    done
    measure breakdown "${NCORES}" "${MAIN_N}" packed sumfact assemble "${PACK_DEFAULT_packed}" --breakdown --kernel-only
fi

echo
echo "Done."
echo "  csv:  ${CSV}   ($(( $(wc -l <"${CSV}") - 1 )) rows)"
echo "  logs: ${OUT_DIR}"
echo
echo "Next:"
echo "  python3 ${SCRIPT_DIR}/report_cvfem_bench.py ${CSV} -o ${OUT_DIR}/report.html"
echo "  python3 ${SCRIPT_DIR}/plot_cvfem_bench.py   ${CSV} -o ${OUT_DIR}/plots"
