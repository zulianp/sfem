#!/usr/bin/env bash
#SBATCH --account=c40
#SBATCH --job-name=cvfem-hex8-perf
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --time=00:45:00
#SBATCH --exclusive
#SBATCH --hint=nomultithread
#SBATCH --uenv-passthrough=ignore
#SBATCH --output=cvfem_hex8_perf_%j.out
#SBATCH --error=cvfem_hex8_perf_%j.err

# Linux perf profiling of the CVFEM HEX8 Navier-Stokes benchmark on one CSCS ALPS
# Grace socket (72 Neoverse V2 cores).
#
# The question this is meant to answer: assembly is bandwidth-bound on a laptop --
# is it also bandwidth-bound on Grace, and if not, what is it bound by? So the
# counter groups are chosen to separate "waiting on memory" from "not enough
# instruction-level parallelism":
#
#   basic     cycles / instructions / branches            -> IPC, branch behaviour
#   stalls    STALL_FRONTEND / STALL_BACKEND / _MEM       -> is it memory or issue?
#   cache     L1D / L2D / LL refills                      -> where the misses land
#   memory    MEM_ACCESS rd/wr, BUS_ACCESS                -> bytes actually moved
#   simd      ASE_SPEC / VFP_SPEC / DP_SPEC vs INST_SPEC  -> how vectorised the kernel is
#   scf       nvidia_scf_pmu cmem counters (if exposed)   -> real DRAM bandwidth
#
# Neoverse V2 has 6 programmable counters, so each group is a separate run rather
# than one multiplexed run -- multiplexed ratios are not trustworthy for a
# roofline argument. Every event is probed against `perf list` first and silently
# dropped if this kernel/PMU does not expose it.
#
# Submit:
#   sbatch perf_hex8_alps.sbatch
#
# Common overrides:
#   N=64 OPS="assemble" LAYOUTS="colored" sbatch perf_hex8_alps.sbatch
#   PERF_RECORD=1 sbatch perf_hex8_alps.sbatch        # + sampled profile per config
#   GROUPS="basic stalls memory" sbatch perf_hex8_alps.sbatch
#
# Output:
#   perf_hex8_alps_<ts>/cvfem_hex8_perf.csv    tidy: config,event,count  (+ derived metrics)
#   perf_hex8_alps_<ts>/perf_<cfg>_<grp>.txt   raw perf stat output
#   perf_hex8_alps_<ts>/report_<cfg>.txt       perf report, when PERF_RECORD=1
#   perf_hex8_alps_<ts>/available_events.txt   what this machine actually exposes

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

FORCE_BUILD="${FORCE_BUILD:-0}"
BUILD_DIR="${BUILD_DIR:-${SCRIPT_DIR}/build_alps}"
BENCH="${BENCH:-${BUILD_DIR}/cvfem_hex8_ns_upwind_bench}"
SFEM_DIR="${SFEM_DIR:-${SCRATCH:-}/installations/sfem/lib/cmake}"
CXX="${CXX:-g++}"
BUILD_JOBS="${BUILD_JOBS:-${SLURM_CPUS_PER_TASK:-72}}"

NCORES="${NCORES:-${SLURM_CPUS_PER_TASK:-72}}"
N="${N:-64}"
REPEAT="${REPEAT:-10}"
WARMUP="${WARMUP:-3}"
KERNEL="${KERNEL:-sumfact}"
OPS="${OPS:-residual jac-action assemble bsr-apply}"
LAYOUTS="${LAYOUTS:-packed colored}"
PACK_packed="${PACK_packed:-2048}"
PACK_colored="${PACK_colored:-512}"
PACK_store="${PACK_store:-384}"
PACK_atomic=0

GROUPS="${GROUPS:-basic stalls cache memory simd scf}"
PERF_STAT_REPEAT="${PERF_STAT_REPEAT:-2}"
PERF_RECORD="${PERF_RECORD:-0}"
PERF_FREQ="${PERF_FREQ:-499}"
CALL_GRAPH="${CALL_GRAPH:-none}"
RECORD_OPS="${RECORD_OPS:-assemble}"
RECORD_LAYOUTS="${RECORD_LAYOUTS:-colored}"

USE_SRUN="${USE_SRUN:-auto}"
USE_UENV="${USE_UENV:-1}"
UENV_IMAGE="${UENV_IMAGE:-prgenv-gnu/24.11:v1}"
UENV_VIEW="${UENV_VIEW:-default}"
CPU_BIND="${CPU_BIND:-cores}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${NCORES}}"
export OMP_PROC_BIND="${OMP_PROC_BIND:-close}"
export OMP_PLACES="${OMP_PLACES:-cores}"

stamp="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${OUT_DIR:-${SCRIPT_DIR}/perf_hex8_alps_${stamp}}"
mkdir -p "${OUT_DIR}"
CSV="${OUT_DIR}/cvfem_hex8_perf.csv"
EVENTS_TXT="${OUT_DIR}/available_events.txt"

die() { echo "error: $*" >&2; exit 1; }

use_srun_now() {
    [[ "${USE_SRUN}" == "0" ]] && return 1
    [[ "${USE_SRUN}" == "1" ]] && return 0
    [[ -n "${SLURM_JOB_ID:-}" ]] && command -v srun >/dev/null 2>&1
}

run_step() {
    if use_srun_now; then
        local -a srun_cmd=(srun -u --ntasks=1 --cpus-per-task="${NCORES}" --cpu-bind="${CPU_BIND}")
        [[ "${USE_UENV}" != "0" ]] && srun_cmd+=(--uenv="${UENV_IMAGE}" --view="${UENV_VIEW}")
        local cmdline; cmdline="$(printf "%q " "$@")"
        "${srun_cmd[@]}" bash -lc "${cmdline}"
    else
        "$@"
    fi
}

# ---------------------------------------------------------------------- build
if [[ "${FORCE_BUILD}" == "1" || ! -x "${BENCH}" ]]; then
    [[ -n "${SFEM_DIR}" ]] || die "SFEM_DIR is empty; set SFEM_DIR or SCRATCH"
    run_step cmake -S "${SCRIPT_DIR}" -B "${BUILD_DIR}" \
        -DCMAKE_CXX_COMPILER="${CXX}" -DSFEM_DIR="${SFEM_DIR}" -DCMAKE_BUILD_TYPE=Release
    run_step cmake --build "${BUILD_DIR}" --target cvfem_hex8_ns_upwind_bench -j "${BUILD_JOBS}"
fi
[[ -x "${BENCH}" ]] || die "benchmark not found after build: ${BENCH}"

run_step perf stat true >/dev/null 2>"${OUT_DIR}/perf_probe.txt" \
    || die "perf stat unavailable in this allocation (perf_event_paranoid?); see ${OUT_DIR}/perf_probe.txt"

# ------------------------------------------------------- event auto-discovery
# Ask the machine what it has rather than assuming a PMU model. Anything not
# listed is dropped, so the same script runs on Grace, on a laptop, and on any
# other Linux box without editing.
perf list --raw-dump 2>/dev/null | tr ' ' '\n' >"${EVENTS_TXT}" || true
if [[ ! -s "${EVENTS_TXT}" ]]; then
    perf list 2>/dev/null | awk '{print $1}' >"${EVENTS_TXT}" || true
fi

have_event() {
    local e="$1"
    grep -qx -- "${e}" "${EVENTS_TXT}" 2>/dev/null && return 0
    grep -qF -- "${e}" "${EVENTS_TXT}" 2>/dev/null && return 0
    # Last resort: ask perf directly, which catches PMU-qualified names that
    # `perf list` formats oddly. Deliberately not through run_step -- this script
    # already runs on the compute node, and a job step per probed event would cost
    # minutes of launch overhead to learn nothing extra.
    perf stat -e "${e}" -- true >/dev/null 2>&1
}

# keep only the events this machine exposes; echo the surviving comma list
filter_events() {
    local -a keep=()
    local e
    for e in "$@"; do
        if have_event "${e}"; then keep+=("${e}"); else echo "    dropped (unavailable): ${e}" >&2; fi
    done
    (IFS=,; printf "%s" "${keep[*]}")
}

group_events() {
    case "$1" in
        basic)  filter_events cycles instructions branches branch-misses ;;
        stalls) filter_events stalled-cycles-frontend stalled-cycles-backend \
                              armv8_pmuv3_0/STALL_FRONTEND/ armv8_pmuv3_0/STALL_BACKEND/ \
                              armv8_pmuv3_0/STALL_BACKEND_MEM/ ;;
        cache)  filter_events L1-dcache-loads L1-dcache-load-misses \
                              armv8_pmuv3_0/L2D_CACHE/ armv8_pmuv3_0/L2D_CACHE_REFILL/ \
                              armv8_pmuv3_0/LL_CACHE_RD/ armv8_pmuv3_0/LL_CACHE_MISS_RD/ ;;
        memory) filter_events armv8_pmuv3_0/MEM_ACCESS/ armv8_pmuv3_0/MEM_ACCESS_RD/ \
                              armv8_pmuv3_0/MEM_ACCESS_WR/ armv8_pmuv3_0/BUS_ACCESS/ \
                              dTLB-loads dTLB-load-misses ;;
        simd)   filter_events armv8_pmuv3_0/INST_SPEC/ armv8_pmuv3_0/ASE_SPEC/ \
                              armv8_pmuv3_0/VFP_SPEC/ armv8_pmuv3_0/DP_SPEC/ \
                              armv8_pmuv3_0/LD_SPEC/ armv8_pmuv3_0/ST_SPEC/ ;;
        scf)    filter_events nvidia_scf_pmu_0/cmem_rd_data/ nvidia_scf_pmu_0/cmem_wr_total_bytes/ \
                              nvidia_scf_pmu_0/cmem_rd_access/ nvidia_scf_pmu_0/cmem_wr_access/ ;;
        *) die "unknown counter group '$1'" ;;
    esac
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

pack_for() {
    case "$1" in
        atomic) printf "%s" "${PACK_atomic}" ;;
        packed) printf "%s" "${PACK_packed}" ;;
        colored) printf "%s" "${PACK_colored}" ;;
        store)  printf "%s" "${PACK_store}" ;;
        *) printf "512" ;;
    esac
}

echo "CVFEM HEX8 perf run"
echo "  bench:    ${BENCH}"
echo "  out_dir:  ${OUT_DIR}"
echo "  n:        ${N}  repeat=${REPEAT} warmup=${WARMUP}"
echo "  threads:  ${OMP_NUM_THREADS} (bind=${CPU_BIND})"
echo "  ops:      ${OPS}"
echo "  layouts:  ${LAYOUTS}"
echo "  groups:   ${GROUPS}"
echo

printf "config,operation,layout,kernel,pack_size,threads,cube_n,group,event,count,run_seconds\n" >"${CSV}"

for op in ${OPS}; do
    flag="$(op_flag "${op}")"
    for layout in ${LAYOUTS}; do
        [[ "${op}" == "bsr-apply" && "${layout}" != "packed" ]] && continue
        ps="$(pack_for "${layout}")"
        cfg="${op}_${layout}"
        cfg="${cfg//-/_}"

        cmd=("${BENCH}" --n "${N}" --repeat "${REPEAT}" --warmup "${WARMUP}"
             --layout "${layout}" --kernel "${KERNEL}")
        [[ -n "${flag}" ]] && cmd+=("${flag}")
        [[ "${layout}" != "atomic" ]] && cmd+=(--pack-size "${ps}")

        # a plain run first, so the csv carries the wall time the counters belong to
        run_step "${cmd[@]}" >"${OUT_DIR}/run_${cfg}.txt" 2>&1 || true
        secs="$(grep -m1 -E 'seconds_per_(assemble|apply|jac_action|bsr_apply):' "${OUT_DIR}/run_${cfg}.txt" \
                | awk '{print $2}')"
        secs="${secs:-nan}"

        for grp in ${GROUPS}; do
            echo "==> ${cfg} / ${grp}"
            evs="$(group_events "${grp}")"
            if [[ -z "${evs}" ]]; then
                echo "    no events available in this group, skipping"
                continue
            fi
            raw="${OUT_DIR}/perf_${cfg}_${grp}.txt"
            if ! run_step perf stat -x, -r "${PERF_STAT_REPEAT}" -e "${evs}" -o "${raw}" -- "${cmd[@]}" \
                    >>"${OUT_DIR}/run_${cfg}.txt" 2>>"${OUT_DIR}/run_${cfg}.err"; then
                echo "    perf stat failed for group ${grp}; see ${raw}"
                continue
            fi
            # perf -x, rows are: count,unit,event,runtime,pct[,...]
            awk -F, -v cfg="${cfg}" -v op="${op}" -v lay="${layout}" -v k="${KERNEL}" \
                    -v ps="${ps}" -v th="${OMP_NUM_THREADS}" -v n="${N}" -v grp="${grp}" -v s="${secs}" '
                /^#/ || NF < 3 { next }
                {
                    count = $1; ev = $3
                    gsub(/^[ \t]+|[ \t]+$/, "", count); gsub(/^[ \t]+|[ \t]+$/, "", ev)
                    if (count == "<not counted>" || count == "<not supported>" || count == "") next
                    printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n", cfg, op, lay, k, ps, th, n, grp, ev, count, s
                }' "${raw}" >>"${CSV}"
            echo "    -> ${raw}"
        done

        if [[ "${PERF_RECORD}" == "1" ]] \
           && [[ " ${RECORD_OPS} " == *" ${op} "* ]] \
           && [[ " ${RECORD_LAYOUTS} " == *" ${layout} "* ]]; then
            echo "==> perf record: ${cfg}"
            rec=("${OUT_DIR}/perf_${cfg}.data")
            record_cmd=(perf record -F "${PERF_FREQ}" -o "${rec[0]}")
            [[ "${CALL_GRAPH}" != "none" ]] && record_cmd+=(-g --call-graph "${CALL_GRAPH}")
            run_step "${record_cmd[@]}" -- "${cmd[@]}" \
                >>"${OUT_DIR}/record_${cfg}.txt" 2>&1 || true
            run_step perf report --stdio --percent-limit 0.5 --sort overhead,symbol,dso -i "${rec[0]}" \
                >"${OUT_DIR}/report_${cfg}.txt" 2>/dev/null || true
            echo "    -> ${OUT_DIR}/report_${cfg}.txt"
        fi
    done
done

echo
echo "Done."
echo "  counters: ${CSV}   ($(( $(wc -l <"${CSV}") - 1 )) rows)"
echo "  events this machine exposes: ${EVENTS_TXT}"
echo
echo "Next:"
echo "  python3 ${SCRIPT_DIR}/report_cvfem_bench.py <bench.csv> --perf ${CSV} -o report.html"
