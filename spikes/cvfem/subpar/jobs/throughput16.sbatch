#!/usr/bin/env bash
#SBATCH --account=c40
#SBATCH --job-name=cvfem-thr16
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=288
#SBATCH --time=01:00:00
#SBATCH --exclusive
#SBATCH --hint=nomultithread
#SBATCH --uenv-passthrough=ignore
#SBATCH --output=thr16_%j.out
#SBATCH --error=thr16_%j.err

# FIRST HONEST THROUGHPUT NUMBERS, AND WHY THE EARLIER ONES WERE NOT
#
# The invariance work is done: four decomposition defects fixed, and MG + Vanka now gives the
# same linear iteration count from 1 to 288 ranks on a four-level 1,098,500-dof problem.
#
# The timings from that run are NOT a scaling result and must not be quoted as one. It used
# 512 macro elements, so at 288 ranks each rank owned one or two elements and the aura
# dominated everything: Vanka went 20.81 / 7.68 / 4.77 / 4.33 / 4.12 / 4.96 s at 1 / 8 / 32 /
# 64 / 128 / 288 -- turning over after 128 -- and block-Jacobi on the smaller case was SLOWER
# than serial at every parallel rank count. That is what an under-resourced decomposition looks
# like, not what the machine can do.
#
# This case fixes the element count. Cavity N=16 at refine level 4:
#
#     4,096 macro elements      274,625 nodes      1,098,500 dof      3 levels
#     14.2 elements per rank at 288, against 1.8 before
#     23.4 s serial on 8 local threads, lin_it 22
#
# Same dof count as the invariance case, eight times the elements -- which is the variable that
# actually decides whether a rank has work to do.
#
# TWO LIMITS, STATED UP FRONT.
#
# This builds THREE levels, not four, because refine 4 derefines 4 -> 2 -> 1. It therefore
# exercises the depth fixes less than the N=8 / refine 8 case did. It is chosen for elements per
# rank, which is what a throughput measurement needs.
#
# And this is a pure-MPI rank sweep, one thread per rank. The rank x thread shape study at
# fixed 288 cores -- 4x72, 8x36, 12x24, 18x16, 36x8, 72x4 -- is separate work and is where the
# NUMA question actually gets answered.
#
# READ, committed before the run:
#
#   lin_it 22 at every rank count       -> invariance holds on this case too, and the timings
#                                          below are comparing the same computation
#   lin_it moving                       -> a decomposition defect this case exposes and the
#                                          others did not; that outranks any timing here
#   time falling then turning over      -> the element count is still too low at the top end;
#                                          the turnover point is the useful number, not the
#                                          minimum
#
# Timings are reported with the dof count, since a second without a problem size is not a
# result. A TIMEOUT is unmeasured. A count at the iteration cap did not converge and is a floor.

set -u

_cvfem_probe=scripts/alps_env.sh
SPIKE_ROOT=""
for _cand in "${SLURM_SUBMIT_DIR:-}" "${SLURM_SUBMIT_DIR:-}/.." "${CVFEM_DIR:-}" \
             "$(dirname "${BASH_SOURCE[0]}")/.."; do
    [ -n "${_cand}" ] && [ -f "${_cand}/${_cvfem_probe}" ] || continue
    SPIKE_ROOT="$(cd "${_cand}" && pwd)"; break
done
[ -n "${SPIKE_ROOT}" ] || { echo "error: could not locate spikes/cvfem" >&2; exit 1; }
cd "$SPIKE_ROOT"

UENV_IMAGE="prgenv-gnu/24.11:v2"
UENV_VIEW="default"
BIN="${CVFEM_BIN:-./build/cvfem_hex8_ns_ssgmg}"
OUT="${OUT:-$SCRATCH/cvfem_thr16/${SLURM_JOB_ID:-manual}}"
RAW="$OUT/raw"
mkdir -p "$RAW"
[ -x "$BIN" ] || { echo "error: no binary at $BIN" >&2; exit 1; }

# Generous: the serial arm is the slow one (23 s on 8 local threads, and this runs one thread
# per rank), and an arm that is merely slow must not be recorded as a timeout.
ARM_TIMEOUT="${ARM_TIMEOUT:-900}"

run_arm() {
    local tag="$1" nranks="$2"; shift 2
    echo "### ${tag} ranks=${nranks} ###"
    timeout --signal=TERM --kill-after=30 "${ARM_TIMEOUT}" \
        srun -u --ntasks="${nranks}" --cpus-per-task=1 --kill-on-bad-exit=1 \
            --time="00:$(printf '%02d' $(( (ARM_TIMEOUT + 59) / 60 ))):00" \
            --uenv="$UENV_IMAGE" --view="$UENV_VIEW" \
            bash -lc "$* stdbuf -oL ${BIN} ${OUT}/${tag}_n${nranks}" 2>&1 \
        | tee "$RAW/${tag}_n${nranks}.log" \
        | stdbuf -oL grep -aE "^nnodes:|^global:|lin_it_total|newton_converged|gmg: [0-9]+ levels|ns_ssgmg: .* seconds"
    local rc=${PIPESTATUS[0]}
    if [ "$rc" = "124" ] || [ "$rc" = "137" ]; then
        echo "  (arm exit: ${rc} -- TIMEOUT after ${ARM_TIMEOUT}s)"
    else
        echo "  (arm exit: ${rc})"
    fi
    echo
}

COMMON="OMP_NUM_THREADS=1 SFEM_FGMRES=1 SFEM_ENABLE_OUTPUT=0 \
        SFEM_PARALLEL_MF_LEGACY=1 SFEM_ALLOW_MPI=1 \
        SFEM_LSOLVE_MAX_IT=4000 SFEM_LSOLVE_RTOL=1e-4 SFEM_LSOLVE_ATOL=1e-14"
CASE="SFEM_CASE=cavity SFEM_N=16 SFEM_ELEMENT_REFINE_LEVEL=4"
MG="SFEM_GMG=1 SFEM_GMG_GALERKIN=2 SFEM_SMOOTHER=vanka"
ONESTEP="SFEM_NL_MAX_IT=1 SFEM_VERIFY_TOL=1e9"

for n in 1 8 32 64 128 288; do
    run_arm "thr" "$n" "${COMMON} ${CASE} ${MG} ${ONESTEP}"
done

echo "raw logs: $RAW"
