#!/usr/bin/env bash
#SBATCH --account=c40
#SBATCH --job-name=cvfem-ranksweep
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=288
#SBATCH --time=00:20:00
#SBATCH --exclusive
#SBATCH --hint=nomultithread
#SBATCH --uenv-passthrough=ignore
#SBATCH --output=ranksweep_%j.out
#SBATCH --error=ranksweep_%j.err

# WHERE BETWEEN ONE RANK AND EIGHT DOES THE COARSE CORRECTION BREAK?
#
# The reproducer is now one Newton step. Run with SFEM_NL_MAX_IT=1 off the pre-Newton state,
# the first linear solve takes
#
#     1 rank 5      4 ranks 5      8 ranks 18
#
# and at that state every object in the coarse correction has been measured IDENTICAL across
# rank counts:
#
#   coarse block diagonal   258.95110245728023 at 1, 2, 4 and 8, every printed figure
#   coarse operator action  abs 0.98625725178447954 / ...965 / ...954 / ...965 at 1/2/4/8
#   transfer adjoint        <Rx,y> = <x,Py> = -1.633495e+02, ratio 1.000000, at 1, 4 and 8
#   coarse LU null space    1 of 500 dropped, component p, at every rank count
#
# The fine operator is clear too: with no multigrid at all the count is 812 / 812 / 812 / 811
# at 1 / 2 / 4 / 8. And with the coarse correction scaled to zero the full solve is 320 at
# four ranks and 320 at eight, identical line for line.
#
# So the question is no longer WHICH object, since they all measure invariant, but WHERE the
# decomposition changes character. With 64 macro elements a cut into two or four ranks can be
# slabs, while eight ranks forces a genuinely three-dimensional cut -- each rank holding a
# 2x2x2 block, acquiring neighbours across edges and corners rather than faces alone. If the
# degradation is a threshold rather than a trend, that geometry is the candidate, and the
# suspect becomes aura coverage for corner-diagonal neighbours rather than any single operator.
#
# READ, committed before the run:
#
#   count flat 1..7 then jumps at 8   -> a threshold tied to the cut becoming 3D; chase the
#                                        aura's corner and edge neighbours
#   count grows smoothly with ranks   -> a trend tied to the ghost fraction; chase whatever
#                                        accumulates per interface, not a geometric special case
#   some rank count errors out        -> the partitioner refuses that factorisation; that arm
#                                        is unmeasured, not a data point
#
# The checksums ride along at every rank count, so if the diagonal or the operator action does
# move at some rank count between 4 and 8, it is caught here rather than assumed invariant.
#
# cgc0 is the control at every rank count: the same cycle with the prolonged correction scaled
# to zero. It must stay flat across the whole sweep. If it does not, the fault is not in the
# coarse correction after all and every conclusion above needs revisiting.
#
# A TIMEOUT is unmeasured, not a result. One Newton step is about a second, so 120 s per arm
# is far above anything progressing.

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
OUT="${OUT:-$SCRATCH/cvfem_ranksweep/${SLURM_JOB_ID:-manual}}"
RAW="$OUT/raw"
mkdir -p "$RAW"
[ -x "$BIN" ] || { echo "error: no binary at $BIN" >&2; exit 1; }

ARM_TIMEOUT="${ARM_TIMEOUT:-120}"

run_arm() {
    local tag="$1" nranks="$2"; shift 2
    echo "### ${tag} ranks=${nranks} ###"
    timeout --signal=TERM --kill-after=30 "${ARM_TIMEOUT}" \
        srun -u --ntasks="${nranks}" --cpus-per-task=1 --kill-on-bad-exit=1 \
            --time="00:$(printf '%02d' $(( (ARM_TIMEOUT + 59) / 60 ))):00" \
            --uenv="$UENV_IMAGE" --view="$UENV_VIEW" \
            bash -lc "$* stdbuf -oL ${BIN} ${OUT}/${tag}_n${nranks}" 2>&1 \
        | tee "$RAW/${tag}_n${nranks}.log" \
        | stdbuf -oL grep -aE "^applysum |^diagsum |lin_it_total|newton_converged|nnodes:|error|Error"
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
        SFEM_LSOLVE_MAX_IT=4000 SFEM_LSOLVE_ATOL=1e-12"
CAVITY="SFEM_CASE=cavity SFEM_N=4 SFEM_ELEMENT_REFINE_LEVEL=2"
MG="SFEM_GMG=1 SFEM_GMG_GALERKIN=2 SFEM_SMOOTHER=bjacobi"
SUMS="SFEM_GMG_APPLY_OWNED_SUM=1 SFEM_GMG_DIAG_OWNED_SUM=1"
ONESTEP="SFEM_NL_MAX_IT=1 SFEM_VERIFY_TOL=1e9"

# One Newton step, every rank count from one to eight.
for n in 1 2 3 4 5 6 7 8; do
    run_arm "step1" "$n" "${COMMON} ${CAVITY} ${MG} ${SUMS} ${ONESTEP}"
done

# The control: the same cycle with the prolonged correction scaled to zero.
for n in 1 2 4 8; do
    run_arm "cgc0" "$n" "${COMMON} ${CAVITY} ${MG} ${ONESTEP} SFEM_GMG_CGC=0"
done

echo "raw logs: $RAW"
