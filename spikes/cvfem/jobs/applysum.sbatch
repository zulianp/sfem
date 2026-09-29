#!/usr/bin/env bash
#SBATCH --account=c40
#SBATCH --job-name=cvfem-applysum
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=288
#SBATCH --time=00:25:00
#SBATCH --exclusive
#SBATCH --hint=nomultithread
#SBATCH --uenv-passthrough=ignore
#SBATCH --output=applysum_%j.out
#SBATCH --error=applysum_%j.err

# IS THE COARSE OPERATOR'S ACTION DECOMPOSITION INVARIANT?
#
# Where the investigation stands. Multigrid + block-Jacobi measures 158 / 158 / 158 / 726 at
# 1 / 2 / 4 / 8 ranks, and with the coarse correction scaled to zero it measures 320 at four
# ranks and 320 at eight -- identical, and the whole diagsum sequence matches line for line.
# So the solver is exactly invariant without a coarse grid, and the fault is in the coarse
# correction, at eight ranks specifically.
#
# Two pieces of that correction are already cleared:
#
#   the coarse block diagonal   pre-Newton owned sum 258.95110245728023 at 1, 2, 4 AND 8
#                               ranks, every printed figure -- the owner is authoritative
#   the transfers               <Rx,y> = <x,Py> = -1.633495e+02, ratio 1.000000, at 1, 2
#                               and 4 ranks -- R is P-transpose under decomposition
#
# What is left is the operator's ACTION. galerkin_apply reduces element contributions through
# a purely local GalerkinReduce table and nothing in cvfem_ss_galerkin.hpp communicates, which
# is the same shape as the fine nodal reconstruction that was the last defect found here: the
# owned output was complete only because the aura happened to cover it, and it stopped being
# complete as the cuts got finer.
#
# ARM ONE, applysum: the coarse operator applied to a COORDINATE-defined direction, summed
# over owned coarse nodes and reduced. Probe free in the sense that matters -- no colouring,
# no composite, no index-keyed vector. Compare the FIRST line of each arm: later lines sit on
# a state the 8-rank run has already diverged from, so only the pre-Newton one is like for
# like.
#
#   first applysum line identical at 1/2/4/8 -> the coarse action is invariant, the fault is
#                                               in the coarse SOLVE or in how the correction
#                                               is applied, and GlobalDenseLU at 8 is next
#   first applysum line differs at 8 only    -> galerkin_apply owes a cross-rank completion,
#                                               exactly as the fine reconstruction did
#
# ARM TWO, adjoint at eight ranks. The transfer check could never run above two ranks: the
# probe loop derives its colour count from the rank-local sparsity pattern while composite()
# applies the collective fine operator, so ranks issued different numbers of collectives and
# deadlocked -- 27 colours on both ranks at two, 18 against something else at four. The count
# is now reduced to a global maximum, verified at four ranks locally (uniform 27 colours, run
# completes, one adjoint line, ratio 1.000000). This is that fix on cray-mpich, and it closes
# the last untested rank count for the transfers.
#
#   ratio 1.000000 at 8 -> transfers exonerated everywhere the cycle is measured
#   ratio != 1 at 8     -> the transfers break at 8 and that is the cause
#
# A TIMEOUT is unmeasured, not a result. The diagsum job ran six arms in 30 s total, so 180 s
# per arm is far above anything progressing and far below the allocation.

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
OUT="${OUT:-$SCRATCH/cvfem_applysum/${SLURM_JOB_ID:-manual}}"
RAW="$OUT/raw"
mkdir -p "$RAW"
[ -x "$BIN" ] || { echo "error: no binary at $BIN" >&2; exit 1; }

ARM_TIMEOUT="${ARM_TIMEOUT:-180}"

run_arm() {
    local tag="$1" nranks="$2"; shift 2
    echo "### ${tag} ranks=${nranks} ###"
    timeout --signal=TERM --kill-after=30 "${ARM_TIMEOUT}" \
        srun -u --ntasks="${nranks}" --cpus-per-task=1 --kill-on-bad-exit=1 \
            --time="00:$(printf '%02d' $(( (ARM_TIMEOUT + 59) / 60 ))):00" \
            --uenv="$UENV_IMAGE" --view="$UENV_VIEW" \
            bash -lc "$* stdbuf -oL ${BIN} ${OUT}/${tag}_n${nranks}" 2>&1 \
        | tee "$RAW/${tag}_n${nranks}.log" \
        | stdbuf -oL grep -aE "^applysum |^diagsum |^  adjoint |^assembly gate|colours,|newton_converged|lin_it_total"
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

# The coarse operator's action, against the diagonal that is already known invariant.
for n in 1 2 4 8; do
    run_arm "applysum" "$n" "${COMMON} ${CAVITY} ${MG} ${SUMS}"
done

# The transfer adjoint, at the rank counts that were unreachable before the colour-count fix.
# One Newton step is enough: check_transfers prints during setup.
for n in 1 4 8; do
    run_arm "adj" "$n" \
        "${COMMON} ${CAVITY} ${MG} SFEM_GMG_CHECK=1 SFEM_NL_MAX_IT=1 SFEM_VERIFY_TOL=1e9"
done

echo "raw logs: $RAW"
