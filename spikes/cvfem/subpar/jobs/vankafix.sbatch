#!/usr/bin/env bash
#SBATCH --account=c40
#SBATCH --job-name=cvfem-vankafix
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=288
#SBATCH --time=00:25:00
#SBATCH --exclusive
#SBATCH --hint=nomultithread
#SBATCH --uenv-passthrough=ignore
#SBATCH --output=vankafix_%j.out
#SBATCH --error=vankafix_%j.err

# THE VANKA SMOOTHER UNDER DECOMPOSITION, AFTER THE COARSE PATH WAS FIXED
#
# WHERE THIS STANDS. The coarse defect is fixed and validated on cray-mpich: GlobalDenseLU
# gathered the coarse right-hand side by rank-order tiling while scattering the answer through
# the explicit local-to-global map, which is only right when a rank's owned coarse ids happen
# to be contiguous. Multigrid with block-Jacobi now measures 158 linear iterations at 1, 2, 4,
# 5, 6 AND 8 ranks, against 158 / 158 / 158 / 726 before, with the awkward cuts at five and six
# -- previously the worst, at 54 and 210 on one Newton step -- now identical to serial.
#
# Vanka is what is left, and it is the larger defect:
#
#     full solve, MG + vanka      61   61   62   75  133  8232   at 1 2 4 5 6 8
#     full solve, MG + bjacobi   158  158  158  158  158   158
#     no MG, vanka               434  443  509             535   at 1 2 4 8
#     no MG, bjacobi             812  812  812             811
#
# Vanka SHOULD be invariant. Its multiplicative sweep stays inside one macro-element on a
# private local vector and corrections combine additively between elements, so with whole
# macro-elements per rank no sweep reads what another rank just wrote. Four candidate causes
# have been eliminated by measurement rather than argument:
#
#   the linearisation state   gmgstate level 0 sum 81, free_abs 0, identical at 1/2/4/8
#   the patch weights         accumulated over the local macro array INCLUDING the aura, and
#                             the aura makes every macro-element touching an owned node local
#   the wrapper               create_parallel_preconditioner gathers GhostsAndAura in place
#                             before the inner apply, and is correct
#   the coarse correction     the drift persists with SFEM_GMG_CGC=0, where the coarse solve
#                             contributes nothing, and without multigrid at all
#
# WHAT THIS JOB CARRIES.
#
# (1) THE NO-MULTIGRID FIX. create_parallel_preconditioner had exactly one call site in the
#     whole driver, on the multigrid fine level, whose comment states the defect plainly:
#     Multigrid hands the smoother a residual reduced over the owned range alone, so a patch
#     reading its macro-element's ghost and aura nodes reads values nothing filled. The
#     no-multigrid branch builds the same smoother and never wrapped it. It does now.
#
# (2) smoothsum, a direct measurement of the smoother AS WRAPPED, which is what the cycle
#     calls. A coordinate-defined direction so every rank count probes the same vector, summed
#     over the owned range and reduced. This is the hypothesis-free test the coarse operator
#     got, and it is what the remaining MG-path drift needs, since every cheap explanation is
#     now eliminated.
#
# READ, committed before the run:
#
#   no-MG vanka flat 1..8        -> the missing wrap was that path's whole defect
#   no-MG vanka still growing    -> the wrap is not sufficient and the sweep itself is at fault
#   smoothsum flat, MG still drifts -> the smoother's apply IS invariant and the difference is
#                                      elsewhere in the cycle; stop bisecting the smoother
#   smoothsum moves with ranks   -> the smoother's apply is provably not invariant, and the
#                                   bisection continues inside vanka_apply_mult
#
# Rank counts run past powers of two because every defect found here appeared at the five-rank
# transition, where a 4x4x4 macro lattice stops cutting on whole layers. A TIMEOUT is
# unmeasured, not a result; the 8232-iteration arm is why the cap is generous.

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
OUT="${OUT:-$SCRATCH/cvfem_vankafix/${SLURM_JOB_ID:-manual}}"
RAW="$OUT/raw"
mkdir -p "$RAW"
[ -x "$BIN" ] || { echo "error: no binary at $BIN" >&2; exit 1; }

ARM_TIMEOUT="${ARM_TIMEOUT:-240}"

run_arm() {
    local tag="$1" nranks="$2"; shift 2
    echo "### ${tag} ranks=${nranks} ###"
    timeout --signal=TERM --kill-after=30 "${ARM_TIMEOUT}" \
        srun -u --ntasks="${nranks}" --cpus-per-task=1 --kill-on-bad-exit=1 \
            --time="00:$(printf '%02d' $(( (ARM_TIMEOUT + 59) / 60 ))):00" \
            --uenv="$UENV_IMAGE" --view="$UENV_VIEW" \
            bash -lc "$* stdbuf -oL ${BIN} ${OUT}/${tag}_n${nranks}" 2>&1 \
        | tee "$RAW/${tag}_n${nranks}.log" \
        | stdbuf -oL grep -aE "^smoothsum |lin_it_total|newton_converged|preconditioner:"
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
        SFEM_LSOLVE_MAX_IT=12000 SFEM_LSOLVE_ATOL=1e-12"
CAVITY="SFEM_CASE=cavity SFEM_N=4 SFEM_ELEMENT_REFINE_LEVEL=2"
MGV="SFEM_GMG=1 SFEM_GMG_GALERKIN=2 SFEM_SMOOTHER=vanka"
ONESTEP="SFEM_NL_MAX_IT=1 SFEM_VERIFY_TOL=1e9"

# (1) The smoother's own apply, measured directly on the wrapped operator.
#
# Measured locally and NOT invariant: the owned-range checksum differs from serial already at
# two ranks, where the iteration count is still 6, so the count was never a sensitive detector.
# The absolute sum grows monotonically -- 2430517 / 2503759 / 2677077 / 2951307 at 1 / 2 / 4 /
# 8, up 21 percent -- which is the signature of contributions being counted more than once
# rather than merely reordered.
for n in 1 2 4 5 6 8; do
    run_arm "smoothsum" "$n" \
        "${COMMON} ${CAVITY} ${MGV} ${ONESTEP} SFEM_GMG_CGC=0 SFEM_SMOOTHER_APPLY_SUM=1"
done

# (1b) The same, additive instead of multiplicative. This splits the fault.
#
# The multiplicative sweep stays inside one macro-element and the corrections combine
# additively between elements, so neither variant should depend on the cut. If additive comes
# out flat and multiplicative does not, the fault is in the sweep; if both drift, it is in what
# they share -- the stencils, the patch weights, or the output side, where a rank holding a
# node as a GHOST sees only the subset of that node's macro-elements local to it and divides by
# a multiplicity computed from that subset, while nothing re-gathers the smoother's output
# before the next operator apply reads it.
for n in 1 2 4 5 6 8; do
    run_arm "smoothsum_add" "$n" \
        "${COMMON} ${CAVITY} ${MGV} ${ONESTEP} SFEM_GMG_CGC=0 SFEM_SMOOTHER_APPLY_SUM=1 \
         SFEM_VANKA_MULT=0"
done

# (2) The no-multigrid path, now wrapped. Was 434 / 443 / 509 / 535 at 1 / 2 / 4 / 8.
for n in 1 2 4 5 6 8; do
    run_arm "nomg_vanka" "$n" "${COMMON} ${CAVITY} SFEM_GMG=0 SFEM_PRECOND=vanka"
done

# (3) The full cycle, which is the number that has to come flat in the end.
for n in 1 2 4 5 6 8; do
    run_arm "full_vanka" "$n" "${COMMON} ${CAVITY} ${MGV}"
done

echo "raw logs: $RAW"
