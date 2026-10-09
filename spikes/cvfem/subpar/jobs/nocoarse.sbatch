#!/usr/bin/env bash
#SBATCH --account=c40
#SBATCH --job-name=cvfem-nocoarse
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=288
#SBATCH --time=00:25:00
#SBATCH --exclusive
#SBATCH --hint=nomultithread
#SBATCH --uenv-passthrough=ignore
#SBATCH --output=nocoarse_%j.out
#SBATCH --error=nocoarse_%j.err

# WITHOUT A COARSE SPACE, IS THE ITERATION COUNT THE SAME AT EVERY RANK COUNT --
# FOR VANKA AS WELL AS FOR JACOBI?
#
# What is already measured without a coarse space:
#
#   FGMRES + block-Jacobi, no multigrid    812 / 812 / 812 / 811  at 1 / 2 / 4 / 8
#   FGMRES + MG, Jacobi, coarse cgc = 0     13 /  13 /  13 /  13  (one Newton step)
#   the same, full solve                          320 / 320       at 4 / 8
#
# Flat, except for 811 against 812 at eight ranks. That is not exact, and the reason is not
# the operator: block-Jacobi inverts one 4x4 block per node, so its owned output depends only
# on owned input and it cannot be partition dependent. It is the Krylov method -- the dots and
# norms reassociate when the domain is cut differently, so a run near the tolerance boundary
# can cross it one iteration early. Exact invariance is a property of the operator; the count
# is invariant to about 0.1 percent.
#
# VANKA HAS NEVER BEEN MEASURED, at any rank count, and must not be assumed to behave like
# Jacobi. It inverts patches that span macro-element boundaries, it needs complete matrix rows,
# and it needs multiplicity weights counted GLOBALLY rather than per rank. Being additive
# across those boundaries, an error there degrades the rate rather than the answer -- which is
# exactly how a rank-dependent iteration count appears. So the expectation here is the opposite
# of the expectation for Jacobi, and this job is what settles it.
#
# Rank counts run to eight because the partition changes character at five: with 64 macro
# elements in a 4x4x4 lattice, cuts into one to four ranks land on whole 16-element layers, and
# from five on they fall inside a layer. Every coarse-space defect found so far appears exactly
# at that transition, so a smoother defect would plausibly do the same and would be missed by a
# sweep of powers of two alone.
#
# READ, committed before the run:
#
#   vanka flat 1..8            -> the smoother is decomposition invariant and the coarse space
#                                 remains the sole defect
#   vanka degrades from n=5    -> the patch weights or the patch rows are incomplete across
#                                 ranks, and that is a SECOND defect, independent of the
#                                 coarse solve, which must be fixed before any Vanka result at
#                                 scale means anything
#   vanka errors out with no MG -> that configuration is unsupported; the cgc=0 arm is then the
#                                 measurement that counts, since it exercises Vanka as a
#                                 smoother with the coarse correction contributing nothing
#
# A TIMEOUT is unmeasured, not a result.

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
OUT="${OUT:-$SCRATCH/cvfem_nocoarse/${SLURM_JOB_ID:-manual}}"
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
        | stdbuf -oL grep -aE "lin_it_total|newton_converged|preconditioner:|error|Error|not support"
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
ONESTEP="SFEM_NL_MAX_IT=1 SFEM_VERIFY_TOL=1e9"

# (a) The reference: Jacobi smoother, coarse correction contributing nothing.
for n in 1 2 4 5 6 8; do
    run_arm "jac_cgc0" "$n" \
        "${COMMON} ${CAVITY} ${ONESTEP} SFEM_GMG=1 SFEM_GMG_GALERKIN=2 \
         SFEM_SMOOTHER=bjacobi SFEM_GMG_CGC=0"
done

# (b) The question: Vanka smoother, coarse correction contributing nothing.
for n in 1 2 4 5 6 8; do
    run_arm "vanka_cgc0" "$n" \
        "${COMMON} ${CAVITY} ${ONESTEP} SFEM_GMG=1 SFEM_GMG_GALERKIN=2 \
         SFEM_SMOOTHER=vanka SFEM_GMG_CGC=0"
done

# (c) No multigrid in the picture at all, both preconditioners, full solve.
#
# These arms drive SFEM_PRECOND, not SFEM_SMOOTHER. SFEM_SMOOTHER selects the smoother INSIDE
# the cycle and is read nowhere on the no-multigrid path, which takes its preconditioner from
# SFEM_PRECOND with bjacobi as the default. Setting SFEM_SMOOTHER here would have run
# block-Jacobi in both loops and reported one of them as Vanka.
for n in 1 2 4 8; do
    run_arm "nomg_jac" "$n" "${COMMON} ${CAVITY} SFEM_GMG=0 SFEM_PRECOND=bjacobi"
done
for n in 1 2 4 8; do
    run_arm "nomg_vanka" "$n" "${COMMON} ${CAVITY} SFEM_GMG=0 SFEM_PRECOND=vanka"
done

echo "raw logs: $RAW"
