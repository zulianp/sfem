#!/usr/bin/env bash
#SBATCH --account=c40
#SBATCH --job-name=cvfem-gmres-inv
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --time=00:30:00
#SBATCH --exclusive
#SBATCH --hint=nomultithread
#SBATCH --uenv-passthrough=ignore
#SBATCH --output=gmres_inv_%j.out
#SBATCH --error=gmres_inv_%j.err

# SHOULD GMRES NOT TAKE THE SAME NUMBER OF ITERATIONS AT EVERY RANK COUNT?
#
# Yes. On a fixed operator with a fixed preconditioner the Krylov space is determined by A
# and b alone, so the iteration count must be independent of how the domain was cut, up to
# floating-point reassociation in the reductions. Block-Jacobi inverts one 4x4 block per
# node, so its owned output depends only on owned input and it cannot introduce partition
# dependence. A count that moves with rank count therefore means the OPERATOR differs, not
# the Krylov method.
#
# An earlier measurement showed the basis staying orthogonal to rounding across ranks
# (orth_worst 1.2e-14 .. 1.05e-13 at two ranks against 1.96e-14 .. 2.99e-13 serial), which
# proves the Gram-Schmidt reductions are collective and correct. That is a weaker statement
# than invariance: a wrong matvec still produces a perfectly orthogonal basis, just for a
# different Krylov space. This job tests the operator instead.
#
# THE SUSPECT is the plan's finding 5. sscvfem_apply reconstructs a NODAL gradient of the
# search direction (sscvfem_nodal_q_grad, cvfem_sshex8_ns.hpp:3109) and the macro sweep then
# consumes it (:3127). The parallel wrapper gathers x once before calling apply and never
# exchanges that reconstructed field. The aura makes every element touching an OWNED node
# local, so owned reconstructions are complete -- but GHOST node reconstructions are not,
# and the sweep reads the nodal gradient at ghost nodes. So A*x on owned rows depends on
# incompletely reconstructed ghost values, and the operator is decomposition dependent.
#
# SFEM_RC_EXACT_JAC=0 is the control the operator already provides: it restores the frozen
# Rhie-Chow Jacobian and skips the reconstruction entirely (the call at :1892 is gated on
# `exact`). No code change is needed to settle this.
#
# PREDICTION, recorded before the run:
#   RC_EXACT_JAC=1  -> per-solve counts drift with rank count (the reconstruction is not
#                      exchanged, so the operator differs)
#   RC_EXACT_JAC=0  -> per-solve counts match serial to within a couple of iterations
#                      (reassociation only)
# If BOTH drift, the missing exchange is not the cause and the matvec must be compared
# directly against a coordinate-defined direction instead.
#
# DEGENERATE SOLVES ARE REMOVED, because they dominated the totals and hid this. Once Newton
# has converged, ||R|| reaches round-off and the test resid/bnorm < rtol asks for a residual
# below the absolute floor atol, so the solve cannot exit and burns its whole cap. Measured:
# capped right-hand sides spanned 1.6e-10 .. 5.6e-06 while the smallest non-capped one was
# 2.2e-03 -- no overlap. Raising atol lets those exit immediately, leaving only productive
# solves, which are the ones the invariance question is about.
#
# NO MULTIGRID and block-Jacobi: the simplest configuration that still exercises the CVFEM
# operator, so a difference can only come from the matvec or the preconditioner, and the
# preconditioner is provably partition independent.
#
# SINGLE-THREADED: pure MPI, so atomics and thread scheduling cannot explain a difference.

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
OUT="${OUT:-$SCRATCH/cvfem_gmres_inv/${SLURM_JOB_ID:-manual}}"
RAW="$OUT/raw"
mkdir -p "$RAW"
[ -x "$BIN" ] || { echo "error: no binary at $BIN" >&2; exit 1; }

# Per-arm wall budget. A hung arm must not consume the allocation and strand every arm
# behind it; an arm that hits the cap is reported as TIMEOUT and the sweep continues.
ARM_TIMEOUT="${ARM_TIMEOUT:-300}"

run_arm() {
    local tag="$1" nranks="$2"; shift 2
    echo "### ${tag} ranks=${nranks} ###"
    timeout --signal=TERM --kill-after=30 "${ARM_TIMEOUT}" \
        srun -u --ntasks="${nranks}" --cpus-per-task=1 --kill-on-bad-exit=1 \
            --time="00:$(printf '%02d' $(( (ARM_TIMEOUT + 59) / 60 ))):00" \
            --uenv="$UENV_IMAGE" --view="$UENV_VIEW" \
            bash -lc "$* stdbuf -oL ${BIN} ${OUT}/${tag}_n${nranks}" 2>&1 \
        | tee "$RAW/${tag}_n${nranks}.log" \
        | stdbuf -oL grep -aE "^fgmres:|^newton |lin_it:|newton_converged|lin_it_total|nnodes:"
    local rc=${PIPESTATUS[0]}
    if [ "$rc" = "124" ] || [ "$rc" = "137" ]; then
        echo "  (arm exit: ${rc} -- TIMEOUT after ${ARM_TIMEOUT}s)"
    else
        echo "  (arm exit: ${rc})"
    fi
    echo
}

COMMON="OMP_NUM_THREADS=1 SFEM_FGMRES=1 SFEM_FGMRES_DIAG=1 SFEM_ENABLE_OUTPUT=0 \
        SFEM_PARALLEL_MF_LEGACY=1 SFEM_ALLOW_MPI=1 \
        SFEM_LSOLVE_MAX_IT=4000 SFEM_LSOLVE_ATOL=1e-12 \
        SFEM_CASE=cavity SFEM_N=4 SFEM_ELEMENT_REFINE_LEVEL=2 \
        SFEM_GMG=0 SFEM_SMOOTHER=bjacobi"

# Exact Rhie-Chow Jacobian: the nodal reconstruction runs and is never exchanged.
for n in 1 2 4 8; do
    run_arm "exact" "$n" "${COMMON} SFEM_RC_EXACT_JAC=1"
done

# Frozen Rhie-Chow Jacobian: no reconstruction inside the apply at all.
for n in 1 2 4 8; do
    run_arm "frozen" "$n" "${COMMON} SFEM_RC_EXACT_JAC=0"
done

echo "raw logs: $RAW"
