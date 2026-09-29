#!/usr/bin/env bash
#SBATCH --account=c40
#SBATCH --job-name=cvfem-egal-coord
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=288
#SBATCH --time=00:30:00
#SBATCH --exclusive
#SBATCH --hint=nomultithread
#SBATCH --uenv-passthrough=ignore
#SBATCH --output=egal_coord_%j.out
#SBATCH --error=egal_coord_%j.err

# RUNG 2: IS MULTIGRID + BLOCK-JACOBI DECOMPOSITION INVARIANT NOW?
#
# Block-Jacobi inverts one 4x4 block per node, so its owned output depends only on owned
# input and it cannot be partition dependent; multigrid built on it inherits that. The
# iteration count is therefore expected to be IDENTICAL at every rank count, and a deviation
# is a defect rather than tuning.
#
# It was not. Measured on the cavity at level 2, single-threaded:
#
#     multigrid + block-Jacobi     1 rank 162    2 ranks 6104    4 ranks 10067
#
# The cause was the same one that broke rung 1: sscvfem_apply reconstructs a NODAL gradient
# (sscvfem_nodal_q_grad) and the macro sweep consumes it, but nothing exchanged that
# reconstructed field, so ghost and aura nodes held a partial numerator over a partial
# denominator and A*x depended on how the domain was cut. Rung 1 is now flat -- no
# multigrid, block-Jacobi, 812 / 812 / 812 / 811 at 1 / 2 / 4 / 8 ranks against
# 812 / 816 / 4765 / 52186 before the exchange was added.
#
# This job asks whether the cycle above it is flat too. The transfers and the coarse
# operator were never touched, so if rung 2 is still rank dependent the fault is in the
# hierarchy rather than in the fine operator, and that is a different investigation.
#
# THE COARSE OPERATOR rides along in the same run. The element-wise Galerkin assembly reads
# the same reconstructed pgx (cvfem_ss_galerkin.hpp:262 feeds it to Hex8RhieChow), and its
# error against its own probed composite grew with rank count:
#
#     egal level 1 vs probed composite:   1 rank 3.94e-02   2 ranks 3.56e-01   4 ranks 1.0179
#
# At four ranks the coarse operator was 100 percent wrong relative to the composite it is
# supposed to reproduce. If that was the unexchanged reconstruction, these numbers collapse
# to the serial value; if they do not, the coarse operator has an independent defect. The
# check arms run one Newton step only, because SFEM_GMG_CHECK=1 prints the comparison during
# setup and the solve after it is not needed.
#
# THE 288-RANK ARM is the original request: no OpenMP, one rank per core, same iteration
# count as serial. It runs on cavity N=8 (512 macro elements, 4913 nodes) rather than level
# 2's 64 elements, because the distributed cube guard refuses a mesh with fewer elements
# than ranks.
#
# Per-arm timeout, because a hung arm must not consume the allocation and strand the arms
# queued behind it. An arm that hits the cap is reported as TIMEOUT and the sweep continues.

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
OUT="${OUT:-$SCRATCH/cvfem_egal_coord/${SLURM_JOB_ID:-manual}}"
RAW="$OUT/raw"
mkdir -p "$RAW"
[ -x "$BIN" ] || { echo "error: no binary at $BIN" >&2; exit 1; }

ARM_TIMEOUT="${ARM_TIMEOUT:-420}"

run_arm() {
    local tag="$1" nranks="$2"; shift 2
    echo "### ${tag} ranks=${nranks} ###"
    timeout --signal=TERM --kill-after=30 "${ARM_TIMEOUT}" \
        srun -u --ntasks="${nranks}" --cpus-per-task=1 --kill-on-bad-exit=1 \
            --time="00:$(printf '%02d' $(( (ARM_TIMEOUT + 59) / 60 ))):00" \
            --uenv="$UENV_IMAGE" --view="$UENV_VIEW" \
            bash -lc "$* stdbuf -oL ${BIN} ${OUT}/${tag}_n${nranks}" 2>&1 \
        | tee "$RAW/${tag}_n${nranks}.log" \
        | stdbuf -oL grep -aE "^fgmres:|newton_converged|lin_it_total|nnodes:|egal level|egal diag|coarse LU"
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


# Re-measure the coarse-operator comparison with a COORDINATE-DEFINED probe direction.
#
# The previous numbers (egal level 1: 3.9358e-02 / 2.5192e-01 / 1.0091e+00 at 1/2/4 ranks)
# were taken with v[k] = sin(0.37 k + 1.1) over the LOCAL dof index. That is not a
# distributed vector: a node carries one local index on its owner and a different one in
# every rank holding it as a ghost, so the same physical node entered the two operators with
# different values and the vector changed shape with the decomposition. The growth with rank
# count could therefore track the rising ghost fraction rather than the operator, and the
# claim that the coarse operator degrades was withdrawn pending this run.
#
# The probe is now sin(2.3x + 1.7y + 1.1z + 0.4c + 1.1) over the node's coordinates, so a
# ghost copy agrees with its owner by construction and every rank count probes the same
# direction. Serial moved 3.9358e-02 -> 2.7138e-02 purely from probing a different
# direction, which is expected and is not a fix.
#
# READ: the level-1 line printed AFTER "newton 0" is the meaningful one; the pair printed
# before "stage 1/5" is the pre-Newton setup, where the state is boundary data alone.
#
#   rel flat across 1/2/4   -> the coarse operator is fine and the earlier growth was the
#                              inconsistent probe; the cgc0 split then needs another cause
#   rel still grows         -> the coarse operator really does degrade, and the missing
#                              cross-rank completion in galerkin_apply is the candidate
for n in 1 2 4; do
    run_arm "egalc" "$n" \
        "${COMMON} ${CAVITY} ${MG} SFEM_GMG_CHECK=1 SFEM_NL_MAX_IT=1 SFEM_VERIFY_TOL=1e9"
done

echo "raw logs: $RAW"
