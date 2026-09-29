#!/usr/bin/env bash
#SBATCH --account=c40
#SBATCH --job-name=cvfem-rung2
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=288
#SBATCH --time=00:30:00
#SBATCH --exclusive
#SBATCH --hint=nomultithread
#SBATCH --uenv-passthrough=ignore
#SBATCH --output=rung2_%j.out
#SBATCH --error=rung2_%j.err

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
OUT="${OUT:-$SCRATCH/cvfem_rung2/${SLURM_JOB_ID:-manual}}"
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

# ---- Rung 2 proper: the cycle's iteration count against rank count.
for n in 1 2 4 8; do
    run_arm "mgjac" "$n" "${COMMON} ${CAVITY} ${MG}"
done

# ---- The coarse operator, during setup. One Newton step is enough to print the comparison.
for n in 1 2 4; do
    run_arm "egal" "$n" \
        "${COMMON} ${CAVITY} ${MG} SFEM_GMG_CHECK=1 SFEM_NL_MAX_IT=1 SFEM_VERIFY_TOL=1e9"
done

# ---- Pure MPI at full node width, on a case with enough elements to partition.
BIGCAV="SFEM_CASE=cavity SFEM_N=8 SFEM_ELEMENT_REFINE_LEVEL=2"
for n in 1 288; do
    run_arm "wide" "$n" "${COMMON} ${BIGCAV} ${MG}"
done

echo "raw logs: $RAW"
