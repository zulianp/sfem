#!/usr/bin/env bash
#SBATCH --account=c40
#SBATCH --job-name=cvfem-cgc-split
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=288
#SBATCH --time=00:30:00
#SBATCH --exclusive
#SBATCH --hint=nomultithread
#SBATCH --uenv-passthrough=ignore
#SBATCH --output=cgc_split_%j.out
#SBATCH --error=cgc_split_%j.err

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
OUT="${OUT:-$SCRATCH/cvfem_cgc_split/${SLURM_JOB_ID:-manual}}"
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


# Is the 8-rank weakness in the SMOOTHER or in the COARSE CORRECTION?
#
# Post-fix, multigrid + block-Jacobi is EXACTLY invariant at 1, 2 and 4 ranks -- the
# per-solve sequence is 5 11 21 6 14 12 7 11 9 8 9 10 at all three. At 8 ranks every solve
# is 3-4x worse from the very first one (18 25 53 32 23 30 27 26 27 53 47 29 119 79 59) with
# no solve hitting the cap, and the first Newton residual is identical (9.057110e-03), so
# the fine OPERATOR is invariant and the CYCLE is not.
#
# SFEM_GMG_CGC=0 scales the prolonged coarse correction to zero (the knob is read at
# cvfem_hex8_ns_ssgmg.cpp:2501 and applied in the prolongation wrapper at :2557-2563),
# leaving the smoother alone. Read:
#
#   CGC=0 makes n=8 match n=4  -> the coarse correction is the weak part, which is where the
#                                 Galerkin assembly reduces through a purely LOCAL table
#                                 (galerkin_build_node_reduce) with no cross-rank completion
#   CGC=0 still 3-4x at n=8    -> the smoother itself degrades, and the coarse operator is
#                                 not the explanation
#
# cavity L2 has 64 macro elements, so at 8 ranks the coarse level is roughly 8 elements over
# 8 ranks -- the regime an uncommunicated coarse assembly should hurt most.
for n in 4 8; do
    run_arm "full" "$n" "${COMMON} ${CAVITY} ${MG}"
    run_arm "cgc0" "$n" "${COMMON} ${CAVITY} ${MG} SFEM_GMG_CGC=0"
done

echo "raw logs: $RAW"
