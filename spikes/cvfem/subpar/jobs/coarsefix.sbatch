#!/usr/bin/env bash
#SBATCH --account=c40
#SBATCH --job-name=cvfem-coarsefix
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=288
#SBATCH --time=00:25:00
#SBATCH --exclusive
#SBATCH --hint=nomultithread
#SBATCH --uenv-passthrough=ignore
#SBATCH --output=coarsefix_%j.out
#SBATCH --error=coarsefix_%j.err

# DOES THE COARSE-GATHER FIX HOLD ON CRAY-MPICH, AND ON THE FULL SOLVE?
#
# THE DEFECT. GlobalDenseLU::apply gathered the coarse right-hand side with an allgatherv and
# no index translation, on the assumption that a rank's owned global ids are the contiguous
# block [displs[rank], displs[rank] + counts[rank]), while the solution was scattered back
# through the explicit local-to-global map. On a derefined coarse mesh that assumption is
# false. With 64 macro elements in a 4x4x4 lattice, cuts into one, two or four ranks land on
# whole layers and every rank's ids are contiguous; five, six and eight ranks leave several
# ranks whose ids are not -- at eight, rank 1's first owned node has global id 15 where the
# tiling expects 30. The right-hand side was assembled in the wrong order and the answer
# distributed in the right one, so every rank agreed on the same wrong coarse correction:
# consistent across ranks, wrong against serial, invisible to any rank-to-rank comparison.
#
# The number of ranks with non-contiguous ids predicted the damage exactly:
#
#     ranks         1    2    4    5    6    8
#     violations    0    0    0    3    5    2
#     lin_it        5    5    5   54  210   18
#
# THE FIX. Scatter each owned value to its global index and reduce, which is what the matrix
# assembly in the same function already does with the owned block rows, on the same
# disjointness argument -- every global entry is written by exactly one rank, so the reduction
# is an assembly and not an average. Measured locally after the fix: 5 linear iterations at
# 1, 2, 4, 5, 6 and 8 ranks, and the coarse solve's owned-range checksum agreeing to thirteen
# figures at every one of them.
#
# This job is that result on cray-mpich rather than local OpenMPI, which is smoke-test tier,
# and on the FULL solve rather than the one-step reproducer. The full solve was 158 / 158 /
# 158 / 726 at 1 / 2 / 4 / 8 before the fix; the expectation now is 158 at every rank count,
# and the rank counts that never divided evenly are included because they are where the defect
# actually lived.
#
# THE SECOND DEFECT, measured and unfixed, carried here to size it. With no coarse space at
# all, block-Jacobi is flat and Vanka is not:
#
#     MG + bjacobi, cgc=0    13  13  13  13  13  13   at 1 2 4 5 6 8
#     MG + vanka,   cgc=0     6   6   6   8  10  14
#     no MG, bjacobi        812 812 812           811 at 1 2 4 8
#     no MG, vanka          434 443 509           535
#
# Vanka should be invariant: its multiplicative sweep stays inside one macro-element on a
# private local vector and the corrections combine additively between elements, so with whole
# macro-elements per rank no sweep reads what another rank just wrote. It is not invariant, so
# that is a defect to find. It cannot be the coarse solve, which contributes nothing at cgc=0
# and does not exist at all without multigrid.
#
# READ, committed before the run:
#
#   bjacobi flat at every rank count, full solve and one step -> the coarse gather fix holds
#   any bjacobi arm still moving                              -> the fix is incomplete and the
#                                                                coarse path owes more than the
#                                                                gather
#   vanka still growing                                       -> as expected; the second defect
#                                                                is untouched by this fix, and
#                                                                sizing it here is the point

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
OUT="${OUT:-$SCRATCH/cvfem_coarsefix/${SLURM_JOB_ID:-manual}}"
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
        | stdbuf -oL grep -aE "^solvesum |lin_it_total|newton_converged|preconditioner:"
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
MGJ="SFEM_GMG=1 SFEM_GMG_GALERKIN=2 SFEM_SMOOTHER=bjacobi"
MGV="SFEM_GMG=1 SFEM_GMG_GALERKIN=2 SFEM_SMOOTHER=vanka"
ONESTEP="SFEM_NL_MAX_IT=1 SFEM_VERIFY_TOL=1e9"

# (1) The one-step reproducer the fix was validated against, on cray-mpich. Expect 5 everywhere.
for n in 1 2 4 5 6 8; do
    run_arm "step1_jac" "$n" "${COMMON} ${CAVITY} ${MGJ} ${ONESTEP} SFEM_GMG_SOLVE_OWNED_SUM=1"
done

# (2) The full solve, which is the claim that matters. Expect 158 everywhere.
for n in 1 2 4 5 6 8; do
    run_arm "full_jac" "$n" "${COMMON} ${CAVITY} ${MGJ}"
done

# (3) The second defect, sized on the full solve with the coarse path now correct.
for n in 1 2 4 5 6 8; do
    run_arm "full_vanka" "$n" "${COMMON} ${CAVITY} ${MGV}"
done

echo "raw logs: $RAW"
