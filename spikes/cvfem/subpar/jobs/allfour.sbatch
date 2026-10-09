#!/usr/bin/env bash
#SBATCH --account=c40
#SBATCH --job-name=cvfem-allfour
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=288
#SBATCH --time=00:30:00
#SBATCH --exclusive
#SBATCH --hint=nomultithread
#SBATCH --uenv-passthrough=ignore
#SBATCH --output=allfour_%j.out
#SBATCH --error=allfour_%j.err

# FOUR DECOMPOSITION DEFECTS, ON CRAY-MPICH
#
# (1) THE COARSE GATHER. GlobalDenseLU::apply gathered the coarse right-hand side with an
#     allgatherv and no index translation, assuming a rank's owned coarse ids form the
#     contiguous block that tiling implies, while scattering the answer back through the
#     explicit local-to-global map. On a derefined coarse mesh they are not contiguous -- at
#     eight ranks, rank 1's first owned node has global id 15 where the tiling expects 30. The
#     right-hand side was assembled in the wrong order and the answer distributed in the right
#     one, so every rank agreed on the same wrong correction. Fixed by scattering each owned
#     value to its global index and reducing, which is what the matrix assembly in the same
#     function already does.
#
# (2) THE FINE VANKA PATCH ENTRIES. The fold in assemble_fine_stencil sums a node pair's
#     contributions over the macro-elements this rank holds. For a pair touching an owned node
#     that is the whole sum, because the one-deep aura makes every element touching an owned
#     node local. For a pair between two NON-owned nodes of an aura element it is not, and
#     those entries sit in the 8x8 block a patch inverts. The missing values all live on the
#     owners, so the completion is a GATHER of each node's 27-direction lattice row. The aura
#     is NOT changed.
#
# (3) THE LEVEL OPERATORS AND SMOOTHERS AT DEPTH. Multigrid asks for ParallelOperator by
#     dynamic_pointer_cast; when the cast fails, level_owned falls back to the LOCAL dof count
#     and reduce_norm2 skips its allreduce. Level 0 carries the face; with SFEM_GMG_GALERKIN=2
#     the coarser levels are replaced by element-matrix or assembled forms, and neither is a
#     ParallelOperator. The smoothers were additionally built on the RAW operator, so a
#     stationary sweep's A x never gathered its input. Both fixed by one wrap with
#     create_parallel_preconditioner placed before the smoother construction.
#
# (4) THE COARSE VANKA PATCH ENTRIES. galerkin_accumulate is rank-local for the same reason
#     the fine fold was, and make_diagonal_vanka_from_bsr reads rowptr[n[a]] for all eight
#     corners of a cell, including rows this rank does not own. Completed the same way, keyed
#     on THAT level's mesh, after level_to_bsr rather than before. Confirmed by disabling the
#     coarse Vanka branch entirely (SFEM_GMG_EGAL_EM=1): four levels then measured 4 / 4 / 4 at
#     1 / 4 / 8 ranks against 3 / 5 / 14 with it.
#
# Also: the coarsest level now carries the face, for the DIAGNOSTICS rather than the solve.
# Without it || r_H || is formed over ghost rows of an incomplete BSR with no allreduce, and
# reported 5.1e-03 at eight ranks in a configuration whose iteration count was flat -- an
# artifact that would read as a broken coarse solve. It is now at round-off on both sides.
#
# MEASURED LOCALLY, every arm matching its own serial value:
#
#     2 levels, full solve      bjacobi 158     vanka 61
#     4 levels, one step        bjacobi  43     vanka  3
#     stencilsum at depth       sum 31.024516703735571   abs 443.41329064263664
#     || r_H ||                 round-off at 1 and 8 ranks
#
# READ, committed before the run:
#
#   every arm equal to its own serial value -> the four fixes hold on cray-mpich
#   a deep vanka arm drifting               -> the coarse completion does not reproduce here;
#                                              the stencilsum and face-check lines in the same
#                                              arm separate the assembly from the bookkeeping
#   a two-level arm moving                  -> a regression from the depth work, and that comes
#                                              first
#
# A TIMEOUT is unmeasured, not a result. A count at the iteration cap did NOT converge and is a
# floor, not a measurement -- check newton_converged before quoting one.

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
OUT="${OUT:-$SCRATCH/cvfem_allfour/${SLURM_JOB_ID:-manual}}"
RAW="$OUT/raw"
mkdir -p "$RAW"
[ -x "$BIN" ] || { echo "error: no binary at $BIN" >&2; exit 1; }

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
        | stdbuf -oL grep -aE "^gmgface|^stencilsum|lin_it_total|newton_converged|gmg: [0-9]+ levels"
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
SHALLOW="SFEM_CASE=cavity SFEM_N=4 SFEM_ELEMENT_REFINE_LEVEL=2"
DEEP="SFEM_CASE=cavity SFEM_N=4 SFEM_ELEMENT_REFINE_LEVEL=8"
ONESTEP="SFEM_NL_MAX_IT=1 SFEM_VERIFY_TOL=1e9"
MG="SFEM_GMG=1 SFEM_GMG_GALERKIN=2"

# (1) Two levels, full solve. Baselines 158 and 61.
for n in 1 2 4 5 6 8; do
    run_arm "flat_jac" "$n" "${COMMON} ${SHALLOW} ${MG} SFEM_SMOOTHER=bjacobi"
done
for n in 1 2 4 5 6 8; do
    run_arm "flat_vanka" "$n" "${COMMON} ${SHALLOW} ${MG} SFEM_SMOOTHER=vanka"
done

# (2) Four levels, one Newton step. Baselines 43 and 3.
for n in 1 2 4 5 6 8; do
    run_arm "deep_jac" "$n" "${COMMON} ${DEEP} ${ONESTEP} ${MG} SFEM_SMOOTHER=bjacobi SFEM_GMG_FACE_CHECK=1"
done
for n in 1 2 4 5 6 8; do
    run_arm "deep_vanka" "$n" \
        "${COMMON} ${DEEP} ${ONESTEP} ${MG} SFEM_SMOOTHER=vanka \
         SFEM_GMG_FACE_CHECK=1 SFEM_VANKA_STENCIL_SUM=1"
done

echo "raw logs: $RAW"
