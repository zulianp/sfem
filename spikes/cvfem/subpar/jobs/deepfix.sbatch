#!/usr/bin/env bash
#SBATCH --account=c40
#SBATCH --job-name=cvfem-deepfix
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=288
#SBATCH --time=00:30:00
#SBATCH --exclusive
#SBATCH --hint=nomultithread
#SBATCH --uenv-passthrough=ignore
#SBATCH --output=deepfix_%j.out
#SBATCH --error=deepfix_%j.err

# THREE DEFECTS, ON CRAY-MPICH
#
# (1) THE COARSE GATHER. GlobalDenseLU::apply gathered the coarse right-hand side with an
#     allgatherv and no index translation, assuming a rank's owned coarse ids are the
#     contiguous block the tiling implies, while scattering the answer back through the
#     explicit local-to-global map. On a derefined coarse mesh they are not contiguous. Fixed
#     by scattering each owned value to its global index and reducing -- what the matrix
#     assembly in the same function already does. Two-level multigrid with block-Jacobi went
#     from 158 / 158 / 158 / 726 at 1 / 2 / 4 / 8 ranks to 158 everywhere.
#
# (2) THE VANKA PATCH ENTRIES. The fold in assemble_fine_stencil sums a node pair's
#     contributions over the macro-elements this rank holds. For a pair touching an owned node
#     that is the whole sum, because the one-deep aura makes every element touching an owned
#     node local. For a pair between two NON-owned nodes of an aura element it is not, and
#     those entries sit in the 8x8 block a patch inverts. The missing values all live on the
#     owners, so the completion is a GATHER of each node's 27-direction lattice row, not a
#     scatter_add. The aura is NOT changed. stencilsum went from abs 81.72 falling to 75.45
#     across rank counts to bit-identical with serial everywhere.
#
# (3) THE LEVEL OPERATORS AT DEPTH. Multigrid asks for ParallelOperator by
#     dynamic_pointer_cast; when the cast fails, level_owned falls back to the LOCAL dof count
#     and reduce_norm2 skips its allreduce. Level 0 carries the face, but with
#     SFEM_GMG_GALERKIN=2 the coarser levels are replaced by the element-matrix or assembled
#     form, and neither is a ParallelOperator. Measured at four levels, two ranks: level 1
#     reported 2916 rows against 1620 owned, level 2 reported 500 against 300. The smoothers
#     were additionally built on the RAW operator, so a stationary sweep's A x never gathered
#     its input. Both are fixed by one wrap with create_parallel_preconditioner, placed before
#     the smoother construction so operator and smoother share it.
#
#     A two-level hierarchy cannot show this: its only smoothed level is the fine one, which
#     has the face. At four levels the count was 8 / 19 / 17 at 1 / 2 / 4 ranks, 8 / 13 / 19
#     with the face alone, and 8 / 8 / 8 with both.
#
# MEASURED LOCALLY, all matching serial:
#
#     2 levels  bjacobi  158 at 1 2 4 5 6 8      vanka 61 at 1 2 4 5 6 8
#     4 levels  bjacobi    8 at 1 2 4            cgc0 22 at 1 2 4
#     stencilsum  sum 9.9143272486398928  abs 81.721892532339552  at 1 2 4 5 6 8
#
# READ, committed before the run:
#
#   every arm equal to its own serial value  -> the three fixes hold on cray-mpich
#   a deep arm drifting                      -> something remains at depth; the face check
#                                               output in the same arm says whether the cast
#                                               succeeded and whether level_owned matches
#                                               true_owned, which separates bookkeeping from
#                                               the operator itself
#   a two-level arm moving                   -> a regression from the depth work, and that
#                                               comes first
#
# A TIMEOUT is unmeasured, not a result. A count at the iteration cap did NOT converge and is
# a floor, not a measurement -- check newton_converged before quoting one.

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
OUT="${OUT:-$SCRATCH/cvfem_deepfix/${SLURM_JOB_ID:-manual}}"
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

# (1) Two levels, full solve. The established baselines: 158 and 61.
for n in 1 2 4 5 6 8; do
    run_arm "flat_jac" "$n" "${COMMON} ${SHALLOW} SFEM_GMG=1 SFEM_GMG_GALERKIN=2 SFEM_SMOOTHER=bjacobi"
done
for n in 1 2 4 5 6 8; do
    run_arm "flat_vanka" "$n" "${COMMON} ${SHALLOW} SFEM_GMG=1 SFEM_GMG_GALERKIN=2 SFEM_SMOOTHER=vanka"
done

# (2) Four levels, one Newton step, both smoothers, with the face check riding along.
for n in 1 2 4 5 6 8; do
    run_arm "deep_jac" "$n" \
        "${COMMON} ${DEEP} ${ONESTEP} SFEM_GMG=1 SFEM_GMG_GALERKIN=2 SFEM_SMOOTHER=bjacobi \
         SFEM_GMG_FACE_CHECK=1"
done
for n in 1 2 4 5 6 8; do
    run_arm "deep_vanka" "$n" \
        "${COMMON} ${DEEP} ${ONESTEP} SFEM_GMG=1 SFEM_GMG_GALERKIN=2 SFEM_SMOOTHER=vanka \
         SFEM_GMG_FACE_CHECK=1 SFEM_VANKA_STENCIL_SUM=1"
done

echo "raw logs: $RAW"
