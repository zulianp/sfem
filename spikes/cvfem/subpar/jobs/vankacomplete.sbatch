#!/usr/bin/env bash
#SBATCH --account=c40
#SBATCH --job-name=cvfem-vankacomplete
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=288
#SBATCH --time=00:25:00
#SBATCH --exclusive
#SBATCH --hint=nomultithread
#SBATCH --uenv-passthrough=ignore
#SBATCH --output=vankacomplete_%j.out
#SBATCH --error=vankacomplete_%j.err

# THE CROSS-RANK COMPLETION OF THE VANKA PATCH ENTRIES, ON CRAY-MPICH
#
# THE DEFECT. The fold in assemble_fine_stencil sums a node pair's contributions over the
# macro-elements THIS RANK holds. For a pair involving an owned node that is the whole sum,
# because the one-deep aura makes every element touching an owned node local. For a pair between
# two NON-owned nodes of an aura element it is not -- and those entries sit in the 8x8 block a
# patch inverts to produce the correction at an owned node. Block-Jacobi is immune because it
# reads only a node's own 4x4 block and never a neighbour's row.
#
# THE FIX. The missing values all live on the owners, so the completion is a GATHER, not a
# scatter_add: there is nothing to accumulate first. Each node's lattice row -- 27 directions x
# 16 -- is packed from the elements that carry each direction, gathered over GhostsAndAura so
# every ghost and aura copy is replaced by its owner's complete row, and unpacked. The aura is
# NOT touched.
#
# The 27-slot direction is a global key only if adjacent macro-elements agree on what a
# direction means. SFEM_VANKA_SLOT_CHECK=1 verifies it and it holds on every case the spike runs
# -- cavity, cavity_reg, step, poiseuille, couette, about two million pairs, no disagreement --
# which is what allows one fixed-width exchange instead of shipping column ids with the values.
# It stays in the code as a guard, because a mesh with inconsistently ordered parent hexes would
# fail silently and wrongly.
#
# MEASURED LOCALLY AFTER THE FIX, all bit-identical to serial:
#
#     stencilsum   sum 9.9143272486398928   abs 81.721892532339552   at 1 2 4 5 6 8
#     smoothsum    abs 2430517.49479465xx                            at 1 2 4 5 6 8
#     full solve, vanka     61  61  61  61      at 1 2 4 8  (was 61 61 62 12245)
#     full solve, bjacobi  158 158 158 158      unchanged, the regression gate
#
# WHAT THIS DOES NOT COVER. A two-level hierarchy solves level 1 with the dense LU, so the
# coarse-level Vanka branch never fires and make_diagonal_vanka_from_bsr -- which reads
# rowptr[n[a]] for all eight patch corners and therefore carries the SAME defect -- is not
# exercised here at all. The deep arm below exists to reach it.
#
# READ, committed before the run:
#
#   stencilsum and smoothsum equal to serial at every rank count -> the completion holds on
#                                                                   cray-mpich
#   full vanka 61 everywhere                                     -> the fine path is invariant
#   full bjacobi not 158                                         -> a regression; that comes
#                                                                   first, since bjacobi never
#                                                                   enters assemble_fine_stencil
#   deep arm drifting while the shallow one is flat              -> the BSR path is the
#                                                                   remaining instance, as
#                                                                   expected, and is next
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
OUT="${OUT:-$SCRATCH/cvfem_vankacomplete/${SLURM_JOB_ID:-manual}}"
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
        | stdbuf -oL grep -aE "^stencilsum|^smoothsum |^vankaslot|lin_it_total|newton_converged|gmg: [0-9]+ levels"
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
MGV="SFEM_GMG=1 SFEM_GMG_GALERKIN=2 SFEM_SMOOTHER=vanka"
MGJ="SFEM_GMG=1 SFEM_GMG_GALERKIN=2 SFEM_SMOOTHER=bjacobi"
ONESTEP="SFEM_NL_MAX_IT=1 SFEM_VERIFY_TOL=1e9"

# (1) The completed stencils and the smoother they produce.
for n in 1 2 4 5 6 8; do
    run_arm "sten" "$n" \
        "${COMMON} ${CAVITY} ${MGV} ${ONESTEP} SFEM_GMG_CGC=0 \
         SFEM_VANKA_STENCIL_SUM=1 SFEM_SMOOTHER_APPLY_SUM=1 SFEM_VANKA_SLOT_CHECK=1"
done

# (2) The full cycle with Vanka: the number that has to come flat.
for n in 1 2 4 5 6 8; do
    run_arm "full_vanka" "$n" "${COMMON} ${CAVITY} ${MGV}"
done

# (3) The regression gate. Block-Jacobi never enters assemble_fine_stencil.
for n in 1 2 4 8; do
    run_arm "full_jac" "$n" "${COMMON} ${CAVITY} ${MGJ}"
done

echo "raw logs: $RAW"
