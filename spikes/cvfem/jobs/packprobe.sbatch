#!/usr/bin/env bash
#SBATCH --account=c40
#SBATCH --job-name=cvfem-packprobe
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --time=00:25:00
#SBATCH --hint=nomultithread
#SBATCH --uenv-passthrough=ignore
#SBATCH --output=packprobe_%j.out
#SBATCH --error=packprobe_%j.err

# WHICH ASSEMBLED OBJECT IS WRONG WHEN THERE IS MORE THAN ONE PACK?
#
# The bisect (job 4719330, cavity N=8 refine 4, 512 elements, SERIAL) located the layer:
#
#     no hierarchy        308 /  308 / 308     <- base operator apply is CLEAN
#     MG + block-Jacobi    11 / 2392 /  11     <- catastrophic, and block-Jacobi cannot itself
#                                                 be pack-dependent: it inverts one 4x4 block
#                                                 per node, so it is reading a WRONG MATRIX
#     MG + Vanka frozen RC  7 /   13 /   7
#     MG + Vanka            7 /   14 /   7
#
# So the fault is in ASSEMBLY, not in the matvec and not in a smoother. This is the same shape
# as the four decomposition defects already fixed in this branch -- a fold that is local to a
# partition and never completed across it -- one level down: pack-local rather than rank-local.
#
# The probes below are the ones that caught those rank-local folds. Pointing them at pack count
# instead of rank count should say WHICH object is short, with no source reading:
#
#   stencilsum   fine Vanka stencils, keyed on owned elements
#   coarsemat    coarse BSR values per level
#   weightsum    Vanka patch multiplicity weights
#
# READ, committed before the run:
#
#   stencilsum differs at pk=256  -> the fine stencil fold is pack-local
#   coarsemat differs             -> the coarse BSR fold is pack-local (and block-Jacobi reads
#                                    exactly these diagonals, which would explain 2392)
#   weightsum differs             -> multiplicity is counted per pack
#   all three identical           -> the corruption is further down, in what the fold READS,
#                                    and the next cut is the element matrices themselves
#
# pk=512 is the control: one pack, must match pk=0 atomic everywhere.

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
OUT="${OUT:-$SCRATCH/cvfem_packprobe/${SLURM_JOB_ID:-manual}}"
RAW="$OUT/raw"
mkdir -p "$RAW"
[ -x "$BIN" ] || { echo "error: no binary at $BIN" >&2; exit 1; }

COMMON="OMP_NUM_THREADS=72 OMP_PROC_BIND=true SFEM_FGMRES=1 SFEM_ENABLE_OUTPUT=0 \
        SFEM_PARALLEL_MF_LEGACY=1 SFEM_ALLOW_MPI=1 \
        SFEM_LSOLVE_MAX_IT=4000 SFEM_LSOLVE_RTOL=1e-4 SFEM_LSOLVE_ATOL=1e-14"
CASE="SFEM_CASE=cavity SFEM_N=8 SFEM_ELEMENT_REFINE_LEVEL=4"
MG="SFEM_GMG=1 SFEM_GMG_GALERKIN=2 SFEM_SMOOTHER=vanka"
ONESTEP="SFEM_NL_MAX_IT=1 SFEM_VERIFY_TOL=1e9"
PROBES="SFEM_VANKA_STENCIL_SUM=1 SFEM_GMG_COARSE_MAT_SUM=1 SFEM_VANKA_WEIGHT_SUM=1"

for pk in 0 256 512; do
    echo "########## SFEM_PACK_SIZE=${pk} ##########"
    timeout --signal=TERM --kill-after=20 300 \
        srun -u --ntasks=1 --cpus-per-task=72 --kill-on-bad-exit=1 \
            --uenv="$UENV_IMAGE" --view="$UENV_VIEW" \
            bash -lc "${COMMON} ${CASE} ${MG} ${ONESTEP} ${PROBES} SFEM_PACK_SIZE=${pk} \
                      stdbuf -oL ${BIN} ${OUT}/pk${pk}" \
        > "$RAW/pk${pk}.log" 2>&1
    echo "  (exit $?)"
    grep -aE "^n_packs:|stencilsum|coarsemat|weight|vankaslot|lin_it_total" "$RAW/pk${pk}.log" | head -20
    echo
done
echo "raw logs: $RAW"
