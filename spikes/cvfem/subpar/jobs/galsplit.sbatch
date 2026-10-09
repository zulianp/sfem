#!/usr/bin/env bash
#SBATCH --account=c40
#SBATCH --job-name=cvfem-galsplit
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --time=00:25:00
#SBATCH --hint=nomultithread
#SBATCH --uenv-passthrough=ignore
#SBATCH --output=galsplit_%j.out
#SBATCH --error=galsplit_%j.err

# IS THE MULTI-PACK DEFECT IN THE GALERKIN FOLD, OR IN THE TRANSFERS / COARSE OPERATOR?
#
# Established: cd8db4f32 is the first bad commit (bisected, f276b554b..50d9c6cf1). It replaced
# a read of the FINE macro-corner slots with a read of the COARSE block's element table. The
# coarse HEX8 mesh is packed with modify_mesh=true during derefine_op -> initialize(), so its
# node ids are renumbered whenever there is more than one pack; the fine SS mesh is never packed
# (SFEM_SS_PACK_SIZE defaults to 0). Hence the new read moves with pack count and the old one
# does not. Measured serially at cavity N=8 refine 4 (512 elements):
#
#       read      pk=0     pk=256 (2 packs)   pk=512 (1 pack)
#       current   7        14                 7          gid sum 1472876 / 1482207 / 1472876
#       legacy    7         7                 7          gid sum 1472876 everywhere
#
# and the coarse matrix loses mass at 2 packs: abs 994.014 -> 990.796, while the fine stencil
# checksums and the Vanka weights are bit-identical. Neither read is right everywhere: at 2 ranks
# with packing OFF the legacy read aborts on `coarse id 17504 is outside n_coarse=405`, which is
# the heap corruption cd8db4f32 existed to fix.
#
# THE GAP THIS JOB CLOSES. cd8db4f32 changed the Galerkin ids AND the prolongation columns, both
# to the coarse block's table. If those two moved together and nothing else, the coarse matrix
# would be a PERMUTED version of the right one and the iteration count would not change. It does
# change, so a third consumer still uses the old numbering. SFEM_GMG_GALERKIN=0 removes the
# Galerkin fold entirely (rediscretisation on the coarse space):
#
#   split GONE at GALERKIN=0    -> the fault is in the Galerkin assembly path
#   split PERSISTS at GALERKIN=0 -> the fault is in the transfers or the coarse operator's own
#                                   numbering, and fixing galerkin_gid_from_spaces alone is not
#                                   enough
#
# GALERKIN=1 is carried as the middle point rather than assumed equivalent to either end.
#
# A count at the iteration cap did not converge and is a floor; newton_converged is printed for
# every arm and must be read before any count is compared.

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
OUT="${OUT:-$SCRATCH/cvfem_galsplit/${SLURM_JOB_ID:-manual}}"
RAW="$OUT/raw"
mkdir -p "$RAW"
[ -x "$BIN" ] || { echo "error: no binary at $BIN" >&2; exit 1; }

COMMON="OMP_NUM_THREADS=72 OMP_PROC_BIND=true SFEM_FGMRES=1 SFEM_ENABLE_OUTPUT=0 \
        SFEM_PARALLEL_MF_LEGACY=1 SFEM_ALLOW_MPI=1 \
        SFEM_LSOLVE_MAX_IT=4000 SFEM_LSOLVE_RTOL=1e-4 SFEM_LSOLVE_ATOL=1e-14"
CASE="SFEM_CASE=cavity SFEM_N=8 SFEM_ELEMENT_REFINE_LEVEL=4"
ONESTEP="SFEM_NL_MAX_IT=1 SFEM_VERIFY_TOL=1e9"

printf "  %-12s %-8s %-12s %s\n" "GALERKIN" "pack" "lin_it" "converged"
for gal in 0 1 2; do
    for pk in 0 256 512; do
        tag="gal${gal}_pk${pk}"
        timeout --signal=TERM --kill-after=20 300 \
            srun -u --ntasks=1 --cpus-per-task=72 --kill-on-bad-exit=1 \
                --uenv="$UENV_IMAGE" --view="$UENV_VIEW" \
                bash -lc "${COMMON} ${CASE} ${ONESTEP} SFEM_GMG=1 SFEM_SMOOTHER=vanka \
                          SFEM_GMG_GALERKIN=${gal} SFEM_PACK_SIZE=${pk} \
                          stdbuf -oL ${BIN} ${OUT}/${tag}" \
            > "$RAW/${tag}.log" 2>&1
        rc=$?
        it=$(grep -a lin_it_total "$RAW/${tag}.log" | head -1 | sed 's/.*lin_it_total: //')
        cv=$(grep -ao "newton_converged: [01]" "$RAW/${tag}.log" | head -1 | awk '{print $2}')
        printf "  %-12s %-8s %-12s %s (rc=%s)\n" "$gal" "$pk" "${it:-NONE}" "${cv:-?}" "$rc"
    done
    echo
done
echo "raw logs: $RAW"
