#!/usr/bin/env bash
#SBATCH --account=c40
#SBATCH --job-name=cvfem-slotcheck
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --time=00:25:00
#SBATCH --hint=nomultithread
#SBATCH --uenv-passthrough=ignore
#SBATCH --output=slotcheck_%j.out
#SBATCH --error=slotcheck_%j.err

# DOES MULTI-PACK RENUMBERING BREAK THE 27-SLOT LATTICE KEY?
#
# The Galerkin BSR addresses a block by (element, local node, 27-slot lattice direction), and
# pat.pos is built on the assumption that (node, direction) identifies a node PAIR uniquely.
# I verified that earlier in this session -- 5 cases, ~2M pairs, 0 mismatches -- but every one
# of those ran at the default pack size, which for those meshes was a SINGLE pack. The regime
# this job tests was never covered.
#
# Why it is the suspect now. cd8db4f32 made the coarse ids come from the coarse HEX8 block's
# element table, and that mesh IS packed (modify_mesh=true -> renumber_nodes) during
# derefine_op -> initialize(). Restriction, prolongation and the Galerkin gid all read that same
# table, so they move TOGETHER -- which rules out a mismatch between them. But a pure permutation
# of coarse ids cannot change the sum of absolute block values, and that sum FELL:
#
#     coarsemat abs   994.01366755410936  (pk=0, and pk=512 = 1 pack)
#                     990.79611102297747  (pk=256 = 2 packs)
#
# Mass is lost, so contributions are being merged or dropped rather than relabelled. A slot
# collision does exactly that: two distinct pairs landing on one block, one overwriting or
# absorbing the other.
#
# READ, committed before the run:
#
#   mismatches at pk=256 only   -> the lattice key is not a key once packing renumbers the
#                                  coarse mesh; that is the mechanism, and the fix must stop
#                                  keying coarse blocks on a lattice direction derived from
#                                  renumbered ids
#   zero mismatches everywhere  -> the key holds and the lost mass comes from elsewhere in the
#                                  fold; this suspect is dead and the next cut is galerkin_
#                                  accumulate itself
#
# SFEM_VANKA_STENCIL_SUM and SFEM_GMG_COARSE_MAT_SUM are carried alongside so the checksums sit
# next to the check that explains them.

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
OUT="${OUT:-$SCRATCH/cvfem_slotcheck/${SLURM_JOB_ID:-manual}}"
RAW="$OUT/raw"
mkdir -p "$RAW"
[ -x "$BIN" ] || { echo "error: no binary at $BIN" >&2; exit 1; }

COMMON="OMP_NUM_THREADS=72 OMP_PROC_BIND=true SFEM_FGMRES=1 SFEM_ENABLE_OUTPUT=0 \
        SFEM_PARALLEL_MF_LEGACY=1 SFEM_ALLOW_MPI=1 \
        SFEM_LSOLVE_MAX_IT=4000 SFEM_LSOLVE_RTOL=1e-4 SFEM_LSOLVE_ATOL=1e-14"
CASE="SFEM_CASE=cavity SFEM_N=8 SFEM_ELEMENT_REFINE_LEVEL=4"
MG="SFEM_GMG=1 SFEM_GMG_GALERKIN=2 SFEM_SMOOTHER=vanka"
ONESTEP="SFEM_NL_MAX_IT=1 SFEM_VERIFY_TOL=1e9"
PROBES="SFEM_VANKA_SLOT_CHECK=1 SFEM_VANKA_STENCIL_SUM=1 SFEM_GMG_COARSE_MAT_SUM=1"

for pk in 0 256 512; do
    echo "########## SFEM_PACK_SIZE=${pk} ##########"
    timeout --signal=TERM --kill-after=20 300 \
        srun -u --ntasks=1 --cpus-per-task=72 --kill-on-bad-exit=1 \
            --uenv="$UENV_IMAGE" --view="$UENV_VIEW" \
            bash -lc "${COMMON} ${CASE} ${MG} ${ONESTEP} ${PROBES} SFEM_PACK_SIZE=${pk} \
                      stdbuf -oL ${BIN} ${OUT}/pk${pk}" \
        > "$RAW/pk${pk}.log" 2>&1
    echo "  (exit $?)"
    grep -aE "^n_packs:|vankaslot|stencilsum|coarsemat|lin_it_total" "$RAW/pk${pk}.log" | head -12
    echo
done
echo "raw logs: $RAW"
