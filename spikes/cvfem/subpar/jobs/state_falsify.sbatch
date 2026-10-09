#!/usr/bin/env bash
#SBATCH --account=c40
#SBATCH --job-name=cvfem-state-falsify
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=18
#SBATCH --time=00:20:00
#SBATCH --exclusive
#SBATCH --hint=nomultithread
#SBATCH --uenv-passthrough=ignore
#SBATCH --output=state_falsify_%j.out
#SBATCH --error=state_falsify_%j.err

# Is the Vanka-only decomposition dependence caused by an UNGATHERED LINEARISATION STATE?
#
# The cycle bisection (job 4714761, cavity L2, single-threaded) split the gap in two:
#
#     arm                          1 rank   2 ranks   ratio
#     bjacobi full                    162      6104     38x
#     bjacobi cgc0 (smoother only)    328      1307    4.0x     <- coarse path carries it
#     vanka   full                     61      3394     56x
#     vanka   cgc0 (smoother only)     115      3100     27x     <- Vanka carries its OWN
#
# Block-Jacobi inverts a 4x4 block per node, so its owned output depends only on owned input
# and it cannot be decomposition-dependent; zeroing the coarse correction duly collapses its
# degradation to 4x. Vanka keeps 27x with no coarse correction reaching the solution at all,
# so it has a second, independent dependence inside the smoother.
#
# The candidate: make_diagonal_vanka linearises every patch about g.states[0], and
# assemble_fine_stencil builds those patches over ALL local macro-elements INCLUDING AURA.
# The Newton loop writes the state over the owned range. If its ghost and aura entries are
# stale, every aura-anchored patch factorisation is built about the wrong state -- per-rank by
# construction, and independent of the coarse path, which is exactly the cgc0 signature.
#
# SFEM_GMG_CONST_STATE=1 falsifies it with no code change: it overwrites EVERY level's state
# with a uniform constant field, identical on every rank including ghosts and aura, so the
# state cannot differ between decompositions.
#
#   if vanka cgc0 degradation collapses under CONST_STATE  -> the state transfer is the cause
#   if 27x survives                                        -> it is not; look elsewhere in the
#                                                             smoother
#
# The const-state arms do NOT solve the same problem, so their absolute iteration counts are
# not comparable to the normal arms. Only the 1-rank-to-2-rank RATIO within an arm is.
#
# 4 ranks included to see whether the Vanka term scales with cut count the way the coarse term
# does (bjacobi full went 38x at two ranks and 62x at four).
#
# SINGLE-THREADED: Vanka's patch corrections combine with atomics, and thread noise already
# faked a 38% effect once in this campaign.

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
OUT="${OUT:-$SCRATCH/cvfem_state_falsify/${SLURM_JOB_ID:-manual}}"
RAW="$OUT/raw"
mkdir -p "$RAW"
[ -x "$BIN" ] || { echo "error: no binary at $BIN" >&2; exit 1; }

run_arm() {
    local tag="$1" nranks="$2"; shift 2
    echo "### ${tag} ranks=${nranks} ###"
    srun -u --ntasks="${nranks}" --cpus-per-task=1 --uenv="$UENV_IMAGE" --view="$UENV_VIEW" \
        bash -lc "$* stdbuf -oL ${BIN} ${OUT}/${tag}_n${nranks}" 2>&1 \
        | tee "$RAW/${tag}_n${nranks}.log" \
        | stdbuf -oL grep -aE "gmg: |newton_converged|lin_it_total"
    echo
}

BASE="OMP_NUM_THREADS=1 SFEM_FGMRES=1 SFEM_ENABLE_OUTPUT=0 \
      SFEM_PARALLEL_MF_LEGACY=1 SFEM_ALLOW_MPI=1 \
      SFEM_CASE=cavity SFEM_N=4 SFEM_ELEMENT_REFINE_LEVEL=2 \
      SFEM_GMG=1 SFEM_GMG_GALERKIN=2 SFEM_SMOOTHER=vanka"

for n in 1 2 4; do
    # Reference: reproduces 61 / 3394 / 7255 and 115 / 3100 / ? from the bisection.
    run_arm "vanka_cgc0"             "$n" "${BASE} SFEM_GMG_CGC=0"
    run_arm "vanka_full"             "$n" "${BASE}"

    # Same two arms with a state that cannot differ between decompositions.
    run_arm "vanka_cgc0_conststate"  "$n" "${BASE} SFEM_GMG_CGC=0 SFEM_GMG_CONST_STATE=1"
    run_arm "vanka_full_conststate"  "$n" "${BASE} SFEM_GMG_CONST_STATE=1"
done

echo "raw logs: $RAW"
