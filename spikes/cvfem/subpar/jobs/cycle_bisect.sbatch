#!/usr/bin/env bash
#SBATCH --account=c40
#SBATCH --job-name=cvfem-cycle-bisect
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=18
#SBATCH --time=00:20:00
#SBATCH --exclusive
#SBATCH --hint=nomultithread
#SBATCH --uenv-passthrough=ignore
#SBATCH --output=cycle_bisect_%j.out
#SBATCH --error=cycle_bisect_%j.err

# WHICH HALF of the V-cycle carries the decomposition dependence?
#
# Measured on the cavity (L2, two levels, every arm converged, single-threaded):
#
#     preconditioner        1 rank   2 ranks   4 ranks     degradation
#     no multigrid, bjac       832      1776      2718      2.1x / 3.3x
#     multigrid + bjacobi      162      6104     10067       38x /  62x
#     multigrid + vanka         61      3394      7255       56x / 119x
#
# So the V-cycle is the decomposition-dependent component, and multi-rank it is worse than
# useless: at two ranks NO multigrid (1776) beats multigrid with either smoother.
#
# It is NOT the smoother. Block-Jacobi reads only its own row -- structurally immune to the
# halo and patch-matrix concerns that were chased earlier -- and still degrades 38x, on the
# same order as Vanka's 56x. Whatever is wrong is shared by both, which leaves the transfers,
# the coarse correction, or the cycle's own bookkeeping.
#
# Two knobs split the cycle in half without touching any code:
#
#   SFEM_GMG_CGC=0     scales the prolonged coarse-grid correction to zero
#                      -> SMOOTHER ONLY. The transfers still run, but nothing they produce
#                         reaches the solution, so any surviving degradation is the smoother's
#                         or the cycle's bookkeeping.
#   SFEM_GMG_SMOOTH=0  gives StationaryIteration max_it = 0
#                      -> COARSE CORRECTION ONLY (plus whatever the zero-guess path does).
#
# Read: whichever half reproduces the 38x is the half that carries it. If neither does, the
# dependence is in the interaction and the next step is the single-V-cycle probe
# (SFEM_GMG_CHECK=2, gated on SFEM_GMG_CHECK_AT) comparing one cycle's output at 1 vs 2 ranks.
#
# Block-Jacobi is the informative smoother here, precisely because it cannot be blamed on
# patch gathers; Vanka is carried along as the second witness.
#
# SINGLE-THREADED throughout: Vanka's patch corrections combine with atomics, and thread
# noise already faked a 38% effect once in this campaign (nozzle 4068/2521/4332 at 72 threads,
# all exactly 3202 at one thread). Time limit is 20 minutes, under the 28 that truncated the
# previous job's last arm.

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
OUT="${OUT:-$SCRATCH/cvfem_bisect/${SLURM_JOB_ID:-manual}}"
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
      SFEM_GMG=1 SFEM_GMG_GALERKIN=2"

for sm in bjacobi vanka; do
    for n in 1 2; do
        # Intact cycle: reproduces the numbers above (bjacobi 162/6104, vanka 61/3394).
        run_arm "${sm}_full"      "$n" "${BASE} SFEM_SMOOTHER=${sm}"

        # Smoother only: transfers run, coarse correction scaled to zero.
        run_arm "${sm}_cgc0"      "$n" "${BASE} SFEM_SMOOTHER=${sm} SFEM_GMG_CGC=0"

        # Coarse correction only: no smoothing sweeps.
        run_arm "${sm}_smooth0"   "$n" "${BASE} SFEM_SMOOTHER=${sm} SFEM_GMG_SMOOTH=0"
    done
done

echo "raw logs: $RAW"
