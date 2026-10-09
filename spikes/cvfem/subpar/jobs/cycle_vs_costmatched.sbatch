#!/usr/bin/env bash
#SBATCH --account=c40
#SBATCH --job-name=cvfem-cycle-vs-cm
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=18
#SBATCH --time=00:20:00
#SBATCH --exclusive
#SBATCH --hint=nomultithread
#SBATCH --uenv-passthrough=ignore
#SBATCH --output=cycle_vs_cm_%j.out
#SBATCH --error=cycle_vs_cm_%j.err

# Is the hierarchy worth anything multi-rank, or is it only an expensive smoother?
#
# The cycle bisection (job 4714761, cavity L2, single-threaded) showed the degradation splits
# in two, and that the coarse correction carries the shared term:
#
#     arm                          1 rank   2 ranks   ratio
#     bjacobi full                    162      6104     38x
#     bjacobi cgc0 (smoother only)    328      1307    4.0x
#     vanka   full                     61      3394     56x
#     vanka   cgc0 (smoother only)     115      3100     27x
#
# SFEM_GMG=2 is the control this driver already provides for exactly that question: the same
# damped block-Jacobi, run as a stationary iteration on the fine level for 2 * SFEM_GMG_SMOOTH
# sweeps -- the same number a V-cycle spends smoothing -- with NO hierarchy under it. Its own
# comment states the reading: if the V-cycle cannot beat it, the hierarchy is only an expensive
# smoother and the fault is in the transfers or the coarse operator, not the smoother.
#
# Serially the cycle wins hugely (832 with no hierarchy, 162 with it, 61 with Vanka). If at two
# and four ranks the cost-matched control matches or beats the V-cycle, the coarse-grid
# correction is not merely degraded but is contributing nothing usable once the domain is cut.
#
# SFEM_GMG_CHECK=2 is the second instrument: it runs the V-cycle STANDALONE as a solver for
# SFEM_GMG_CHECK_IT cycles on one Newton step's right-hand side and prints the per-component
# residual reduction. That isolates the cycle from the outer Krylov entirely, so a rate that
# degrades with rank count implicates the cycle itself rather than FGMRES around it. It is
# gated at Newton step SFEM_GMG_CHECK_AT (default 0), and it replaces the solve, so those arms
# report a rate rather than an iteration count.
#
# SINGLE-THREADED: Vanka's patch corrections combine with atomics, and thread noise already
# faked a 38% effect once in this campaign (nozzle 4068/2521/4332 at 72 threads, all exactly
# 3202 at one thread).

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
OUT="${OUT:-$SCRATCH/cvfem_cycle_vs_cm/${SLURM_JOB_ID:-manual}}"
RAW="$OUT/raw"
mkdir -p "$RAW"
[ -x "$BIN" ] || { echo "error: no binary at $BIN" >&2; exit 1; }

run_arm() {
    local tag="$1" nranks="$2"; shift 2
    echo "### ${tag} ranks=${nranks} ###"
    srun -u --ntasks="${nranks}" --cpus-per-task=1 --uenv="$UENV_IMAGE" --view="$UENV_VIEW" \
        bash -lc "$* stdbuf -oL ${BIN} ${OUT}/${tag}_n${nranks}" 2>&1 \
        | tee "$RAW/${tag}_n${nranks}.log" \
        | stdbuf -oL grep -aE "gmg: |newton_converged|lin_it_total|preconditioner|residual by component|^  (ux|uy|uz|p)  "
    echo
}

BASE="OMP_NUM_THREADS=1 SFEM_FGMRES=1 SFEM_ENABLE_OUTPUT=0 \
      SFEM_PARALLEL_MF_LEGACY=1 SFEM_ALLOW_MPI=1 \
      SFEM_CASE=cavity SFEM_N=4 SFEM_ELEMENT_REFINE_LEVEL=2"

for n in 1 2 4; do
    # The cost-matched control: same sweeps, no hierarchy.
    run_arm "costmatched"   "$n" "${BASE} SFEM_GMG=2"

    # The V-cycle it has to beat.
    run_arm "cycle_bjacobi" "$n" "${BASE} SFEM_GMG=1 SFEM_GMG_GALERKIN=2 SFEM_SMOOTHER=bjacobi"
    run_arm "cycle_vanka"   "$n" "${BASE} SFEM_GMG=1 SFEM_GMG_GALERKIN=2 SFEM_SMOOTHER=vanka"

    # The cycle measured on its own terms: 20 standalone V-cycles, per-component reduction.
    # Reports a rate, not an iteration count -- this arm replaces the solve.
    run_arm "cyclerate_vanka" "$n" \
        "${BASE} SFEM_GMG=1 SFEM_GMG_GALERKIN=2 SFEM_SMOOTHER=vanka \
         SFEM_GMG_CHECK=2 SFEM_GMG_CHECK_IT=20 SFEM_NL_MAX_IT=1 SFEM_VERIFY_TOL=1e9"
done

echo "raw logs: $RAW"
