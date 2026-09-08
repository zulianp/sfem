#!/usr/bin/env bash
#SBATCH --account=c40
#SBATCH --job-name=poi-sweep
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=288
#SBATCH --time=00:30:00
#SBATCH --exclusive
#SBATCH --hint=nomultithread
#SBATCH --uenv-passthrough=ignore
#SBATCH --output=poi_sweep_%j.out
#SBATCH --error=poi_sweep_%j.err

# Where does the Poiseuille case stop scaling, and what does the linear solver do there?
#
# Two attempts at 200M dofs died with std::bad_alloc a few seconds after the first residual
# evaluation, at 66 GB RSS on a node with 850 GB -- so an allocation request is wrong, not
# the machine. Switching the coarse operators from assembled to rediscretised did not change
# it, so the earlier BSR-overflow reading explains at most part of it. Rather than guess
# again, this walks the size up and reports the largest that runs, with the linear
# convergence history at each.
#
# Each size does one Newton step with a 30-iteration cap: enough to see the convergence
# slope and the per-iteration cost, short enough that four sizes fit the debug allocation.
# `|| true` so one failure does not abandon the remaining sizes.

set -x
# Locate the spike root -- the directory holding CMakeLists.txt, drivers/ and src/.
# Resolved from this script's own path rather than a hardcoded scratch path, so the job
# does not carry one user's $SCRATCH layout and works from any checkout.
SPIKE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SPIKE_ROOT"
source ./scripts/alps_env.sh
export OMP_NUM_THREADS=288 OMP_PROC_BIND=spread OMP_PLACES=cores

for cfg in "18:24M" "23:50M" "29:101M" "36:193M"; do
  m=${cfg%%:*}; label=${cfg##*:}
  echo "################ SIZE $label  (NX=$((2*m)) NY=$m NZ=$m) ################"
  cvfem_uenv stdbuf -oL -eL env \
    SFEM_CASE=poiseuille \
    SFEM_NX=$((2*m)) SFEM_NY=$m SFEM_NZ=$m SFEM_ELEMENT_REFINE_LEVEL=8 \
    SFEM_MU=0.01 \
    SFEM_GMG=1 SFEM_GMG_GALERKIN=0 SFEM_GMG_EGAL=0 \
    SFEM_SMOOTHER=bjacobi \
    SFEM_FGMRES_RESTART=10 \
    SFEM_LINEAR_MAX_IT=30 \
    SFEM_LIN_PRINT_EVERY=1 \
    SFEM_NL_MAX_IT=1 \
    SFEM_VERIFY_TOL=1e9 \
    SFEM_ENABLE_OUTPUT=0 \
    ./build_omp/cvfem_hex8_ns_ssgmg /dev/null || true
  echo "################ END $label ################"
done
