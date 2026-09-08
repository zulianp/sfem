#!/usr/bin/env bash
#SBATCH --account=c40
#SBATCH --job-name=poi20M
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=288
#SBATCH --time=00:30:00
#SBATCH --exclusive
#SBATCH --hint=nomultithread
#SBATCH --uenv-passthrough=ignore
#SBATCH --output=poi20M_%j.out
#SBATCH --error=poi20M_%j.err

# Poiseuille at ~20M dofs, with the linear solver's convergence history.
#
# 20M is almost exactly where the assembled-matrix path stops working. The fine BSR holds
# nodes * 27 nnz/row * 16 entries scalar values, and that count passes the signed 32-bit
# limit at
#
#     nodes < 4,971,026   ->   ndof < 19,884,104
#
# so 17.1M dofs fits (1.85e9 values, 13.8 GB) and 20.5M does not (2.21e9 values). Three arms
# separate the ceiling from the physics rather than assuming which one bites:
#
#   A  17.1M  assembled + Vanka        under the limit -- the good smoother, should work
#   B  20.5M  assembled + Vanka        over  the limit -- expected to fail, pins the ceiling
#   C  20.5M  matrix-free + bjacobi    assembles nothing -- diagnostics at the full 20M
#
# `|| true` so a failing arm does not abandon the rest. SFEM_LIN_PRINT_EVERY=1 and stdbuf
# keep every linear iteration visible while the job runs.

set -x
# Locate the spike root -- the directory holding CMakeLists.txt, drivers/ and src/.
# Resolved from this script's own path rather than a hardcoded scratch path, so the job
# does not carry one user's $SCRATCH layout and works from any checkout.
SPIKE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SPIKE_ROOT"
source ./scripts/alps_env.sh
export OMP_NUM_THREADS=288 OMP_PROC_BIND=spread OMP_PLACES=cores

run_arm() {  # run_arm <label> <m> <extra env...>
  local label=$1 m=$2; shift 2
  echo "################ ARM $label  NX=$((2*m)) NY=$m NZ=$m ################"
  cvfem_uenv stdbuf -oL -eL env \
    SFEM_CASE=poiseuille \
    SFEM_NX=$((2*m)) SFEM_NY=$m SFEM_NZ=$m SFEM_ELEMENT_REFINE_LEVEL=8 \
    SFEM_MU=0.01 SFEM_GMG=1 \
    SFEM_FGMRES_RESTART=10 \
    SFEM_LINEAR_MAX_IT=100 \
    SFEM_LIN_PRINT_EVERY=1 \
    SFEM_NL_MAX_IT=3 \
    SFEM_VERIFY_TOL=1e9 \
    SFEM_ENABLE_OUTPUT=0 \
    "$@" \
    ./build_omp/cvfem_hex8_ns_ssgmg /dev/null || true
  echo "################ END $label ################"
}

run_arm "A_17.1M_assembled_vanka" 16 SFEM_GMG_GALERKIN=2 SFEM_GMG_EGAL=1 SFEM_SMOOTHER=vanka
run_arm "B_20.5M_assembled_vanka" 17 SFEM_GMG_GALERKIN=2 SFEM_GMG_EGAL=1 SFEM_SMOOTHER=vanka
run_arm "C_20.5M_matrixfree_bjac" 17 SFEM_GMG_GALERKIN=0 SFEM_GMG_EGAL=0 SFEM_SMOOTHER=bjacobi
