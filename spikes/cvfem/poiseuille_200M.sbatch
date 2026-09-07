#!/usr/bin/env bash
#SBATCH --account=c40
#SBATCH --job-name=poi200M
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=288
#SBATCH --time=00:30:00
#SBATCH --exclusive
#SBATCH --hint=nomultithread
#SBATCH --uenv-passthrough=ignore
#SBATCH --output=poi200M_%j.out
#SBATCH --error=poi200M_%j.err

# Poiseuille at ~200M dofs, to capture the linear solver's convergence history.
#
#   nodes = (NX*L+1)(NY*L+1)(NZ*L+1) = 601 * 289 * 289 = 50,196,121
#   ndof  = 4 * nodes                = 200,784,484
#
# Coarse operators are rediscretised (GALERKIN=0, EGAL=0), not assembled: at 200M dofs the
# BSR scalar-value count is nnz*16 = 2.17e10, which overflows the 32-bit count type, and so
# does the first coarse level at 2.7e9. The assembled path tops out near 160M dofs. The job
# died with std::bad_alloc at only 66 GB RSS on a 850 GB node, which is the signature.
#
# Smoother is block-Jacobi, not Vanka, and that is a memory decision rather than a
# preference: the Vanka patch operator is read from the assembled fine BSR, which at this
# size is about 162 GB (50.2M nodes * 27 nnz/row * 16 * 8 B) against 128 GB on a Grace
# module. The block diagonal is 6 GB. Vanka here would need a matrix-free patch assembly
# or a distributed run.
#
# FGMRES restart is 10 rather than the default 30: the Krylov basis plus the flexible Z
# vectors is 2*restart+1 vectors at 1.5 GB each, so 30 would be 91 GB before anything else
# is counted.
#
# SFEM_LIN_PRINT_EVERY=1 prints every linear iteration. At this size one iteration is
# seconds of work, so the default stride of 50 would emit almost nothing inside the
# allocation. stdbuf keeps output flowing to the .out file instead of sitting in a 4 KB
# block buffer -- a job that prints only at the end is indistinguishable from a hung one.

set -x
cd /capstor/scratch/cscs/zulianp/sfem/spikes/cvfem
source ./alps_env.sh

export OMP_NUM_THREADS=288
export OMP_PROC_BIND=spread
export OMP_PLACES=cores

cvfem_uenv stdbuf -oL -eL env \
  SFEM_CASE=poiseuille \
  SFEM_NX=75 SFEM_NY=36 SFEM_NZ=36 SFEM_ELEMENT_REFINE_LEVEL=8 \
  SFEM_MU=0.01 \
  SFEM_GMG=1 SFEM_GMG_GALERKIN=0 SFEM_GMG_EGAL=0 \
  SFEM_SMOOTHER=bjacobi \
  SFEM_FGMRES_RESTART=10 \
  SFEM_LINEAR_MAX_IT=200 \
  SFEM_LIN_PRINT_EVERY=1 \
  SFEM_NL_MAX_IT=2 \
  SFEM_VERIFY_TOL=1e9 \
  SFEM_ENABLE_OUTPUT=0 \
  ./build_omp/cvfem_hex8_ns_ssgmg /dev/null
