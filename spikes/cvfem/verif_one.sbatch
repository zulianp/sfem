#!/usr/bin/env bash
#SBATCH --account=c40
#SBATCH --job-name=verif1
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --time=00:30:00
#SBATCH --exclusive
#SBATCH --hint=nomultithread
#SBATCH --uenv-passthrough=ignore
#SBATCH --output=verif1_%x_%j.out
#SBATCH --error=verif1_%x_%j.err

# One verification case per job, at ~20M dofs on a single Grace socket.
#
# Three cases in one allocation does not work here. The trace is written from a static
# destructor that runs only on a normal exit, so a case killed by `timeout` writes nothing;
# and at this size a single case can spend the whole allocation before its first linear
# iteration completes. Splitting gives each case the full 30 minutes and a real chance to
# exit on its own, which is the only way its breakdown reaches disk.
#
# CASE, NXYZ and MU come from the submitting environment (sbatch --export), so one script
# serves all three. CASE_EXTRA carries per-case solver settings.
#
# The linear iteration cap is SFEM_LSOLVE_MAX_IT. SFEM_LINEAR_MAX_IT, which earlier runs in
# this work set, is read by nothing -- the cap stayed at its default of 1000 and those runs
# were uncapped. That is why a case asked for one iteration printed seven, and why several
# jobs ran to the wall and were killed before the trace could be written.

set -x
cd /capstor/scratch/cscs/zulianp/sfem/spikes/cvfem
source ./alps_env.sh
export OMP_NUM_THREADS=72 OMP_PROC_BIND=close OMP_PLACES=cores

cvfem_uenv stdbuf -oL -eL env \
  SMESH_TRACE_FILE="trace_${CASE_TAG}.csv" \
  SFEM_CASE=${CASE_NAME} \
  SFEM_NX=${CASE_NX} SFEM_NY=${CASE_NY} SFEM_NZ=${CASE_NZ} \
  SFEM_ELEMENT_REFINE_LEVEL=8 \
  SFEM_MU=${CASE_MU} \
  SFEM_GMG=1 SFEM_GMG_GALERKIN=2 SFEM_GMG_EGAL=1 \
  SFEM_FGMRES_RESTART=10 \
  SFEM_LSOLVE_MAX_IT=${CASE_LSOLVE_MAX_IT:-3} \
  ${CASE_EXTRA:-SFEM_DUMMY=0} \
  SFEM_LIN_PRINT_EVERY=1 \
  SFEM_NL_MAX_IT=1 \
  SFEM_VERIFY_TOL=1e9 \
  SFEM_ENABLE_OUTPUT=0 \
  ./build_omp/cvfem_hex8_ns_ssgmg /dev/null
