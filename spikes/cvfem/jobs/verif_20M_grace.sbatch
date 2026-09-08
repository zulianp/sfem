#!/usr/bin/env bash
#SBATCH --account=c40
#SBATCH --job-name=verif20M
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --time=00:30:00
#SBATCH --exclusive
#SBATCH --hint=nomultithread
#SBATCH --uenv-passthrough=ignore
#SBATCH --output=verif20M_%j.out
#SBATCH --error=verif20M_%j.err

# The three Farrell/Mitchell/Wechsung verification cases at ~20M dofs, one Grace socket.
#
# Sizes are set just under the point where the assembled-matrix path stops working: the fine
# BSR holds nodes*27*16 scalar values, which passes the signed 32-bit limit at
# nodes = 4,971,026 (ndof 19,884,104). All three sit below that deliberately, so this
# measures the solver rather than the ceiling.
#
#   MMS     NX=NY=NZ=21  L=8   4,826,809 nodes   19,307,236 dof
#   cavity  NX=NY=NZ=21  L=8   4,826,809 nodes   19,307,236 dof
#   step    NX=70 NY=16 NZ=8   L=8  ~4,483,898 nodes  ~17,935,592 dof
#
# NX must be a multiple of 10: the step sits at x=1 of Lx=10, and create_hex8_lshape
# requires step_x/xmax*nx to be an integer. NX=72 gives 7.2 and aborts before any solve.
#
# One socket is 72 cores. A linear iteration cost about 10 s at 17.1M dofs on 288 threads, so
# expect roughly 40 s here -- which is why each case is capped at one Newton step and 12
# linear iterations, and wrapped in `timeout` so that a slow case cannot eat the allocation
# and leave the others unrun. That mistake cost the previous job its last two arms.
#
# One linear iteration per case, not twelve. The trace is dumped from Tracer's destructor,
# which runs only on a normal exit -- `timeout` sends SIGTERM and the process dies without
# writing anything. A case must therefore finish by itself for its breakdown to exist, and
# at this size the previous run showed setup, not iteration, is where the time goes: both
# MMS and the cavity reached newton 0 and produced no linear iteration in eight minutes.
# One iteration is enough to time the operator and the preconditioner apply; the setup
# phases (vanka_setup, galerkin_assemble, assemble_hierarchy, hessian_bsr) are what this
# run is really measuring.

set -x
# Locate the spike root -- the directory holding CMakeLists.txt, drivers/ and src/.
# Resolved from this script's own path rather than a hardcoded scratch path, so the job
# does not carry one user's $SCRATCH layout and works from any checkout.
SPIKE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SPIKE_ROOT"
source ./scripts/alps_env.sh
export OMP_NUM_THREADS=72 OMP_PROC_BIND=close OMP_PLACES=cores

run_case() {  # run_case <label> <env...>
  local label=$1; shift
  echo "################ CASE $label ################"
  # timeout goes INSIDE the uenv: cvfem_uenv is a shell function, and timeout can only
  # exec a real binary -- outside, it fails instantly with "No such file or directory".
  cvfem_uenv timeout 540 stdbuf -oL -eL env \
    SMESH_TRACE_FILE="trace_${label}.csv" \
    SFEM_ELEMENT_REFINE_LEVEL=8 \
    SFEM_GMG=1 SFEM_GMG_GALERKIN=2 SFEM_GMG_EGAL=1 \
    SFEM_FGMRES_RESTART=10 \
    SFEM_LINEAR_MAX_IT=1 \
    SFEM_LIN_PRINT_EVERY=1 \
    SFEM_NL_MAX_IT=1 \
    SFEM_VERIFY_TOL=1e9 \
    SFEM_ENABLE_OUTPUT=0 \
    "$@" \
    ./build_omp/cvfem_hex8_ns_ssgmg /dev/null || true
  echo "################ END $label ################"
}

run_case "MMS_19.3M"    SFEM_CASE=mms         SFEM_NX=21 SFEM_NY=21 SFEM_NZ=21 SFEM_MU=1.0
run_case "CAVITY_19.3M" SFEM_CASE=cavity_reg  SFEM_NX=21 SFEM_NY=21 SFEM_NZ=21 SFEM_MU=0.02
run_case "STEP_17.9M"   SFEM_CASE=step        SFEM_NX=70 SFEM_NY=16 SFEM_NZ=8  SFEM_MU=0.1
