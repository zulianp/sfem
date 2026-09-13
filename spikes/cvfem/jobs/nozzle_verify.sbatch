#!/usr/bin/env bash
#SBATCH --account=c40
#SBATCH --job-name=cvfem-nozzle
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --time=00:28:00
#SBATCH --exclusive
#SBATCH --hint=nomultithread
#SBATCH --uenv-passthrough=ignore
#SBATCH --output=nozzle_%j.out
#SBATCH --error=nozzle_%j.err

# The FDA nozzle: is the operator right on a curved mesh, and what does the verification
# case cost?
#
#   sbatch jobs/nozzle_verify.sbatch            (from spikes/cvfem)
#
# 1. A refinement ladder at throat Re = 1, over the cross-section at fixed axial resolution.
#    At Re = 1 the flow is developed within a diameter, so the answer is known where it
#    matters: 2 u_in on the axis of the inlet pipe (x = -0.088) and 2 u_t = 18 u_in on the
#    axis of the throat (x = -0.020). A two-ring mesh measured 3.96 and 5.17 there -- both
#    wrong, and in opposite directions -- while conserving mass exactly. Under-resolution and
#    an operator defect on curved elements both predict that; only the trend under
#    refinement tells them apart.
#
# 2. The same smallest mesh with SFEM_RC_SCALE=0. If the pressure correction is carrying the
#    flux the nodal velocities do not, removing it moves the nodal profile toward the one the
#    flux implies.
#
# 3. The mesh scripts/verify_report.sh runs the nozzle group on, at Re = 500: conservation,
#    the inflow error against the report's 5% tolerance, and the cost.
#
# Flat and isoparametric -- the semi-structured operator refuses curved macro elements, so
# there is no multigrid arm. FGMRES explicitly, preconditioned by the dense LU.
set -u

SPIKE_ROOT=${SPIKE_ROOT:-$SLURM_SUBMIT_DIR}
if [ ! -f "$SPIKE_ROOT/CMakeLists.txt" ]; then
    echo "nozzle_verify.sbatch: \$SPIKE_ROOT=$SPIKE_ROOT is not the spike root" >&2
    exit 1
fi
cd "$SPIKE_ROOT"

export OMP_NUM_THREADS=72 OMP_PROC_BIND=true OMP_PLACES=cores
UENV_ARGS=$(grep '^uenv start' ~/init.sh | sed 's/^uenv start //')
BUILD=${BUILD:-$SPIKE_ROOT/build_solver}
DRIVER=$BUILD/cvfem_hex8_ns_ssgmg
OUT=${OUT:-$SCRATCH/cvfem_nozzle/$SLURM_JOB_ID}
mkdir -p "$OUT"
echo "host $(hostname)  threads $OMP_NUM_THREADS  driver $DRIVER  out $OUT  $(date)"

COMMON="SFEM_CASE=nozzle SFEM_FGMRES=1 SFEM_GMG=0 SFEM_PRECOND=direct SFEM_DIRECT_MAX_DOF=40000 SFEM_ELEMENT_REFINE_LEVEL=1"

# run <tag> <env...>: complete log in $OUT/<tag>.log, profiles in $OUT/<tag>/, filtered view here.
run() {
    local tag=$1; shift
    uenv run $UENV_ARGS -- env CVRUN_DIR="$OUT" CVRUN_OUT="$OUT/$tag" \
        CVRUN_FILTER='nnodes:|ndof:|^nozzle: |newton_converged|highest Re SOLVED|sum of continuity|t_solve|lin_it_total|verification' \
        timeout 900 bash scripts/cvrun.sh "$tag" "$DRIVER" $COMMON "$@" || true
}

# Job 4656808 ran the ladder at Re 1 and settled it: core 2 and core 4 converge at second
# order to the known centreline values (core 6, 33,820 dof, did not finish -- the probed LU
# cost 782 s already at 15,740). What is left:
#
#   re10_core2    the smoke run's Re 10 on the ladder's axial resolution. The smoke run
#                 (axial 2 2 4 6) was wrong; the ladder (axial 8 6 16 24) was right at Re 1.
#                 This separates the Reynolds number from the axial resolution.
#   re500_verify  exactly what scripts/verify_report.sh runs: the gates and the cost.
NOZZLE_CORE2="SFEM_NOZZLE_NCORE=2 SFEM_NOZZLE_NBORE=1 SFEM_NOZZLE_NOUTER=1"
run re10_core2 SFEM_NOZZLE_RE=10 $NOZZLE_CORE2 "SFEM_NOZZLE_NAXIAL=8 6 16 24"
run re500_verify SFEM_NOZZLE_RE=500 $NOZZLE_CORE2 "SFEM_NOZZLE_NAXIAL=8 6 16 24"

echo "=== DONE $(date) ==="
