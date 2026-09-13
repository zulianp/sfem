#!/usr/bin/env bash
#SBATCH --account=c40
#SBATCH --job-name=cvfem-pump-large
#SBATCH --partition=normal
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=72
#SBATCH --time=24:00:00
#SBATCH --exclusive --hint=nomultithread --uenv-passthrough=ignore
#SBATCH --output=pumplarge_%j.out
#SBATCH --error=pumplarge_%j.err
#
# The diaphragm pump, one full cycle, at the size and Reynolds number the debug partition
# cannot reach.
#
#   N=64     274,625 nodes, 262,144 elements, 1,098,500 dof -- 7.6x the N=32 run
#   Re=500   mu=0.002, against 200 at N=32 and 50 in viz/pump_cycle
#   dt=1/128 of the period, 4x finer than the 1/32 the earlier cycles used, 128 steps
#
# Twenty-four hours because the debug partition's thirty minutes bought 16 of 32 steps at a
# seventh of this size. One Grace socket: the solver is OpenMP within a single rank, so the
# other three modules on the node would only add NUMA distance, and --exclusive keeps a
# neighbour from making a day-long run unreproducible.
#
# The finer timestep is not only for accuracy, it is what pays for the larger mesh. The
# transient term puts a0 rho/dt times the control volume on each velocity diagonal
# (cvfem_hex8_layout_common.hpp:725), so quartering dt quadruples the block that block-Jacobi
# inverts and each solve gets correspondingly better conditioned. It does nothing for the
# pressure coupling, which stays the hard part.
#
# Frames are written every step. A run that is cut short -- by the wall clock, by a node
# failure, by anything -- then leaves every step it did finish usable, which is how the
# 30-minute attempt still produced half a cycle.
set -u
D=${DRIVER:?set DRIVER to the solver binary}
OUT=${OUT:-$SCRATCH/cvfem_pumplarge/${SLURM_JOB_ID:-manual}}
mkdir -p "$OUT"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-72} OMP_PROC_BIND=true OMP_PLACES=cores
U=$(grep '^uenv start' ~/init.sh | sed 's/^uenv start //')
N=${PUMP_N:-64}
MU=${PUMP_MU:-0.002}
DT=${PUMP_DT:-0.0078125}
NS=${PUMP_NSTEPS:-128}
echo "host $(hostname)  threads $OMP_NUM_THREADS  $(date)"
echo "out  $OUT"
echo "N=$N  mu=$MU (Re=$(awk -v m=$MU 'BEGIN{printf "%g", 1/m}'))  dt=$DT  steps=$NS"

( export SFEM_CASE=pump SFEM_N=$N SFEM_MU=$MU SFEM_U=1
  export SFEM_GMG=0 SFEM_FGMRES=1 SFEM_FGMRES_RESTART=480 SFEM_PRECOND=bjacobi
  export SFEM_NL_MAX_IT=12 SFEM_LSOLVE_MAX_IT=6000
  export SFEM_DT=$DT SFEM_NSTEPS=$NS SFEM_PUMP_PERIOD=1 SFEM_BDF_ORDER=2
  export SFEM_WRITE_STEPS=1
  uenv run $U -- "$D" "$OUT" ) > "$OUT/run.log" 2>&1 &
RUNPID=$!

# The whole log is kept regardless; this is the view. Per-step quantities rather than
# end-of-run ones, so a day-long job is legible while it runs instead of only after.
( tail -F "$OUT/run.log" 2>/dev/null | stdbuf -oL \
    grep -E "^pump: t =|^pump: chamber|ndof:|newton_converged|lin_it_total|diverg|abandon" ) &
TAILPID=$!
wait $RUNPID; RC=$?
sleep 2; kill $TAILPID 2>/dev/null

echo
echo "exit=$RC"
grep -E "^pump: chamber|ndof:|^pump: swept|^pump: \|port" "$OUT/run.log" | tail -6
echo "frames: $(ls -d $OUT/step_* 2>/dev/null | wc -l) of $NS"
du -sh "$OUT"
echo "=== DONE $(date) ==="
