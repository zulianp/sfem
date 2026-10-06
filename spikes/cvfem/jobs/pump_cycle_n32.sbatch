#!/usr/bin/env bash
#SBATCH --account=c40
#SBATCH --job-name=cvfem-pumpcycle
#SBATCH --partition=debug
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=72
#SBATCH --time=00:28:00
#SBATCH --exclusive --hint=nomultithread --uenv-passthrough=ignore
#SBATCH --output=pumpcycle_%j.out
#SBATCH --error=pumpcycle_%j.err
#
# One full diaphragm cycle at N=32 and Re=200, written frame by frame for ParaView.
#
# The existing viz/pump_cycle is N=16 at Re=50 (mu=0.02). This is 7.3x the unknowns and 4x
# the Reynolds number: 35,937 nodes, 32,768 elements, 143,748 dof, mu=0.005.
#
# Transient rather than steady, and that is what makes the Reynolds number affordable rather
# than merely desirable. Each step is a Newton solve started from the previous step's state
# with the BDF2 mass term on the diagonal, so there is no Reynolds continuation to walk and
# the linear system is better conditioned than the steady one at the same mu -- the steady
# solve at N=32 and Re=20 alone took 84 s.
#
# 32 steps of dt=0.03125 with SFEM_PUMP_PERIOD=1 is exactly one period, so the diaphragm
# returns to where it started. The pump has no valves, so over a full cycle it moves fluid
# back and forth and nets nothing; what there is to watch is the jet through the port
# reversing, which is the part Re changes.
#
# Frames are written as they are produced, so a run that runs out of wall clock leaves
# everything up to that point usable rather than nothing.
set -u
D=${DRIVER:?set DRIVER to the solver binary}
OUT=${OUT:-$SCRATCH/cvfem_pumpcycle/${SLURM_JOB_ID:-manual}}
mkdir -p "$OUT"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-72} OMP_PROC_BIND=true OMP_PLACES=cores
U=$(grep '^uenv start' ~/init.sh | sed 's/^uenv start //')
echo "host $(hostname)  threads $OMP_NUM_THREADS  $(date)"
echo "out  $OUT"

( export SFEM_CASE=pump SFEM_N=${SFEM_N:-32} SFEM_MU=${SFEM_MU:-0.005} SFEM_U=1
  export SFEM_GMG=0 SFEM_FGMRES=1 SFEM_FGMRES_RESTART=480 SFEM_PRECOND=bjacobi
  export SFEM_NL_MAX_IT=12 SFEM_LSOLVE_MAX_IT=4000
  export SFEM_DT=${SFEM_DT:-0.03125} SFEM_NSTEPS=${SFEM_NSTEPS:-32}
  export SFEM_PUMP_PERIOD=1 SFEM_BDF_ORDER=2 SFEM_WRITE_STEPS=1
  uenv run $U -- "$D" "$OUT" ) > "$OUT/run.log" 2>&1 &
RUNPID=$!

# Progress the reader can time themselves against: the driver names each step's time and
# diaphragm velocity, so a slow run is distinguishable from a stuck one without attaching
# anything. The raw log is kept whole regardless of what this prints.
( tail -F "$OUT/run.log" 2>/dev/null | stdbuf -oL grep -E "^pump: t =|newton_converged|ndof:|^pump: chamber" ) &
TAILPID=$!
wait $RUNPID; RC=$?
sleep 2; kill $TAILPID 2>/dev/null

echo
echo "exit=$RC"
grep -E "^pump: chamber|ndof:|^pump: swept|^pump: \|port" "$OUT/run.log" | tail -6
echo "frames: $(ls -d $OUT/step_* 2>/dev/null | wc -l) of ${SFEM_NSTEPS:-32}"
du -sh "$OUT"
echo "=== DONE $(date) ==="
