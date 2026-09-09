#!/usr/bin/env bash
#SBATCH --account=c40
#SBATCH --job-name=cvfem-pumpviz
#SBATCH --partition=debug
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=72
#SBATCH --time=00:28:00
#SBATCH --exclusive --hint=nomultithread --uenv-passthrough=ignore
#SBATCH --output=pumpviz_%j.out
#SBATCH --error=pumpviz_%j.err
# One full diaphragm cycle at a resolution worth looking at, written frame by frame for
# ParaView. FGMRES rather than the dense LU: at this size the LU is not affordable, and the
# identity reproduces under both.
set -u
BLK=$SCRATCH/cvfem_blk; B=$BLK/after/build_blk
OUT=${OUT:-$SCRATCH/cvfem_pumpviz/$SLURM_JOB_ID}
mkdir -p "$OUT"
export OMP_NUM_THREADS=72 OMP_PROC_BIND=close OMP_PLACES=cores
U=$(grep '^uenv start' ~/init.sh | sed 's/^uenv start //')
echo "host $(hostname) threads $OMP_NUM_THREADS $(date)"
echo "out $OUT"
# N=16 and BiCGStab, both chosen from measurement rather than taste: two time steps cost
# 5 s here, against 38 s for the dense LU at a quarter the size, and N=24 with block-Jacobi
# did not finish a single step in twenty minutes.
( export SFEM_CASE=pump SFEM_N=${SFEM_N:-16} SFEM_MU=0.02 SFEM_U=1
  export SFEM_GMG=0 SFEM_PRECOND=bjacobi
  export SFEM_NL_MAX_IT=12 SFEM_LSOLVE_MAX_IT=3000
  export SFEM_DT=0.03125 SFEM_NSTEPS=32 SFEM_PUMP_PERIOD=1 SFEM_BDF_ORDER=2
  export SFEM_WRITE_STEPS=1
  uenv run $U -- $B/cvfem_hex8_ns_ssgmg "$OUT" ) > $OUT/run.log 2>&1
echo "exit=$?"
grep -E "^pump: chamber|ndof:|newton_converged|^pump: swept|^pump: \|port" $OUT/run.log
echo "frames: $(ls -d $OUT/step_* 2>/dev/null | wc -l)"
du -sh $OUT
echo "=== DONE $(date) ==="
