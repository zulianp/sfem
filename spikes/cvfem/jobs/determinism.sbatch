#!/usr/bin/env bash
#SBATCH --account=c40
#SBATCH --job-name=cvfem-determinism
#SBATCH --partition=debug
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=72
#SBATCH --time=00:20:00
#SBATCH --exclusive --hint=nomultithread --uenv-passthrough=ignore
#SBATCH --output=determinism_%j.out
#SBATCH --error=determinism_%j.err
# Is the solver bit-reproducible with itself? Run one case five times at one thread and
# five times at 72, and compare the linear iteration count and the residual at a fixed
# iteration. At one thread every reduction has a single fixed order and the repeats agree;
# at 72 they do not, because the boundary closure scatters into shared nodes with
# atomic_add (src/hex8/cvfem_hex8_layout_common.hpp) and the Vanka smoother has atomics of
# its own -- and floating-point addition is not associative.
#
# This exists because "the same binary gave a different answer" is otherwise indistinguishable
# from a regression, and that question came up the first time an A/B showed a convergence
# difference on one case. The port at p_bar = 3.0 is used because it is the most sensitive
# case in the verification matrix, not because it is representative.
set -u
export OMP_PROC_BIND=true OMP_PLACES=cores
U=$(grep '^uenv start' ~/init.sh | sed 's/^uenv start //')
D=${DRIVER:?set DRIVER to the solver binary}
OUT=${OUT:-$SCRATCH/cvfem_determinism/$SLURM_JOB_ID}; mkdir -p $OUT
echo "host $(hostname)  $(date)"
echo "=== same binary, same case, repeated: port p_bar=3.0, 33124 dof ==="
printf "%-4s %-4s %-6s %-8s %s\n" thr rep conv lin_it t_solve
for t in 1 72; do
for i in 1 2 3 4 5; do
  r=$OUT/t${t}_r$i; mkdir -p $r
  ( cd $r && uenv run $U -- env OMP_NUM_THREADS=$t OMP_PROC_BIND=true OMP_PLACES=cores \
      SFEM_LX=4 SFEM_LY=1 SFEM_MU=0.01 SFEM_U=1 SFEM_CASE=poiseuille SFEM_N=12 \
      SFEM_ELEMENT_REFINE_LEVEL=1 SFEM_BOUNDARY_MASK=1 SFEM_OUTLET=dirichlet \
      SFEM_PRESSURE_SIDESET=outlet SFEM_PRESSURE=3.0 SFEM_NL_MAX_IT=20 \
      $D $r/out > $r/log 2>&1 )
  c=$(grep -o 'newton_converged: [01]' $r/log | head -1 | awk '{print $2}')
  li=$(grep -o 'lin_it_total: [0-9]*' $r/log | head -1 | awk '{print $2}')
  ts=$(grep -o 't_solve: [0-9.]*' $r/log | head -1 | awk '{print $2}')
  printf "%-4s %-4s %-6s %-8s %s\n" $t $i "${c:-?}" "${li:-?}" "${ts:-?}"
done
done
echo
echo "=== the first Newton residual, printed to full precision by the run ==="
for t in 1 72; do for i in 1 2 3 4 5; do
  printf "thr=%-3s rep=%-2s  %s\n" $t $i "$(grep -m2 'residual abs' $OUT/t${t}_r$i/log | tail -1)"
done; done
echo "=== DONE $(date) ==="
