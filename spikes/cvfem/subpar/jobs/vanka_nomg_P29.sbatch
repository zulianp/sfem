#!/usr/bin/env bash
#SBATCH --account=c40
#SBATCH --job-name=cvfem-vanka-nomg-P29
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --time=00:25:00
#SBATCH --exclusive
#SBATCH --hint=nomultithread
#SBATCH --uenv=prgenv-gnu/24.11:v2
#SBATCH --view=default
#SBATCH --output=vanka_nomg_P29_%j.out
#SBATCH --error=vanka_nomg_P29_%j.err
# The stencil path without multigrid. The earlier with/without-multigrid check ran the no-multigrid
# default, which is block-Jacobi (SFEM_PRECOND), so it never reached make_diagonal_vanka at all.
# SFEM_PRECOND=vanka does, and SFEM_VANKA_CHECK=1 makes this run verify the stencils against a BSR
# built on the same state in that path too. FGMRES throughout; no multigrid needs a longer restart
# and a higher linear cap than the defaults (30 and 1000).
set -u
SPIKE=$SCRATCH/sfem/spikes/cvfem; B=$SPIKE/build_gate
export OMP_NUM_THREADS=72 OMP_PROC_BIND=true OMP_PLACES=cores
OUT=$SCRATCH/cvfem_vanka_nomg/$SLURM_JOB_ID; mkdir -p "$OUT"
NOZ="SFEM_CASE=nozzle SFEM_NOZZLE_RE=500 SFEM_GEOM=affine SFEM_NOZZLE_NCORE=2 SFEM_NOZZLE_NBORE=1 SFEM_NOZZLE_NOUTER=1"
cd "$OUT"
stdbuf -oL -eL timeout 900 env $NOZ SFEM_ELEMENT_REFINE_LEVEL=4 "SFEM_NOZZLE_NAXIAL=4 3 8 12" \
    SFEM_FGMRES=1 SFEM_FGMRES_RESTART=480 SFEM_GMG=0 SFEM_PRECOND=vanka \
    SFEM_LSOLVE_MAX_IT=3000 SFEM_NL_MAX_IT=3 SFEM_VANKA_CHECK=1 \
    "$B/cvfem_hex8_ns_ssgmg" /dev/null > out.log 2>&1
echo "exit $?"
grep -E "ndof:|preconditioner:|vanka stencil vs bsr|newton [0-9]|newton_converged|t_solve" out.log | head -14
echo "=== DONE $(date)"
