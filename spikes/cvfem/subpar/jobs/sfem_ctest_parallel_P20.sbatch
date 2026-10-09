#!/usr/bin/env bash
#SBATCH --account=c40
#SBATCH --job-name=sfem-ctest-par-P20
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=8
#SBATCH --time=00:15:00
#SBATCH --exclusive
#SBATCH --hint=nomultithread
#SBATCH --uenv=prgenv-gnu/24.11:v2
#SBATCH --view=default
#SBATCH --output=sfem_ctest_parallel_P20_%j.out
#SBATCH --error=sfem_ctest_parallel_P20_%j.err
# SFEM's MPI-parallel tests on the clean tree with the zero-start algebra edits
# ($SCRATCH/sfem-zsg-src/build-test). In P19 they never ran: ctest was wrapped in `uenv run`, and
# the srun steps it launched did not see the uenv, so every rank failed to load libmpi. Here the
# uenv is mounted for the whole allocation, so every step has it.
set -u
B=$SCRATCH/sfem-zsg-src/build-test
cd "$B"
export OMP_NUM_THREADS=2 OMP_PROC_BIND=false
echo "host $(hostname) build $B $(date)"
ctest -R Parallel -N | tail -1
ctest -R Parallel --output-on-failure --timeout 600 2>&1 | tail -60
echo "=== DONE $(date)"
