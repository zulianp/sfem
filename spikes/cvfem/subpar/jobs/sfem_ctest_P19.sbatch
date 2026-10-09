#!/usr/bin/env bash
#SBATCH --account=c40
#SBATCH --job-name=sfem-ctest-P19
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --time=00:29:00
#SBATCH --exclusive
#SBATCH --hint=nomultithread
#SBATCH --uenv-passthrough=ignore
#SBATCH --output=sfem_ctest_P19_%j.out
#SBATCH --error=sfem_ctest_P19_%j.err
# SFEM's own test suite on a clean tree at HEAD plus the two algebra edits of the zero-start
# pre-smoothing change (sfem_Multigrid.hpp gains set_initial_guess_zero and hints its smoothers;
# sfem_Stationary.hpp maps the hint onto use_arg_as_first_residual). Built in
# $SCRATCH/sfem-zsg-src/build-test with the sfem-omp configuration and SFEM_ENABLE_TESTING=ON.
set -u
B=$SCRATCH/sfem-zsg-src/build-test
cd "$B"
export OMP_NUM_THREADS=8 OMP_PROC_BIND=false
UENV_ARGS=$(grep '^uenv start' ~/init.sh | sed 's/^uenv start //')
echo "host $(hostname) build $B $(date)"
uenv run $UENV_ARGS -- bash -c "ctest -N | tail -1; ctest --output-on-failure --timeout 600 -j 8 2>&1 | tail -60"
echo "=== DONE $(date)"
