#!/usr/bin/env bash
#SBATCH --account=c40
#SBATCH --job-name=cvfem-getrf-P7
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --time=00:10:00
#SBATCH --exclusive
#SBATCH --hint=nomultithread
#SBATCH --uenv-passthrough=ignore
#SBATCH --output=getrf_bench_P7_%j.out
#SBATCH --error=getrf_bench_P7_%j.err
# dgetrf alone on a 2,320 x 2,320 matrix -- the nozzle's coarsest level -- with the uenv's
# OpenBLAS, to separate the factorisation from the rest of the driver's coarse_factor phase,
# which measured ~600 ms per factorisation whatever OPENBLAS_NUM_THREADS was set to.
set -u
SPIKE_ROOT=${SPIKE_ROOT:-$SLURM_SUBMIT_DIR}
UENV_ARGS=$(grep '^uenv start' ~/init.sh | sed 's/^uenv start //')
B=$SCRATCH/cvfem_getrf_bench; mkdir -p "$B"
uenv run $UENV_ARGS -- bash -c "mpicxx -O3 -fopenmp $SPIKE_ROOT/jobs/getrf_bench.cpp -o $B/getrf_bench -lopenblas" || exit 1
for cfg in "OMP_NUM_THREADS=72 OMP_PROC_BIND=true OMP_PLACES=cores" \
           "OMP_NUM_THREADS=72 OMP_PROC_BIND=false" \
           "OMP_NUM_THREADS=72 OMP_PROC_BIND=true OMP_PLACES=cores OPENBLAS_NUM_THREADS=1" \
           "OMP_NUM_THREADS=1" \
           "OMP_NUM_THREADS=18 OMP_PROC_BIND=true OMP_PLACES=cores"; do
    echo "=== $cfg"
    uenv run $UENV_ARGS -- env $cfg "$B/getrf_bench" 2320 4
done
echo "=== DONE $(date)"
