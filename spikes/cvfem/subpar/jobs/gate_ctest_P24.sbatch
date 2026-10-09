#!/usr/bin/env bash
#SBATCH --account=c40
#SBATCH --job-name=cvfem-gate-ctest-P24
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --time=00:30:00
#SBATCH --exclusive
#SBATCH --hint=nomultithread
#SBATCH --uenv=prgenv-gnu/24.11:v2
#SBATCH --view=default
#SBATCH --output=gate_ctest_P24_%j.out
#SBATCH --error=gate_ctest_P24_%j.err
# The spike's unit tests on the clean HEAD 935297062 build (gate_build_P23). The uenv is mounted
# for the whole allocation, so any srun a test launches sees it (the P19 failure mode).
set -u
B=$SCRATCH/cvfem_gate_head/build_gate
cd "$B"
export OMP_NUM_THREADS=8 OMP_PROC_BIND=true OMP_PLACES=cores
echo "host $(hostname) build $B $(date)"
ctest -N | tail -1
ctest -j 8 --output-on-failure --timeout 900 > ctest_full.log 2>&1
echo "ctest rc $?"
grep -E "tests passed|Failed|\*\*\*|Not Run|Timeout" ctest_full.log | head -40
echo "=== DONE $(date)"
