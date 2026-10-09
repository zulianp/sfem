#!/usr/bin/env bash
#SBATCH --account=c40
#SBATCH --job-name=cvfem-ss-ab
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --time=00:28:00
#SBATCH --exclusive
#SBATCH --hint=nomultithread
#SBATCH --uenv-passthrough=ignore
#SBATCH --output=cvfem_ab_ss_%j.out
#SBATCH --error=cvfem_ab_ss_%j.err
#
# Usage:
#   sbatch --export=ALL,CVFEM_SRC=$PWD,CVFEM_REF=/path/to/built/reference jobs/ab_semistructured.sbatch
#
# THE SEMI-STRUCTURED A/B, which scripts/perf_regression.sh does not cover: it measures the flat
# packed throughput, and the semi-structured family is the one this refactor changed most --
# every signature, its per-thread scratch, and its threading model.
#
# Interleaved in one allocation with the order alternating, for the same reason the flat gate is:
# two separate jobs cannot tell a regression from a different node.
set -u
NEW=${CVFEM_SRC}
REF=${CVFEM_REF}
export CVFEM_SRC CVFEM_SFEM_INSTALL
source "$NEW/scripts/alps_env.sh"
export OMP_NUM_THREADS=72 OMP_PROC_BIND=true OMP_PLACES=cores
export SFEM_BENCH_MACROS=8 SFEM_BENCH_LEVELS=8 SFEM_BENCH_REPS=5 SFEM_BENCH_WARMUP=2
for rep in 1 2 3 4 5 6 7; do
    if [ $((rep % 2)) -eq 1 ]; then order="ref new"; else order="new ref"; fi
    for side in $order; do
        case $side in
            ref) B=$REF/build/cvfem_sshex8_bench ;;
            new) B=$NEW/build/cvfem_sshex8_bench ;;
        esac
        printf 'REP %s SIDE %s ' "$rep" "$side"
        cvfem_uenv "$B" 2>/dev/null | awk 'NR>1 {print $1, $2, $3, $4, $5, $6, $9, $10, $11, $12}' | tail -1
    done
done
echo "### done"

# Reading the output: the columns are ns/dof, so a LARGER number is slower. Take the median per
# side and compare; the same-side spread over the reps is the floor below which a difference says
# nothing. Measured once at 4,343,300 dof and 72 threads across this refactor: four arms faster
# (naive -0.6%, macro_local -0.8%, affine -1.3%, pgrad -1.6%) and four inside their own spread,
# worst +1.1% on the macro block diagonal against a 0.7% same-side spread.
