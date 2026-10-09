#!/usr/bin/env bash
#SBATCH --account=c40
#SBATCH --job-name=cvfem-tet4-ab
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --time=00:28:00
#SBATCH --exclusive
#SBATCH --hint=nomultithread
#SBATCH --uenv-passthrough=ignore
#SBATCH --output=cvfem_tet4_ab_%j.out
#SBATCH --error=cvfem_tet4_ab_%j.err
#
# THE TET4 A/B, which the flat and semi-structured gates do not cover: scripts/perf_regression.sh
# measures the packed HEX8 throughput and jobs/ab_semistructured.sbatch the semi-structured
# family, and neither of them links a TET4 kernel.
#
# It exists for one change in particular. python/cvfem_codegen.py's set_stable_pow pins the
# reciprocal spelling to the scalar type, and TET4 had it off, which left `const auto x0 =
# 1.0/det;` in thirty-five places -- a double division, and with `auto` a double variable, so a
# single-precision instantiation would have computed most of its work in double. Turning it on
# is what DESIGN.md's third correction needs; the generator's own note said the flip "belongs in
# its own commit with a benchmark behind it", and this is that benchmark.
#
# Interleaved in one allocation with the order alternating, for the reason the other two gates
# are: two separate jobs cannot tell a change from a different node.
#
# Usage:
#   sbatch --export=ALL,CVFEM_SRC=$PWD,CVFEM_REF=/path/to/built/reference jobs/ab_tet4.sbatch
#
# SLURM_SUBMIT_DIR, not BASH_SOURCE: Slurm copies the batch script to
# /var/spool/slurmd/job<id>/slurm_script.
set -u
: "${CVFEM_SRC:=${SLURM_SUBMIT_DIR:-$PWD}}"
: "${CVFEM_REF:?set CVFEM_REF to a built reference tree}"
: "${CVFEM_SFEM_INSTALL:=$SCRATCH/installations/sfem-omp}"
export CVFEM_SRC CVFEM_SFEM_INSTALL
source "$CVFEM_SRC/scripts/alps_env.sh"
cd "$CVFEM_SRC"

NEWB="$CVFEM_SRC/build/cvfem_tet4_ns_upwind_bench"
REFB="$CVFEM_REF/build/cvfem_tet4_ns_upwind_bench"
for b in "$NEWB" "$REFB"; do
    [ -x "$b" ] || { echo "not executable: $b" >&2; exit 2; }
done

export OMP_NUM_THREADS=72 OMP_PROC_BIND=true OMP_PLACES=cores
N=${N:-96}
REPEAT=${REPEAT:-20}
REPS=${REPS:-7}

# Three operations: a residual winner need not be an assembly winner, which is what the HEX8
# campaign found.
for op in "" "--assemble" "--jac-action"; do
    echo "### op ${op:-residual}"
    for rep in $(seq 1 "$REPS"); do
        if [ $((rep % 2)) -eq 1 ]; then order="ref new"; else order="new ref"; fi
        for side in $order; do
            case $side in
                ref) B=$REFB ;;
                new) B=$NEWB ;;
            esac
            printf 'REP %s SIDE %s ' "$rep" "$side"
            # MELEM/s is the rate this bench reports per element; the checksum is printed so a
            # change in the arithmetic cannot hide behind a timing comparison.
            cvfem_uenv "$B" --n "$N" --repeat "$REPEAT" --warmup 3 $op 2>/dev/null |
                awk '/MELEM\/s:/ {m=$2} /checksum:/ {c=$2} END {print m, c}'
        done
    done
done
echo "### done"
