#!/usr/bin/env bash
#SBATCH --account=c40
#SBATCH --job-name=cvfem-packsz
#SBATCH --partition=debug
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=72
#SBATCH --time=00:28:00
#SBATCH --exclusive --hint=nomultithread --uenv-passthrough=ignore
#SBATCH --output=packsz3_%j.out
#SBATCH --error=packsz3_%j.err
#
# Is the flat baseline actually tuned?
#
# Every semi-structured-vs-flat number so far compares against the flat path at its DEFAULT
# pack size of 2048, which nothing in this session verified is its best. If a different pack
# is faster then the gap is larger than reported and the target moved; if 2048 is the
# optimum then the comparison stands and that is worth knowing rather than assuming.
#
# The bench rather than the solver, because this is a kernel property: one allocation, the
# same mesh, ten timed applies after five warmups, and nothing of the Krylov method or the
# preconditioner in the way.
#
# The pack size sets how many elements share a staging buffer and therefore how much of a
# pack's gather is ghost traffic -- the same trade the macro level makes on the
# semi-structured path, where level 8 fits its 729-node staging in cache and level 16 does
# not. So a sweep here is the flat twin of the level sweep.
set -u
ROOT=${SLURM_SUBMIT_DIR:-$PWD}
B=${BIN:-$ROOT/build_solver/cvfem_hex8_ns_upwind_bench}
[ -x "$B" ] || { echo "packsz: no bench at $B" >&2; exit 1; }
OUT=${OUT:-$SCRATCH/cvfem_packsz/${SLURM_JOB_ID:-manual}}
mkdir -p "$OUT"
export OMP_NUM_THREADS=72 OMP_PROC_BIND=true OMP_PLACES=cores
U=$(grep '^uenv start' ~/init.sh | sed 's/^uenv start //')
echo "host $(hostname)  threads 72  $(date)"
echo "n=100 -> 4,121,204 dof, the size the semi-structured comparison was made at"
echo
# Three passes over the whole sweep, best of three reported.
#
# One measurement per configuration cannot separate this band: a first pass read 909.9, 829.1,
# 904.7 and 880.6 MDOF/s at 256, 512, 1024 and 2048 -- a 9% spread with no monotone trend,
# which is inside this machine's run-to-run variation. Best-of-three is the right estimator
# here for the reason the performance gate uses it: background load can only make a run
# slower, so the fastest observation is the closest to the machine's actual capability.
#
# Passing over the whole sweep three times rather than repeating each configuration three
# times in place also spreads each one across three positions in the run order, which is the
# effect that cannot be averaged out of a single sequence.
run_one() {  # run_one <flags...>
    ( cd "$OUT" && uenv run $U -- "$B" --n 100 --repeat 10 --warmup 5 "$@" 2>/dev/null \
        | grep -m1 '  MDOF/s:' | awk '{printf "%.1f", $2}' )
}
declare -A JBEST RBEST
for pass in 1 2 3; do
    for ps in 256 512 1024 2048 4096 8192 16384; do
        j=$(run_one --pack-size $ps --layout packed --jac-action --rhie-chow --boundary)
        r=$(run_one --pack-size $ps --layout packed --rhie-chow --pgrad-per-apply --boundary)
        cur=${JBEST[$ps]:-0}; awk -v a="$j" -v b="$cur" 'BEGIN{exit !(a>b)}' && JBEST[$ps]=$j
        cur=${RBEST[$ps]:-0}; awk -v a="$r" -v b="$cur" 'BEGIN{exit !(a>b)}' && RBEST[$ps]=$r
    done
    echo "  pass $pass done $(date +%H:%M:%S)"
done
echo
printf "%-12s %14s %14s\n" pack_size "jac MDOF/s" "res MDOF/s"
for ps in 256 512 1024 2048 4096 8192 16384; do
    printf "%-12s %14s %14s\n" "$ps" "${JBEST[$ps]:-FAIL}" "${RBEST[$ps]:-FAIL}"
done
echo
echo "OUT=$OUT"
echo "=== DONE $(date) ==="
