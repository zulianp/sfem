#!/usr/bin/env bash
#SBATCH --account=c40
#SBATCH --job-name=cvfem-solver-ab
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --time=00:28:00
#SBATCH --exclusive
#SBATCH --hint=nomultithread
#SBATCH --uenv-passthrough=ignore
#SBATCH --output=solver_ab_%j.out
#SBATCH --error=solver_ab_%j.err

# Two solver binaries against each other on one Grace socket, at sizes that saturate it.
#
#   sbatch --export=ALL,AB_NEW=/path/new/cvfem_hex8_ns_ssgmg,AB_REF=/path/ref/cvfem_hex8_ns_ssgmg \
#          jobs/solver_ab.sbatch
#
# This exists because scripts/perf_regression.sh compares the BENCH binary's kernels, and
# the question it cannot answer is what the change is worth to the solver's matvec once the
# Krylov method, the preconditioner and the constraints are in the picture. Here the
# comparable quantity is us_per_lin_it and the per-scope us/call in the trace, NOT t_solve:
# the two binaries reduce the nodal gradient in different orders, so they take different
# numbers of linear iterations on the same problem -- 1539 against 2930 has been observed on
# one case -- and a wall clock that divides by different iteration counts says nothing.
#
# Each run gets its own directory because the tracer writes smesh.trace.csv into the current
# one under a hard-coded name; feed those to python/cvfem_trace_report.py afterwards.
#
# The thread sweep is here rather than in a separate script because a case that fails at one
# thread count and passes at another is round-off fragile rather than regressed, and telling
# those apart is the first thing wanted when an A/B shows a convergence difference.
set -u
export OMP_PROC_BIND=true OMP_PLACES=cores

NEW=${AB_NEW:?set AB_NEW to the new solver binary}
REF=${AB_REF:?set AB_REF to the reference solver binary}
for d in "$NEW" "$REF"; do
    [ -x "$d" ] || { echo "solver_ab: no binary at $d" >&2; exit 1; }
done

UENV_ARGS=$(grep '^uenv start' ~/init.sh | sed 's/^uenv start //')
OUT=${OUT:-$SCRATCH/cvfem_solver_ab/$SLURM_JOB_ID}
mkdir -p "$OUT"
echo "host $(hostname)  $(date)"
echo "new $NEW"
echo "ref $REF"
echo "out $OUT"

# One run, reported as the line of numbers that survives comparison.
ab_run() {
    local tag=$1 thr=$2; shift 2
    local r=$OUT/$tag; mkdir -p "$r"
    ( cd "$r" && uenv run $UENV_ARGS -- env OMP_NUM_THREADS=$thr OMP_PROC_BIND=true \
        OMP_PLACES=cores "$@" > "$r/log" 2>&1 )
    local nd c li ts up
    nd=$(grep -o 'ndof: [0-9]*'             "$r/log" | head -1 | awk '{print $2}')
    c=$( grep -o 'newton_converged: [01]'   "$r/log" | head -1 | awk '{print $2}')
    li=$(grep -o 'lin_it_total: [0-9]*'     "$r/log" | head -1 | awk '{print $2}')
    ts=$(grep -o 't_solve: [0-9.]*'         "$r/log" | head -1 | awk '{print $2}')
    up=$(grep -o 'us_per_lin_it: [0-9.]*'   "$r/log" | head -1 | awk '{print $2}')
    printf "%-6s %-4s %-10s %-6s %-8s %-11s %s\n" \
        "$tag" "$thr" "${nd:-?}" "${c:-?}" "${li:-?}" "${ts:-?}" "${up:-?}"
}

PORT_ENV="SFEM_LX=4 SFEM_LY=1 SFEM_MU=0.01 SFEM_U=1 SFEM_CASE=poiseuille
          SFEM_ELEMENT_REFINE_LEVEL=1 SFEM_BOUNDARY_MASK=1 SFEM_OUTLET=dirichlet
          SFEM_PRESSURE_SIDESET=outlet SFEM_NL_MAX_IT=20"

echo
echo "=== thread sweep: the pressure port at p_bar = 3.0, the edge of the sweep ==="
printf "%-6s %-4s %-10s %-6s %-8s %-11s %s\n" tag thr ndof conv lin_it t_solve us_per_lin_it
for bin in new ref; do
    d=$( [ "$bin" = new ] && echo "$NEW" || echo "$REF" )
    for t in 1 8 18 36 72; do
        ab_run "p3_${bin}_$t" $t $PORT_ENV SFEM_N=12 SFEM_PRESSURE=3.0 "$d" "$OUT/p3_${bin}_$t/out"
    done
done

echo
echo "=== size sweep at 72 threads: the channel, to saturation ==="
printf "%-6s %-4s %-10s %-6s %-8s %-11s %s\n" tag thr ndof conv lin_it t_solve us_per_lin_it
for N in ${AB_SIZES:-24 40 56}; do
    for bin in new ref; do
        d=$( [ "$bin" = new ] && echo "$NEW" || echo "$REF" )
        ab_run "ch_${bin}_$N" 72 SFEM_LX=4 SFEM_LY=1 SFEM_MU=0.01 SFEM_U=1 \
            SFEM_CASE=poiseuille SFEM_N=$N SFEM_ELEMENT_REFINE_LEVEL=1 SFEM_NL_MAX_IT=20 \
            "$d" "$OUT/ch_${bin}_$N/out"
    done
done

echo
echo "=== manufactured solution at 72 threads ==="
printf "%-6s %-4s %-10s %-6s %-8s %-11s %s\n" tag thr ndof conv lin_it t_solve us_per_lin_it
for n in ${AB_MMS:-64}; do
    for bin in new ref; do
        d=$( [ "$bin" = new ] && echo "$NEW" || echo "$REF" )
        ab_run "mms_${bin}_$n" 72 SFEM_CASE=mms SFEM_N=$n SFEM_ELEMENT_REFINE_LEVEL=1 \
            SFEM_MU=1 SFEM_RHO=1 SFEM_NL_MAX_IT=20 "$d" "$OUT/mms_${bin}_$n/out"
    done
done

echo
echo "OUT=$OUT"
echo "=== DONE $(date) ==="
