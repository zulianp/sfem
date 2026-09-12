#!/usr/bin/env bash
#SBATCH --account=c40
#SBATCH --job-name=cvfem-600M
#SBATCH --partition=normal
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=72
#SBATCH --time=08:00:00
#SBATCH --exclusive --hint=nomultithread --uenv-passthrough=ignore
#SBATCH --output=scale600_%j.out
#SBATCH --error=scale600_%j.err
#
# How far does the packed semi-structured operator go, and what does a matvec cost there?
#
# 600M dof is a memory question before it is a speed one, and it is the regime the
# semi-structured path exists for. At 592M dof its connectivity is 0.84 GB against the flat
# equivalent's 4.71 GB of element table plus 11.78 GB of precomputed affine geometry -- a mesh
# that does not fit becomes one that occupies under a gigabyte.
#
# A FEW GMRES steps, not a solve. SFEM_NL_MAX_IT=1 with SFEM_LSOLVE_MAX_IT=5 makes this a
# measurement of the operator in its real context -- inside the Krylov loop, with the
# constraints, the boundary closure and the preconditioner around it -- rather than of the
# nonlinear problem, which at these sizes would tell us about the solver and not the kernel.
# The restart is 5 for the same reason it has to be: FGMRES stores restart+2 vectors and one
# state vector is 4.74 GB at this size, so the default 30 would ask for 142 GB.
#
# The ladder runs small to large and each size writes its own line, so a run killed by memory
# at the top still reports everything below it -- which is the point of a ladder rather than a
# single heroic run.
set -u
D=${DRIVER:?set DRIVER}
OUT=${OUT:-$SCRATCH/cvfem_scale600/${SLURM_JOB_ID:-manual}}
mkdir -p "$OUT"
export OMP_NUM_THREADS=72 OMP_PROC_BIND=true OMP_PLACES=cores
U=$(grep '^uenv start' ~/init.sh | sed 's/^uenv start //')
echo "host $(hostname)  threads 72  $(date)"
echo "out  $OUT"
free -g 2>/dev/null | head -2
echo
printf "%-10s %-12s %-10s %-10s %-12s %-12s %s\n" N ndof lin_it us_per_it peakRSS_GB apply_us setup_s

one() {  # one <N> <level> <packsize>
    local n=$1 lvl=$2 ps=$3
    local r=$OUT/N$n; mkdir -p "$r"
    ( cd "$r" && /usr/bin/time -f "%M" -o "$r/rss" \
        uenv run $U -- env OMP_NUM_THREADS=72 OMP_PROC_BIND=true OMP_PLACES=cores \
        SFEM_CASE=cavity_reg SFEM_NX=$n SFEM_NY=$n SFEM_NZ=$n \
        SFEM_ELEMENT_REFINE_LEVEL=$lvl SFEM_SS_PACK_SIZE=$ps \
        SFEM_GMG=0 SFEM_FGMRES=1 SFEM_FGMRES_RESTART=5 SFEM_PRECOND=bjacobi \
        SFEM_NL_MAX_IT=1 SFEM_LSOLVE_MAX_IT=5 SFEM_MU=0.01 SFEM_ENABLE_OUTPUT=0 \
        "$D" "$r/out" > "$r/log" 2>&1 )
    local nd li up rss ap setup
    nd=$( grep -o 'ndof: [0-9]*'         "$r/log" | head -1 | awk '{print $2}')
    li=$( grep -o 'lin_it_total: [0-9]*' "$r/log" | head -1 | awk '{print $2}')
    up=$( grep -o 'us_per_lin_it: [0-9.]*' "$r/log" | head -1 | awk '{print $2}')
    rss=$(awk '/^[0-9]+$/{v=$1} END{if (v) printf "%.1f", v/1048576}' "$r/rss" 2>/dev/null)
    # The element sweep's own scope, which is the operator without the Krylov method around it.
    # The trace CSV is name,calls,total,avg -- $4 is ALREADY the per-call average, so it needs
    # scaling to microseconds and nothing else. Dividing it by the call count reads six times
    # the throughput the same kernel measures elsewhere, which is how this was caught.
    ap=$(awk -F, '/apply_macro_local_hoisted/{printf "%.0f", $4*1e6}' "$r/smesh.trace.csv" 2>/dev/null)
    setup=$(grep -o 'cvfem_hex8_ns_ssgmg: [0-9.]*' "$r/log" | head -1 | awk '{print $2}')
    printf "%-10s %-12s %-10s %-10s %-12s %-12s %s\n" \
        "$n" "${nd:-OOM/fail}" "${li:-?}" "${up:-?}" "${rss:-?}" "${ap:-?}" "${setup:-?}"
}

for n in 16 32 48 56 64 66; do one $n 8 8; done
echo
echo "OUT=$OUT"
echo "=== DONE $(date) ==="
