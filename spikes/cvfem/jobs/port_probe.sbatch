#!/usr/bin/env bash
#SBATCH --account=c40
#SBATCH --job-name=cvfem-portp
#SBATCH --partition=debug
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=72
#SBATCH --time=00:20:00
#SBATCH --exclusive --hint=nomultithread --uenv-passthrough=ignore
#SBATCH --output=portp_%j.out
#SBATCH --error=portp_%j.err
#
# Why does the pressure-port case fail on Grace and converge on the laptop?
#
# Same source, same configuration, 33,124 dof: locally p_bar = 0 converges in 5 Newton steps
# and 1536 linear iterations, and on Grace it fails at newton 0 with the line search finding no
# decrease at any alpha down to 1/256. The operator is thread-deterministic -- measured
# bit-identical at 1, 2, 4 and 8 threads locally -- so this is not a reduction-order effect,
# and cross-machine guessing has already cost enough. This measures it where it fails.
#
# THREADS first, because it is the one difference that is cheap to sweep and the local sweep
# only went to 8. Then the two discriminators that localised the earlier regression: an exact
# linear solve, which takes the preconditioner out of the question, and the frozen Jacobian,
# under which the operator and the block diagonal agree by construction.
set -u
ROOT=${SLURM_SUBMIT_DIR:-$PWD}
D=${BIN:-$ROOT/build_solver/cvfem_hex8_ns_ssgmg}
[ -x "$D" ] || { echo "portp: no driver at $D" >&2; exit 1; }
OUT=${OUT:-$SCRATCH/cvfem_portp/${SLURM_JOB_ID:-manual}}
mkdir -p "$OUT"
export OMP_PROC_BIND=true OMP_PLACES=cores
U=$(grep '^uenv start' ~/init.sh | sed 's/^uenv start //')
echo "host $(hostname)  $(date)"
echo

B="SFEM_LX=4 SFEM_LY=1 SFEM_MU=0.01 SFEM_U=1 SFEM_CASE=poiseuille SFEM_N=12
   SFEM_BOUNDARY_MASK=1 SFEM_OUTLET=dirichlet SFEM_PRESSURE_SIDESET=outlet SFEM_NL_MAX_IT=40"

one() {  # one <tag> <threads> <env...>
    tag=$1; th=$2; shift 2
    log="$OUT/$tag.log"
    OMP_NUM_THREADS=$th stdbuf -oL -eL env $B "$@" uenv run $U -- "$D" "$OUT/$tag" > "$log" 2>&1
    printf '%-28s %4s %5s %7s %8s %8s  %s\n' "$tag" "$th" \
        "$(sed -n 's/^newton_converged: \([01]\).*/\1/p' "$log" | head -1)" \
        "$(sed -n 's/.*newton_total: \([0-9]*\).*/\1/p' "$log" | head -1)" \
        "$(sed -n 's/.*lin_it_total: \([0-9]*\).*/\1/p' "$log" | head -1)" \
        "$(sed -n 's/.*highest Re SOLVED = \([0-9.]*\).*/\1/p' "$log" | head -1)" \
        "$(grep -oE 'line search failed|diverged after [0-9]+ iterations' "$log" | head -1)"
}
hdr() { printf '%-28s %4s %5s %7s %8s %8s\n' tag thr conv newton lin_it hiRe; }

echo "== thread sweep, p_bar = 0 and 0.5 (the two that regressed)"
hdr
for PB in 0 0.5; do
    for TH in 1 8 36 72; do one "p${PB}_th${TH}" $TH SFEM_PRESSURE=$PB; done
done
echo
echo "== the two discriminators, at 72 threads"
hdr
for PB in 0 0.5; do
    one "p${PB}_direct"  72 SFEM_PRESSURE=$PB SFEM_PRECOND=direct
    one "p${PB}_frozen"  72 SFEM_PRESSURE=$PB SFEM_RC_EXACT_JAC=0
done
echo
echo "logs in $OUT"
echo "done $(date)"
