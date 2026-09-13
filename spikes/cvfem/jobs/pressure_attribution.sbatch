#!/usr/bin/env bash
#SBATCH --account=c40
#SBATCH --job-name=cvfem-pattr
#SBATCH --partition=normal
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=72
#SBATCH --time=01:30:00
#SBATCH --exclusive --hint=nomultithread --uenv-passthrough=ignore
#SBATCH --output=pattr_%j.out
#SBATCH --error=pattr_%j.err
#
# WHICH VARIABLE ACTUALLY CAUSED THE PORT FAILURES: the linear solver, or the size?
#
# The story this job exists to test was: a prescribed pressure far from the one the flow
# produces leaves the seed O(p_bar) away from the answer IN THE PRESSURE LEVEL, Newton
# cannot cross that, and a continuation or a level predictor fixes it.
#
# The control in job 4655282 does not support that. With both mechanisms OFF -- the code
# exactly as it stood when p_bar = 1.0 failed -- p_bar = 1.0 CONVERGES in three Newton steps
# to u_linf 5.03e-14. Every p_bar from -3 to +10 converges. So Newton's basin was never the
# problem at that configuration.
#
# But two things changed at once between that control and the original failure, and the job
# cannot tell them apart:
#
#     original failure :  33,124 dof,  iterative
#     passing control  :  10,692 dof,  SFEM_PRECOND=direct (dense LU)
#
# This job changes them ONE AT A TIME. Four cells, and the diagonal of the 2x2 is the answer:
#
#   A  10,692  direct     -- known to pass (reproduced here as the anchor)
#   B  10,692  iterative  -- if this FAILS, the linear solver is the whole story and size is
#                            irrelevant; the continuation works by keeping each linear system
#                            inside what the Krylov method can handle, not by shortening a
#                            Newton step.
#   C  33,124  direct     -- if this PASSES, size is irrelevant with an exact solve, which
#                            says the same thing from the other side. Needs the dof cap
#                            raised: a dense 33,124^2 double matrix is 8.8 GB, which does not
#                            fit a laptop and does fit a Grace node. That is the only reason
#                            this cell can be run at all and the reason it is run here.
#   D  33,124  iterative  -- the original failing configuration, reproduced verbatim.
#
# Each cell runs with the continuation OFF (SFEM_P_PREDICT=0 SFEM_P_STAGES=0) and again with
# it ON at the default, so the rescue is measured in whichever cell actually fails rather
# than asserted from the one that does not.
set -u
ROOT=${SLURM_SUBMIT_DIR:-$PWD}
D=${BIN:-$ROOT/build_solver/cvfem_hex8_ns_ssgmg}
[ -x "$D" ] || { echo "pattr: no driver at $D" >&2; exit 1; }
OUT=${OUT:-$SCRATCH/cvfem_pattr/${SLURM_JOB_ID:-manual}}
mkdir -p "$OUT"
export OMP_NUM_THREADS=72 OMP_PROC_BIND=true OMP_PLACES=cores
U=$(grep '^uenv start' ~/init.sh | sed 's/^uenv start //')
echo "host $(hostname)  cores 72 (Grace Neoverse V2)  $(date)"
echo

# SFEM_FGMRES=1 named explicitly, even though it is now the driver default. The first run of
# this job left it unset while the default was still `gmg ? 1 : 0`, so every "iterative" cell
# ran BiCGStab -- which breaks down on this saddle-point system in ways that look like a
# modelling failure and are not, and the resulting table had to be discarded. Naming it here
# means this script keeps measuring the intended solver even if a default moves again.
BASE="SFEM_CASE=poiseuille SFEM_LX=4 SFEM_LY=1 SFEM_LZ=1 SFEM_MU=0.01 SFEM_U=1
      SFEM_BOUNDARY_MASK=1 SFEM_OUTLET=dirichlet SFEM_PRESSURE_SIDESET=outlet
      SFEM_ELEMENT_REFINE_LEVEL=1 SFEM_FGMRES=1"

run_one() {
    local tag=$1; shift
    local log="$OUT/$tag.log"
    stdbuf -oL -eL env $BASE "$@" uenv run $U -- "$D" "$OUT/$tag" > "$log" 2>&1
    local rc=$? ndof conv nt li u p why
    ndof=$(sed -n 's/.*ndof: \([0-9]*\).*/\1/p' "$log" | head -1)
    conv=$(sed -n 's/.*newton_converged: \([0-9]*\).*/\1/p' "$log" | head -1)
    nt=$(sed -n 's/.*newton_total: \([0-9]*\).*/\1/p' "$log" | head -1)
    li=$(sed -n 's/.*lin_it_total: \([0-9]*\).*/\1/p' "$log" | head -1)
    u=$(sed -n 's/^u_linf: \([^ ]*\).*/\1/p' "$log" | head -1)
    p=$(sed -n 's/.*p_linf: \(.*\)$/\1/p' "$log" | head -1)
    # WHY it stopped, not just that it did. A converged run and a run that abandoned a stage
    # after a diverging Krylov solve are different findings and the table has to separate them.
    why=$(grep -oE "linear solve diverged|line search failed|refuses [0-9]+ dofs|SHORT OF TARGET" "$log" | head -1)
    printf '%-22s %9s %5s %7s %9s %14s %14s  %s\n' \
        "$tag" "${ndof:--}" "${conv:-$rc}" "${nt:--}" "${li:--}" "${u:--}" "${p:--}" "${why:-ok}"
}
HDR() { printf '%-22s %9s %5s %7s %9s %14s %14s  %s\n' \
        cell ndof conv newton lin_it u_linf p_linf why; }

# 8.8 GB dense factor. Stated here rather than buried: this is the cell that cannot be run
# anywhere but on a node with real memory, and it is the one that isolates the solver.
BIGDIRECT="SFEM_PRECOND=direct SFEM_GMG=0 SFEM_DIRECT_MAX_DOF=40000"
SMALLDIRECT="SFEM_PRECOND=direct SFEM_GMG=0"

for PB in 1.0 3.0 10.0; do
    echo "== p_bar = $PB"
    HDR
    for CONT in "off SFEM_P_PREDICT=0 SFEM_P_STAGES=0" "on SFEM_P_PREDICT=1 SFEM_P_STAGES=4"; do
        set -- $CONT; c=$1; shift
        run_one "A_small_direct_$c"  SFEM_N=8  $SMALLDIRECT "$@" SFEM_PRESSURE=$PB
        run_one "B_small_iter_$c"    SFEM_N=8  SFEM_GMG=0   "$@" SFEM_PRESSURE=$PB
        run_one "C_big_direct_$c"    SFEM_N=12 $BIGDIRECT   "$@" SFEM_PRESSURE=$PB
        run_one "D_big_iter_$c"      SFEM_N=12 SFEM_GMG=0   "$@" SFEM_PRESSURE=$PB
    done
    echo
done
echo "logs in $OUT"
echo "done $(date)"
