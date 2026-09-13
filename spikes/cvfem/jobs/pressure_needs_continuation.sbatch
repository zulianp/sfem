#!/usr/bin/env bash
#SBATCH --account=c40
#SBATCH --job-name=cvfem-pneed
#SBATCH --partition=normal
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=72
#SBATCH --time=01:00:00
#SBATCH --exclusive --hint=nomultithread --uenv-passthrough=ignore
#SBATCH --output=pneed_%j.out
#SBATCH --error=pneed_%j.err
#
# DOES THE PRODUCTION SOLVER NEED THE PRESSURE CONTINUATION AT ALL?
#
# This is the control every table so far has been missing, and without it the change is
# unjustified rather than merely unexplained.
#
# What is established, and why it is not enough:
#
#   * With an EXACT linear solve (dense LU, 10,692 dof) the old code solves p_bar from -3 to
#     +10 in three Newton steps. A prescribed pressure is not a Newton difficulty.
#   * With FGMRES and NO multigrid, at the same 10,692 dof, the old code fails at p_bar = 1:
#     the linear solve runs out at its 5000-iteration cap, the step it returns is meaningless
#     and the line search abandons it. Same size, same everything, only the linear solver
#     differs -- so the failure is in the linear solve, not in the size.
#   * Nothing has yet run the configuration the spike actually uses -- FGMRES preconditioned
#     by MULTIGRID on a semi-structured mesh -- with the continuation OFF.
#
# That last gap matters more than the rest of the sequence. If the production solver handles
# p_bar = 10 with no continuation, the continuation is compensating for a weak preconditioner
# in a configuration nobody runs, and the honest conclusion is to fix the preconditioner
# rather than ship a workaround. If it does not, the continuation earns its place and this
# table says by how much.
#
# Four cells per pressure, the axes being {multigrid, none} x {continuation, none}:
#
#   mg_off     semi-structured + GMG, continuation off   <- THE MISSING CONTROL
#   mg_on      semi-structured + GMG, continuation on
#   nomg_off   flat, no multigrid, continuation off      <- reproduces the known failure
#   nomg_on    flat, no multigrid, continuation on       <- does it rescue a weak precond?
#
# The dense-LU cells of the previous version are gone. They asked whether the failure was
# size-dependent; the small FGMRES cell answered that at a stroke, and an 8.8 GB
# factorisation that blocked the job for twelve minutes to re-answer a settled question is
# cost with no information behind it.
set -u
ROOT=${SLURM_SUBMIT_DIR:-$PWD}
D=${BIN:-$ROOT/build_solver/cvfem_hex8_ns_ssgmg}
[ -x "$D" ] || { echo "pneed: no driver at $D" >&2; exit 1; }
OUT=${OUT:-$SCRATCH/cvfem_pneed/${SLURM_JOB_ID:-manual}}
mkdir -p "$OUT"
export OMP_NUM_THREADS=72 OMP_PROC_BIND=true OMP_PLACES=cores
U=$(grep '^uenv start' ~/init.sh | sed 's/^uenv start //')
echo "host $(hostname)  cores 72 (Grace Neoverse V2)  $(date)"
echo

# SFEM_FGMRES=1 named explicitly. It is the driver default now, but a job that relies on a
# default to choose its Krylov method is exactly how the previous run of this experiment came
# to measure BiCGStab and produce a table that had to be discarded.
BASE="SFEM_CASE=poiseuille SFEM_LX=4 SFEM_LY=1 SFEM_LZ=1 SFEM_MU=0.01 SFEM_U=1
      SFEM_BOUNDARY_MASK=1 SFEM_OUTLET=dirichlet SFEM_PRESSURE_SIDESET=outlet SFEM_FGMRES=1"

run_one() {
    local tag=$1; shift
    local log="$OUT/$tag.log"
    stdbuf -oL -eL env $BASE "$@" uenv run $U -- "$D" "$OUT/$tag" > "$log" 2>&1
    local rc=$? ndof conv nt li ts u p why
    ndof=$(sed -n 's/.*ndof: \([0-9]*\).*/\1/p' "$log" | head -1)
    conv=$(sed -n 's/.*newton_converged: \([0-9]*\).*/\1/p' "$log" | head -1)
    nt=$(sed -n 's/.*newton_total: \([0-9]*\).*/\1/p' "$log" | head -1)
    li=$(sed -n 's/.*lin_it_total: \([0-9]*\).*/\1/p' "$log" | head -1)
    ts=$(sed -n 's/.*t_solve: \([0-9.]*\) s.*/\1/p' "$log" | head -1)
    u=$(sed -n 's/^u_linf: \([^ ]*\).*/\1/p' "$log" | head -1)
    p=$(sed -n 's/.*p_linf: \(.*\)$/\1/p' "$log" | head -1)
    why=$(grep -oE "linear solve diverged|line search failed|SHORT OF TARGET|refuses [0-9]+ dofs" "$log" | head -1)
    printf '%-16s %9s %5s %7s %9s %9s %14s %14s  %s\n' \
        "$tag" "${ndof:--}" "${conv:-$rc}" "${nt:--}" "${li:--}" "${ts:--}" "${u:--}" "${p:--}" "${why:-ok}"
}
HDR() { printf '%-16s %9s %5s %7s %9s %9s %14s %14s  %s\n' \
        cell ndof conv newton lin_it t_solve u_linf p_linf why; }

MG="SFEM_N=2 SFEM_ELEMENT_REFINE_LEVEL=8 SFEM_GMG=1"
NOMG="SFEM_N=8 SFEM_ELEMENT_REFINE_LEVEL=1 SFEM_GMG=0"
OFF="SFEM_P_PREDICT=0 SFEM_P_STAGES=0"
ON="SFEM_P_PREDICT=1 SFEM_P_STAGES=4"

for PB in 1.0 3.0 10.0; do
    echo "== p_bar = $PB   (p_linf must be |p_bar + 0.16| wherever conv=1)"
    HDR
    run_one "mg_off_p$PB"    $MG   $OFF SFEM_PRESSURE=$PB
    run_one "mg_on_p$PB"     $MG   $ON  SFEM_PRESSURE=$PB
    run_one "nomg_off_p$PB"  $NOMG $OFF SFEM_PRESSURE=$PB
    run_one "nomg_on_p$PB"   $NOMG $ON  SFEM_PRESSURE=$PB
    echo
done
echo "logs in $OUT"
echo "done $(date)"
