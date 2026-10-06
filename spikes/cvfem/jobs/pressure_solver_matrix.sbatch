#!/usr/bin/env bash
#SBATCH --account=c40
#SBATCH --job-name=cvfem-psolv
#SBATCH --partition=normal
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=72
#SBATCH --time=01:30:00
#SBATCH --exclusive --hint=nomultithread --uenv-passthrough=ignore
#SBATCH --output=psolv_%j.out
#SBATCH --error=psolv_%j.err
#
# THE PRESCRIBED-PRESSURE CASES ACROSS THE SOLVER STACKS THAT ACTUALLY EXIST.
#
# Every previous table in this investigation named at most half a solver, and each time the
# missing half was what produced the result:
#
#   * "the default solver"    was BiCGStab, which breaks down on this saddle-point system in
#                             ways that look like a modelling failure and are not;
#   * "FGMRES without MG"     was FGMRES + point block-Jacobi, the weakest preconditioner in
#                             the driver -- measured on the standalone smoother probe,
#                             block-Jacobi and SIMPLE diverge at omega >= 0.7 while Vanka at
#                             omega = 1 converges at 0.8279.
#
# So a "without multigrid" run on a FLAT mesh is not a fair without-multigrid run at all:
# `simple` and `vanka` need a micro-element lattice, so on a flat mesh the only choices are
# bjacobi and direct, and the configuration is crippled by construction rather than by the
# boundary condition under test.
#
# Every cell here is therefore SEMI-STRUCTURED, so the preconditioner is a free choice, and
# every cell names its whole stack. Three stacks, with and without multigrid as Patrick asked:
#
#   mg        FGMRES + geometric multigrid (Vanka smoother)   -- the production solver
#   vanka     FGMRES + Vanka, no multigrid                    -- the fair without-MG run
#   bjacobi   FGMRES + block-Jacobi, no multigrid             -- the weak baseline, kept
#                                                                because it is what the
#                                                                verification matrix has been
#                                                                running by default and its
#                                                                numbers are the ones that
#                                                                need explaining
#
# crossed with the continuation off and on, at p_bar spanning the range that used to fail.
# The question each column answers: does this stack need the continuation, and what does the
# continuation cost it when it does not.
set -u
ROOT=${SLURM_SUBMIT_DIR:-$PWD}
D=${BIN:-$ROOT/build_solver/cvfem_hex8_ns_ssgmg}
[ -x "$D" ] || { echo "psolv: no driver at $D" >&2; exit 1; }
OUT=${OUT:-$SCRATCH/cvfem_psolv/${SLURM_JOB_ID:-manual}}
mkdir -p "$OUT"
export OMP_NUM_THREADS=72 OMP_PROC_BIND=true OMP_PLACES=cores
U=$(grep '^uenv start' ~/init.sh | sed 's/^uenv start //')
echo "host $(hostname)  cores 72 (Grace Neoverse V2)  $(date)"
echo

# Nothing is left to a default. SFEM_FGMRES, SFEM_GMG and SFEM_PRECOND are all named in every
# cell below, because each of them has silently decided a previous result in this sequence.
BASE="SFEM_CASE=poiseuille SFEM_LX=4 SFEM_LY=1 SFEM_LZ=1 SFEM_MU=0.01 SFEM_U=1
      SFEM_BOUNDARY_MASK=1 SFEM_OUTLET=dirichlet SFEM_PRESSURE_SIDESET=outlet
      SFEM_FGMRES=1 SFEM_N=3 SFEM_ELEMENT_REFINE_LEVEL=4"

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
    why=$(grep -oE "linear solve diverged|line search failed|SHORT OF TARGET|needs a semi-structured|refuses [0-9]+ dofs" "$log" | head -1)
    printf '%-22s %8s %5s %7s %9s %9s %14s %14s  %s\n' \
        "$tag" "${ndof:--}" "${conv:-$rc}" "${nt:--}" "${li:--}" "${ts:--}" "${u:--}" "${p:--}" "${why:-ok}"
}
HDR() { printf '%-22s %8s %5s %7s %9s %9s %14s %14s  %s\n' \
        cell ndof conv newton lin_it t_solve u_linf p_linf why; }

MG="SFEM_GMG=1"
VANKA="SFEM_GMG=0 SFEM_PRECOND=vanka SFEM_GMG_OMEGA=1.0"
BJAC="SFEM_GMG=0 SFEM_PRECOND=bjacobi"
OFF="SFEM_P_PREDICT=0 SFEM_P_STAGES=0"
ON="SFEM_P_PREDICT=1 SFEM_P_STAGES=4"

# p_bar = -0.16 is the level the flow produces on its own, so it is the zero-severity control:
# the row where the boundary condition asks for nothing unusual. Every claim about the rows
# above it is relative to this one.
for PB in -0.16 1.0 3.0 10.0; do
    echo "== p_bar = $PB   (p_linf must be |p_bar + 0.16| wherever conv=1)"
    HDR
    run_one "mg_off_p$PB"       $MG    $OFF SFEM_PRESSURE=$PB
    run_one "mg_on_p$PB"        $MG    $ON  SFEM_PRESSURE=$PB
    run_one "vanka_off_p$PB"    $VANKA $OFF SFEM_PRESSURE=$PB
    run_one "vanka_on_p$PB"     $VANKA $ON  SFEM_PRESSURE=$PB
    run_one "bjacobi_off_p$PB"  $BJAC  $OFF SFEM_PRESSURE=$PB
    run_one "bjacobi_on_p$PB"   $BJAC  $ON  SFEM_PRESSURE=$PB
    echo
done
echo "logs in $OUT"
echo "done $(date)"
