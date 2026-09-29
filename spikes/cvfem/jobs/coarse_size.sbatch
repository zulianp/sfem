#!/usr/bin/env bash
#SBATCH --account=c40
#SBATCH --job-name=cvfem-csize
#SBATCH --partition=debug
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=72
#SBATCH --time=00:29:00
#SBATCH --exclusive --hint=nomultithread --uenv-passthrough=ignore
#SBATCH --output=csize_%j.out
#SBATCH --error=csize_%j.err

# HOW BIG SHOULD THE COARSEST LEVEL BE, AT CONSTANT FINE RESOLUTION.
#
# The hierarchy bottoms out at the MACRO mesh, so the coarsest order is fixed by N alone and
# not by the refinement level: N macro elements per direction give (N+1)^3 * 4 coarse dofs,
# while the fine size is (N * L + 1)^3 * 4. Holding N * L fixed therefore varies the coarse
# order by orders of magnitude with the fine problem bit-identical in size.
#
# Why it matters here: at N = 16, L = 4 the coarsest level is 19,652 dofs, which is a DENSE LU
# of order 19,652 -- a 3.1 GB matrix. Measured on the cavity at 1,098,500 dof after the
# factorisation cache landed, coarse_solve is 22.703 s (33.6%, 148 calls at 153 ms) and
# coarse_factor 17.906 s (26.5%, 5 calls at 3.58 s): three fifths of the run is the coarse
# level. The solve is two triangular sweeps streaming that 3.1 GB, so it is memory-bound and
# quadratic, and the factorisation is cubic.
#
# The tension this measures, stated in make_gmg_levels: a macro mesh of a handful of cells may
# not approximate a convection-dominated operator at all, and because the coarsest level is
# solved to tolerance the cycle prolongs a confidently wrong correction. So the cheap end can
# lose. At toy size (2,916 dof, 4 stages) it did not -- linear iterations fell 47, 37, 24 as
# the coarse order fell 500, 108, 32 -- but a toy at low Re is not evidence for 1,098,500 dof
# at Re 100, which is what this job is for.
#
# ARMS entries are "label|N|LEVEL". Every arm must print the same ndof or the comparison is
# void, so the table prints ndof per row rather than assuming it.

set -uo pipefail
cd "$SLURM_SUBMIT_DIR" || exit 1
source scripts/alps_env.sh || exit 1

D=$PWD/build_solver/cvfem_hex8_ns_ssgmg
OUT=${OUT:-$SCRATCH/cvfem_csize/${SLURM_JOB_ID:-manual}}
mkdir -p "$OUT"

# One line: a literal newline inside the bash -lc string splits it in two and the driver then
# runs without SFEM_CASE, prints its usage, and every cell of the table becomes a dash.
BASE=${BASE:-"SFEM_CASE=cavity SFEM_FGMRES=1 SFEM_GMG=1 SFEM_ENABLE_OUTPUT=0 SFEM_MU=0.01 SFEM_NL_MAX_IT=8"}
echo "### BASE: $BASE"

ARMS=${ARMS:-"n32l2|32|2 n16l4|16|4 n8l8|8|8 n4l16|4|16"}

printf '%-8s %4s %4s %9s %8s %7s %7s %9s %6s %9s %9s %7s\n' \
       arm N L ndof coarse newton linear cfactor n_cf csolve precond wall_s
for a in $ARMS; do
    # A fourth field carries per-arm environment, so a before/after that differs in BOTH the
    # hierarchy shape and the freeze defaults can run inside ONE allocation. Across allocations
    # this bench has measured the same binary at 510.3, 484.2 and 225.5 MDOF/s, so a comparison
    # split over two jobs is not a comparison.
    IFS='|' read -r lbl n l extra <<<"$a"
    # Commas rather than spaces inside the field, because ARMS itself is space-separated.
    extra=${extra//,/ }
    log=$OUT/$lbl.log
    t0=$(date +%s)
    # Per-arm cap against the reference: the N=16 arm is ~53 s with the factorisation cached,
    # so an arm past 400 s is stuck rather than slow, and uncapped one such arm takes the whole
    # wall clock and the job measures a single row.
    timeout "${ARM_CAP:-400}" \
      srun -u --ntasks=1 --cpus-per-task=72 --uenv="$CVFEM_UENV" --view="$CVFEM_VIEW" \
      bash -lc "cd $PWD && OMP_NUM_THREADS=72 OMP_PROC_BIND=true OMP_PLACES=cores env $BASE SFEM_N=$n SFEM_ELEMENT_REFINE_LEVEL=$l ${extra:-SFEM_DUMMY=0} $D $OUT/out_$lbl" \
      > "$log" 2>&1
    wall=$(( $(date +%s) - t0 ))

    ndof=$(sed -n 's/.*ndof: \([0-9]*\).*/\1/p' "$log" | head -1)
    # The coarsest order, taken from the LU's own rank report rather than recomputed here, so
    # the number is the one the factorisation actually saw.
    crs=$(sed -n 's/.*coarse LU: [0-9]* of \([0-9]*\) directions.*/\1/p' "$log" | head -1)
    conv=$(sed -n 's/.*newton_converged: \([0-9]*\).*/\1/p' "$log" | head -1)
    nwt=$(sed -n 's/.*newton_total: \([0-9]*\).*/\1/p' "$log" | head -1)
    lin=$(sed -n 's/.*lin_it_total: \([0-9]*\).*/\1/p' "$log" | head -1)
    read -r n_cf t_cf <<<"$(awk '/^ +coarse_factor /{print $3, $2}' "$log" | head -1)"
    t_cs=$(awk '/^ +coarse_solve /{print $2}' "$log" | head -1)
    t_pc=$(awk '/^ +precond_total /{print $2}' "$log" | head -1)
    [ "${conv:-0}" = "1" ] || lbl="$lbl*"
    printf '%-8s %4s %4s %9s %8s %7s %7s %9s %6s %9s %9s %7s\n' \
           "$lbl" "$n" "$l" "${ndof:--}" "${crs:--}" "${nwt:--}" "${lin:--}" \
           "${t_cf:--}" "${n_cf:--}" "${t_cs:--}" "${t_pc:--}" "$wall"
done
echo
echo "* = newton_converged 0; that row's counts are a floor, not a comparable result"
echo "logs in $OUT"
