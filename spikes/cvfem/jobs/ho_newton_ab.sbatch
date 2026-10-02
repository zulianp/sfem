#!/usr/bin/env bash
#SBATCH --account=c40
#SBATCH --job-name=cvfem-honewton
#SBATCH --partition=debug
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=72
#SBATCH --time=00:29:00
#SBATCH --exclusive --hint=nomultithread --uenv-passthrough=ignore
#SBATCH --output=honewton_%j.out
#SBATCH --error=honewton_%j.err

# WHAT THE EXACT HIGHER-ORDER JACOBIAN BUYS THE NEWTON LOOP.
#
# The throughput comparison (jobs/ho_exact_jac.sbatch) says the exact action costs 4.4x per
# matvec on the packed layout. That number is only half a decision: a Jacobian that is the actual
# derivative of the residual restores the Newton rate the lagged one caps, so the question is
# steps, not seconds per step. This measures the steps.
#
# THREE THINGS HAVE TO BE SET OR THE COMPARISON IS VACUOUS, and each was got wrong once:
#   SFEM_MATRIX_FREE=1  -- the Krylov solve defaults to the ASSEMBLED matrix, in which case the
#                          action is never called and both arms return the same count.
#   SFEM_CONV_FREEZE=0  -- with the correction frozen through a continuation stage it is a
#                          constant source and its exact derivative is zero by construction.
#   SFEM_CONV_HO=1      -- otherwise there is no correction to differentiate.
#
# A run that hits the iteration cap is a floor, not a result, so newton_converged is printed
# beside every count and a non-converged row must not be read as a step count.

set -uo pipefail
cd "$SLURM_SUBMIT_DIR" || exit 1
source scripts/alps_env.sh || exit 1

B=$PWD/build/cvfem_hex8_ns_ssgmg
[ -x "$B" ] || { echo "no driver at $B" >&2; exit 1; }
OUT=${OUT:-$SCRATCH/cvfem_honewton/${SLURM_JOB_ID:-manual}}
mkdir -p "$OUT"
N=${N:-16}

# The lid-driven cavity, not the channel. Poiseuille and Couette are close enough to linear that
# the first-order arm converges in one step and the correction has nothing to do; the cavity at
# Re 100 over five continuation stages is where the deferred correction is actually active and
# where the freeze was introduced.
# ONE LINE, deliberately. This string is interpolated into a `bash -lc "..."` command, so a
# newline inside it splits that command in two: the first half became `env <assignments>` with no
# command, which prints the environment and exits 0, and the driver in the second half got a
# malformed line and printed its usage. Every row of the table came back empty.
# Block-Jacobi, and a parameter so the row can say which preconditioner it was measured under.
# This job ran under SFEM_PRECOND=direct once: that is an exact inverse of the operator, so every
# solve took one iteration and 98.6% of the time was the factorisation, and neither the operator's
# cost nor the conditioning it produces was measurable.
PRECOND=${PRECOND:-bjacobi}
COMMON="SFEM_CASE=cavity SFEM_N=$N SFEM_FGMRES=1 SFEM_GMG=0 SFEM_PRECOND=$PRECOND SFEM_ENABLE_OUTPUT=0 SFEM_ELEMENT_REFINE_LEVEL=1 SFEM_CONV_HO=1"

# The three axes are overridable so a missing slice can be re-run without a second copy of this
# job. The full 2x2x2 does not fit the debug partition's 29 minutes -- the first run of this
# reached arm six of eight and was cancelled mid-solve, leaving the limited-and-unfrozen pair
# unmeasured, which is the pair the question is actually about. Run it as LIMS=2 FRS=0.
LIMS=${LIMS:-"0 2"}
FRS=${FRS:-"1 0"}
EXS=${EXS:-"1 0"}

echo "### preconditioner: $PRECOND"
printf '%-8s %-9s %-8s %11s %10s %12s %10s\n' freeze limiter jacobian newton_total converged lin_it_total t_solve_s
for lim in $LIMS; do
for fr in $FRS; do
for ex in $EXS; do
    tag=lim${lim}_fr${fr}_ex${ex}
    log=$OUT/$tag.log
    timeout "${RUN_TIMEOUT:-900}" \
      srun -u --ntasks=1 --cpus-per-task=72 --uenv="$CVFEM_UENV" --view="$CVFEM_VIEW" \
      bash -lc "cd $PWD && OMP_NUM_THREADS=72 OMP_PROC_BIND=true OMP_PLACES=cores \
                env $COMMON SFEM_CONV_LIMITER=$lim SFEM_CONV_FREEZE=$fr SFEM_HO_EXACT_JAC=$ex \
                $B $OUT/o_$tag" > "$log" 2>&1
    nt=$(sed -n 's/.*newton_total: \([0-9]*\).*/\1/p' "$log" | tail -1)
    cv=$(sed -n 's/.*newton_converged: \([0-9]*\).*/\1/p' "$log" | tail -1)
    li=$(sed -n 's/.*lin_it_total: \([0-9]*\).*/\1/p' "$log" | tail -1)
    # The Krylov solve's own seconds, which is the number the steps and the iterations exist to
    # explain: the exact action trades more cost per linear iteration for fewer of them, and only
    # this says which way the trade went. t_precond and t_operator are in the log beside it;
    # under a direct preconditioner t_solve is a rounding error on t_precond, which is why that
    # configuration cannot answer the question at all.
    ts=$(sed -n 's/.*t_solve: \([0-9.]*\) s.*/\1/p' "$log" | tail -1)
    printf '%-8s %-9s %-8s %11s %10s %12s %10s\n' \
           "$([ "$fr" = 1 ] && echo frozen || echo unfrozen)" "$lim" \
           "$([ "$ex" = 1 ] && echo exact || echo lagged)" "${nt:--}" "${cv:--}" "${li:--}" "${ts:--}"
done
done
done
echo
echo "logs in $OUT"
