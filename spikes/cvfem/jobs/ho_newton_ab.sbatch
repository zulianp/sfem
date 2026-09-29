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
COMMON="SFEM_CASE=cavity SFEM_N=$N SFEM_FGMRES=1 SFEM_GMG=0 SFEM_PRECOND=direct SFEM_ENABLE_OUTPUT=0 SFEM_ELEMENT_REFINE_LEVEL=1 SFEM_CONV_HO=1"

printf '%-8s %-9s %-8s %11s %10s %12s\n' freeze limiter jacobian newton_total converged lin_it_total
for lim in 0 2; do
for fr in 1 0; do
for ex in 1 0; do
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
    printf '%-8s %-9s %-8s %11s %10s %12s\n' \
           "$([ "$fr" = 1 ] && echo frozen || echo unfrozen)" "$lim" \
           "$([ "$ex" = 1 ] && echo exact || echo lagged)" "${nt:--}" "${cv:--}" "${li:--}"
done
done
done
echo
echo "logs in $OUT"
