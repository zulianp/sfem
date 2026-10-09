#!/usr/bin/env bash
#SBATCH --account=c40
#SBATCH --job-name=cvfem-pfreeze
#SBATCH --partition=debug
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=72
#SBATCH --time=00:29:00
#SBATCH --exclusive --hint=nomultithread --uenv-passthrough=ignore
#SBATCH --output=pfreeze_%j.out
#SBATCH --error=pfreeze_%j.err

# WHAT REUSING THE PRECONDITIONER'S SETUP BUYS, AND WHAT IT COSTS IN ITERATIONS.
#
# Where the time goes, measured on this case at 1,098,500 dof (job 4768083): of 130.919 s in
# top-level phases, coarse_factor 50.405 s (38.5%, 14 calls) with 46.750 s of that inside
# getrf, coarse_solve 22.860 s (17.5%, 148 calls), smooth[L0] 5.928 s (4.5%), vanka_setup
# 2.466 s (1.9%, 14 calls). A perf profile of the same run puts dgemm_kernel_NEOVERSEV1 at
# 54.5% of samples -- getrf's blocked update.
#
# So there are two setups rebuilt once per Newton step, and one of them is twenty times the
# other. Both freeze on the same policy (precond_freeze_rebuild), with separate strides.
#
# THE RULE THIS JOB EXISTS TO OBEY, stated in the driver next to the smoother's knob: any
# report of a freeze stride quotes Newton and linear iterations beside wall time, or it is not
# a result. Skipped setups are paid for in coarse solves and smoother sweeps, and at stride
# "never" the toy case went from 47 linear iterations to 281 -- the trade is real and can lose.
#
# EVERY invocation runs its own baseline arm. The same binary measured this case at 510.3,
# 484.2 and 225.5 MDOF/s on three separate allocations, so a number from another job is not a
# control. Only within-job differences are quoted.
#
# ARMS overrides the list; each entry is "label|VANKA_FREEZE|COARSE_FREEZE".

set -uo pipefail
cd "$SLURM_SUBMIT_DIR" || exit 1
source scripts/alps_env.sh || exit 1

D=$PWD/build_solver/cvfem_hex8_ns_ssgmg
OUT=${OUT:-$SCRATCH/cvfem_pfreeze/${SLURM_JOB_ID:-manual}}
mkdir -p "$OUT"

# One line. A literal newline inside the bash -lc string splits it into two commands and the
# driver then runs with no SFEM_CASE, prints its usage, and the table renders all dashes.
#
# BASE is overridable so a second case confirms the same knob without a second copy of this
# script. Three near-identical copies of a job is a mistake this directory has already made
# once: they drift, and the copy that was taken before the parent became parameterised then
# silently measures the wrong arms.
BASE=${BASE:-"SFEM_CASE=cavity SFEM_FGMRES=1 SFEM_GMG=1 SFEM_ENABLE_OUTPUT=0 SFEM_N=16 SFEM_ELEMENT_REFINE_LEVEL=4 SFEM_MU=0.01 SFEM_NL_MAX_IT=8"}
echo "### BASE: $BASE"

ARMS=${ARMS:-"base|1|1 vf2|2|1 vf4|4|1 vf8|8|1"}

printf '%-8s %5s %5s %8s %6s %7s %9s %6s %9s %6s %9s %8s\n' \
       arm VF CF ndof newton linear cfactor n_cf csolve n_vs vsetup wall_s
for a in $ARMS; do
    IFS='|' read -r lbl vf cf <<<"$a"
    log=$OUT/$lbl.log
    # Per-arm cap against the serial reference: the baseline run of this case is ~86 s, so an
    # arm that has not finished in 330 s is not slow, it is stuck, and without a cap one such
    # arm eats the whole wall clock.
    t0=$(date +%s)
    timeout "${ARM_CAP:-330}" \
      srun -u --ntasks=1 --cpus-per-task=72 --uenv="$CVFEM_UENV" --view="$CVFEM_VIEW" \
      bash -lc "cd $PWD && OMP_NUM_THREADS=72 OMP_PROC_BIND=true OMP_PLACES=cores env $BASE SFEM_VANKA_FREEZE=$vf SFEM_COARSE_FREEZE=$cf $D $OUT/out_$lbl" \
      > "$log" 2>&1
    wall=$(( $(date +%s) - t0 ))

    ndof=$(sed -n 's/.*ndof: \([0-9]*\).*/\1/p' "$log" | head -1)
    conv=$(sed -n 's/.*newton_converged: \([0-9]*\).*/\1/p' "$log" | head -1)
    nwt=$(sed -n 's/.*newton_total: \([0-9]*\).*/\1/p' "$log" | head -1)
    lin=$(sed -n 's/.*lin_it_total: \([0-9]*\).*/\1/p' "$log" | head -1)
    read -r n_cf t_cf <<<"$(awk '/^ +coarse_factor /{print $3, $2}' "$log" | head -1)"
    read -r n_vs t_vs <<<"$(awk '/^ +vanka_setup /{print $3, $2}' "$log" | head -1)"
    t_cs=$(awk '/^ +coarse_solve /{print $2}' "$log" | head -1)
    # An unconverged run's counts are a floor, not a result, so the row says so rather than
    # inviting a comparison against a converged one.
    [ "${conv:-0}" = "1" ] || lbl="$lbl*"
    printf '%-8s %5s %5s %8s %6s %7s %9s %6s %9s %6s %9s %8s\n' \
           "$lbl" "$vf" "$cf" "${ndof:--}" "${nwt:--}" "${lin:--}" \
           "${t_cf:--}" "${n_cf:--}" "${t_cs:--}" "${n_vs:--}" "${t_vs:--}" "$wall"
done
echo
echo "* = newton_converged 0; its counts are a floor, not a comparable result"
echo "logs in $OUT"
