#!/usr/bin/env bash
#SBATCH --account=c40
#SBATCH --job-name=cvfem-convcost
#SBATCH --partition=debug
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=72
#SBATCH --time=00:29:00
#SBATCH --exclusive --hint=nomultithread --uenv-passthrough=ignore
#SBATCH --output=convcost_%j.out
#SBATCH --error=convcost_%j.err

# What the convective scheme COSTS, per arm.
#
# The kernel benchmark cannot answer this: cvfem_hex8_ns_upwind_bench has no
# deferred-correction flag and its residual call passes no limiter, so every number in
# docs/CVFEM_Kernels.md is the first-order scheme. The solver's own phase breakdown does
# answer it -- newton_residual reports seconds and calls, so us/call is a throughput.
#
# The layout column is the point. SFEM_CONV_HO=1 forces the atomic sum-factored path, because
# the packed and SIMD residual kernels take no ugrad8 or limiter argument. So there IS no
# packed number for any higher-order arm, and the first-order row is there to show what is
# given up: the campaign measured packed at 2624 MDOF/s against atomic's 858 at 8,586,756 dof.

set -uo pipefail
cd "$SLURM_SUBMIT_DIR" || exit 1
source scripts/alps_env.sh || exit 1

D=$PWD/build_solver/cvfem_hex8_ns_ssgmg
OUT=${OUT:-$SCRATCH/cvfem_convcost/${SLURM_JOB_ID:-manual}}
mkdir -p "$OUT"

# ONE LINE, deliberately. This was wrapped across two, and the literal newline inside the
# double-quoted bash -lc string split it into two commands: an `env` with nothing to run,
# which printed the environment, and then the driver without SFEM_CASE, which printed its
# usage and exited 1. The job completed, the table rendered, and every cell was a dash.
BASE="SFEM_CASE=cavity SFEM_FGMRES=1 SFEM_GMG=1 SFEM_ENABLE_OUTPUT=0 SFEM_N=16 SFEM_ELEMENT_REFINE_LEVEL=4 SFEM_MU=0.01 SFEM_NL_MAX_IT=8"

printf '%-30s %10s %8s %10s %12s\n' arm ndof calls us_call MDOF_s
for a in "lo|SFEM_CONV_HO=0|" \
         "ho_unlimited|SFEM_CONV_HO=1|SFEM_CONV_LIMITER=0" \
         "ho_clip|SFEM_CONV_HO=1|SFEM_CONV_LIMITER=1" \
         "ho_venkat_k0|SFEM_CONV_HO=1|SFEM_CONV_LIMITER=2" \
         "ho_dm|SFEM_CONV_HO=1|SFEM_CONV_LIMITER=3"; do
    IFS='|' read -r lbl e1 e2 <<<"$a"
    log=$OUT/$lbl.log
    # A cap per arm. The previous run had none, sized itself at 3.65M dof, and spent its whole
    # 29-minute wall on the FIRST arm -- so it timed out having measured one row of five. The
    # rule about capping each arm against a reference is one this job needed and did not have.
    timeout "${ARM_CAP:-360}" \
      srun -u --ntasks=1 --cpus-per-task=72 --uenv="$CVFEM_UENV" --view="$CVFEM_VIEW" \
      bash -lc "cd $PWD && OMP_NUM_THREADS=72 OMP_PROC_BIND=true OMP_PLACES=cores \
                env $BASE $e1 ${e2:-SFEM_DUMMY=0} SFEM_CONV_FREEZE=0 $D $OUT/out_$lbl" \
      > "$log" 2>&1
    ndof=$(sed -n 's/.*ndof: \([0-9]*\).*/\1/p' "$log" | head -1)
    read -r calls us <<<"$(awk '/newton_residual/{print $3, $4}' "$log" | head -1)"
    if [ -n "${ndof:-}" ] && [ -n "${us:-}" ]; then
        mdofs=$(awk -v n="$ndof" -v u="$us" 'BEGIN{printf "%.1f", n/u}')
    else
        mdofs="-"; calls=${calls:--}; us=${us:--}
    fi
    printf '%-30s %10s %8s %10s %12s\n' "$lbl" "${ndof:--}" "${calls:--}" "${us:--}" "$mdofs"
done
echo
echo "logs in $OUT"
