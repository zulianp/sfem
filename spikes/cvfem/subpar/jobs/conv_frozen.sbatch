#!/usr/bin/env bash
#SBATCH --account=c40
#SBATCH --job-name=cvfem-convfz
#SBATCH --partition=debug
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=72
#SBATCH --time=00:29:00
#SBATCH --exclusive --hint=nomultithread --uenv-passthrough=ignore
#SBATCH --output=convfz_%j.out
#SBATCH --error=convfz_%j.err

# The order of accuracy of each convective arm, which is the half of the limiter question
# jobs/conv_limiter.sbatch cannot answer.
#
# THAT JOB MEASURED CONVERGENCE AND THE ANSWER WAS EMPHATIC: on the backward-facing step at
# Re 40, Venkatakrishnan's eps^2 took the fine cell from 303 Newton steps at K = 0 to 41 at
# K = 5, and to non-convergence at K = 30 -- where it has become the unlimited arm, which
# also fails there. So eps^2 is a continuous dial from "limiter fully on" to "limiter fully
# off" and both ends behave like their endpoint.
#
# WHICH IS EXACTLY WHY CONVERGENCE CANNOT DECIDE K. An arm that has switched itself off
# converges beautifully and bounds nothing, and K = 30 proves that state is reachable on this
# same dial. The question this job answers is whether K = 5 is ACCURATE or merely OFF, and
# the discriminator is the order of accuracy against a manufactured solution at a Reynolds
# number where the donor split is active: an arm that recovers the whole gap to the unlimited
# reconstruction has almost certainly stopped limiting.
#
# Per-case cap: the arms here can fail to converge, and VERIFY_RUN_TIMEOUT stops one of them
# taking the allocation and making every later case look like a crash.

set -uo pipefail
cd "$SLURM_SUBMIT_DIR" || exit 1
source scripts/alps_env.sh || exit 1

OUT=${OUT:-$SCRATCH/cvfem_conv/${SLURM_JOB_ID:-manual}}
mkdir -p "$OUT"

echo "### host $(hostname)  $(date)"

srun -u --ntasks=1 --cpus-per-task=72 --uenv="$CVFEM_UENV" --view="$CVFEM_VIEW" \
  bash -lc "cd $PWD && OMP_NUM_THREADS=72 OMP_PROC_BIND=true OMP_PLACES=cores \
            VERIFY_GROUPS=conv VERIFY_RUN_TIMEOUT=240 CONV_ARMS='lo ho_lim0 ho_lim2_k0 ho_lim0_frozen ho_lim2_k0_frozen' CONV_LADDER='8 16 32' \
            DRIVER=$PWD/build_solver/cvfem_hex8_ns_ssgmg OUT=$OUT \
            ./scripts/verify_report.sh"

echo
echo "### report"
python3 python/cvfem_verify_report.py "$OUT" -o "$OUT/CVFEM_Verification_Report.md" 2>&1 | tail -3
echo "### arm table:"
# Not an awk range: the heading matches the range's own end pattern, so /A/,/^## / selects
# exactly the one line and the table never prints. A flag is the fix.
awk '/^## The convective scheme/{f=1;next} f&&/^## /{exit} f' \
    "$OUT/CVFEM_Verification_Report.md" 2>/dev/null | grep '^|' | head -14
