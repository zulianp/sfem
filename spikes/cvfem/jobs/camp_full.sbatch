#!/usr/bin/env bash
#SBATCH --account=c40
#SBATCH --job-name=cvfem-camp-full
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --time=03:00:00
#SBATCH --hint=nomultithread
#SBATCH --uenv-passthrough=ignore
#SBATCH --output=cvfem-camp-full_%j.out
#SBATCH --error=cvfem-camp-full_%j.err

# The standard-vs-packed layout campaign, on the hardware every performance decision in this
# spike is made on. The laptop can run the same script and its output is plumbing, not a
# result: a number from an Apple M1 is not comparable with one from 72 Grace cores, and
# comparing numbers with each other is the entire purpose of this sweep.
#
# ONE srun step for the whole sweep, not one per run. The campaign script execs the binary
# directly, many times in sequence; putting each exec in its own srun would nest steps and
# is what made an earlier job hang until its wall clock ran out. One step also means one set
# of CPU bindings for every configuration, which is what makes the readings comparable.
#
# The environment comes from scripts/alps_env.sh, which is the tree's own recipe -- `uenv
# run` with the image the installed SFEM was built against. A batch shell that skips it gets
# a binary that cannot find libmpi and every arm fails identically for one reason that has
# nothing to do with what was being measured. That has happened.

set -uo pipefail
cd "$SLURM_SUBMIT_DIR" || exit 1
source scripts/alps_env.sh || exit 1

echo "### host $(hostname)  $(date '+%F %T')"
echo "### sizes 64 96 128 160 192  reps 5"

# SIZES and OUT come from the environment so the sweep can be SPLIT across two debug jobs and
# still land in one csv. The whole sweep takes 33.5 min, just past the debug partition's 30, and
# n=192 alone is half of that -- the cost goes as the element count, so 64..160 is 17 min and 192
# is 16. Two debug jobs sharing an OUT beat one normal job that has queued for a day, and the
# bench appends to an existing csv with a matching header, so the halves accumulate rather than
# overwrite. Run them in sequence, not concurrently, or the appends interleave.
CAMP_SIZES=${CAMP_SIZES:-"64 96 128 160 192"}
CAMP_OUT=${CAMP_OUT:-perf/campaign_grace_${SLURM_JOB_ID:-manual}}
echo "### sizes  : $CAMP_SIZES"
echo "### out    : $CAMP_OUT"

srun -u --ntasks=1 --cpus-per-task=72 --uenv="$CVFEM_UENV" --view="$CVFEM_VIEW" \
  bash -lc "cd $PWD && BIN=build/cvfem_hex8_ns_upwind_bench THREADS=72 \
            SIZES='$CAMP_SIZES' REPS=5 RUN_TIMEOUT=900 OUT=$CAMP_OUT \
            ./scripts/layout_campaign.sh"

echo
echo "### report:"
OUT=$CAMP_OUT
python3 python/cvfem_kernel_report.py $OUT/campaign.csv -o $OUT/CVFEM_Kernels.md 2>&1 | tail -2
