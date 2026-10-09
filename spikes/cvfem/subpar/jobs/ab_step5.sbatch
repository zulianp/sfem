#!/usr/bin/env bash
#SBATCH --account=c40
#SBATCH --job-name=cvfem-ab5
#SBATCH --partition=debug
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=72
#SBATCH --time=00:29:00
#SBATCH --exclusive --hint=nomultithread --uenv-passthrough=ignore
#SBATCH --output=ab5_%j.out
#SBATCH --error=ab5_%j.err

# Clearing the step-5 move: the generated Jacobian-action arrangements and the affine
# generated residual quarantined into subpar/.
#
# WHY AN A/B IS REQUIRED FOR THIS AND NOT FOR THE EARLIER STEPS. Everything before it moved
# files and renamed symbols, which the compiler resolves identically. This one DELETES
# function definitions from a header that every layout dispatch includes -- 26,196 lines down
# to 16,677 -- and what the compiler sees is what decides inlining and I-cache layout. A
# kernel that did not change a line of its own source can still get slower because something
# beside it went away. That is the whole reason the plan gates this step and only this step.
#
# --against, not the recorded baseline: node-to-node variation is 5-11% here and position
# within an allocation is worth 12%, both far larger than a regression worth finding. The
# two binaries are measured in one allocation with the order alternating every repetition.
#
# THE REFERENCE IS A SUPERSET, NOT THE EXACT "BEFORE", and the result has to be read that
# way. The exact pre-move binary was destroyed by a build script that overwrote its own
# reference, and it cannot be rebuilt from this tree: the generated header, the stubs standing
# in for the retired functions and the CLI rejections all changed together, so reverting any
# one of them alone does not compile.
#
# build_refsubpar is -DCVFEM_ENABLE_SUBPAR=ON, which puts every retired arrangement back into
# the translation unit -- the seven from this move AND the four assembly arrangements retired
# before this session. 5,344,128 bytes against the new binary's 4,818,592.
#
# That makes the test conservative in the right direction: the surviving kernels are measured
# against a build carrying MORE code beside them than the true reference did, so a null result
# is stronger evidence than a null against the exact reference would have been. What it cannot
# do is say which removal moved a number, if one did.

set -uo pipefail
cd "$SLURM_SUBMIT_DIR" || exit 1
source scripts/alps_env.sh || exit 1

REF=$PWD/build_refsubpar/cvfem_hex8_ns_upwind_bench
NEW=$PWD/build/cvfem_hex8_ns_upwind_bench
[ -x "$REF" ] || { echo "no reference binary at $REF" >&2; exit 1; }
[ -x "$NEW" ] || { echo "no new binary at $NEW" >&2; exit 1; }

echo "### host $(hostname)  $(date)"
echo "### ref $(ls -la "$REF" | awk '{print $5, $6, $7, $8}')"
echo "### new $(ls -la "$NEW" | awk '{print $5, $6, $7, $8}')"

srun -u --ntasks=1 --cpus-per-task=72 --uenv="$CVFEM_UENV" --view="$CVFEM_VIEW" \
  bash -lc "cd $PWD && BIN=$NEW THREADS=72 REPS=9 ./scripts/perf_regression.sh --against $REF"
echo "### perf_regression exit: $?"
