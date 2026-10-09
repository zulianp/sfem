#!/usr/bin/env bash
#SBATCH --account=c40
#SBATCH --job-name=cvfem-ctest
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --time=00:28:00
#SBATCH --hint=nomultithread
#SBATCH --uenv-passthrough=ignore
# THE VIEW IS SET ON THE ALLOCATION, not around the ctest command, and both halves of that matter.
# Wrapping ctest in `srun --uenv=...` deadlocks, because cvfem_mpi_guard launches its own step and
# a step inside a step hangs -- measured, 28 minutes and not one per-test line. Setting it only in
# the batch shell with `uenv run` fixes the deadlock and breaks those same tests the other way:
# their inner srun starts fresh tasks that never see the view, and they died on
# "libmpi_gnu_123.so.12: cannot open shared object file". On the allocation, every step inherits it
# and neither happens.
#SBATCH --uenv=prgenv-gnu/24.11:v2
#SBATCH --view=default
#SBATCH --output=ctest_%j.out
#SBATCH --error=ctest_%j.err

# THE STANDING LOCAL GATE, WHICH THIS TREE HAS NEVER BEEN RUN AGAINST.
#
# The CVFEM spike is gated by ctest locally and by Alps batch scripts. Roughly 300 lines of
# changed code sit in this tree -- the coarse-packing fix in derefine_op plus the earlier
# decomposition fixes (GlobalDenseLU's scatter-then-sum gather, the level parallel face, the
# collective colour count, the lattice-row exchange, the stencil and coarse-BSR completions) --
# and ctest has not been run against any of it in this session.
#
# The verification matrix's `unit` group is what normally runs ctest, and the subset I ran
# (job 4720471) covered only `port` and `budget`, the two groups that derefine. The full matrix
# (job 4720012) is queued behind 1,130 jobs and is estimated to start at 01:36, which is too
# long to leave the cheapest gate untouched before a commit is considered.
#
# It runs here rather than on the laptop because it takes past nine minutes, and on the same
# Alps build the matrix will use, so a pass here means the matrix's unit group should pass too.
#
# READ, committed before the run:
#
#   all tests pass              -> the unit gate is clear; the remaining commit evidence is the
#                                  full matrix's byte-for-byte report diff
#   a test fails                -> that outranks every distributed and throughput result in this
#                                  branch, and the commit is blocked until it is understood
#   ctest cannot find the suite -> the gate did not run; report that, do NOT read a clean exit
#                                  as a pass

set -u

_cvfem_probe=scripts/alps_env.sh
SPIKE_ROOT=""
for _cand in "${SLURM_SUBMIT_DIR:-}" "${SLURM_SUBMIT_DIR:-}/.." "${CVFEM_DIR:-}" \
             "$(dirname "${BASH_SOURCE[0]}")/.."; do
    [ -n "${_cand}" ] && [ -f "${_cand}/${_cvfem_probe}" ] || continue
    SPIKE_ROOT="$(cd "${_cand}" && pwd)"; break
done
[ -n "${SPIKE_ROOT}" ] || { echo "error: could not locate spikes/cvfem" >&2; exit 1; }
cd "$SPIKE_ROOT"

UENV_IMAGE="prgenv-gnu/24.11:v2"; UENV_VIEW="default"
BUILD="${BUILD:-$SPIKE_ROOT/build}"
OUT="${OUT:-$SCRATCH/cvfem_ctest/${SLURM_JOB_ID:-manual}}"; mkdir -p "$OUT"

echo "build dir : $BUILD"
[ -d "$BUILD" ] || { echo "error: no build directory at $BUILD" >&2; exit 1; }
echo "tests registered:"
( cd "$BUILD" && ctest -N 2>&1 | tail -5 )
echo

# NOT WRAPPED IN srun, and with -j. Both are corrections to a job that timed out at 28 minutes
# having produced not one per-test line -- it was not running slowly, it was deadlocked. The
# batch step of a one-node job already runs on the compute node, so the outer srun bought
# nothing; what it cost is that cvfem_mpi_guard launches its own step, and a step inside a step
# is the nesting this tree has already had hang until its wall clock ran out (see
# scripts/layout_campaign.sh's note). Serially the suite also does not fit the debug window: it
# is roughly seven minutes at -j 4 on a laptop, and 35 tests one at a time here is more than
# twenty-eight. -j 8 with eight threads each fills the socket.
( cd "$BUILD" && OMP_NUM_THREADS=8 OMP_PROC_BIND=true ctest -j 8 --output-on-failure ) \
    > "$OUT/ctest.log" 2>&1
rc=$?
echo "ctest exit: $rc"
echo

SUMMARY=$(grep -aE "tests passed|tests failed" "$OUT/ctest.log" | tail -2)
if [ -z "$SUMMARY" ]; then
    echo "NO SUMMARY LINE -- the suite did not report; this is NOT a pass."
    echo "--- tail of the log:"
    tail -20 "$OUT/ctest.log"
else
    echo "=== summary"
    echo "$SUMMARY"
fi
echo
echo "=== any failing test names"
grep -aE "\(Failed\)|\(Timeout\)|\*\*\*Failed" "$OUT/ctest.log" | head -20
echo
echo "log: $OUT/ctest.log"
