#!/usr/bin/env bash
#SBATCH --account=c40
#SBATCH --job-name=cvfem-phasea
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --time=00:28:00
#SBATCH --hint=nomultithread
#SBATCH --uenv-passthrough=ignore
#SBATCH --output=phasea_%j.out
#SBATCH --error=phasea_%j.err

# PHASE A ON A CONSISTENT TOOLCHAIN.
#
# Two questions, and they are different.
#
# (1) Does the blk_agree fix hold here? Locally it took the three failing rows from
#     1.70e-08 / 1.11e-08 / 3.50e-09 to ~3e-16 and the gate went green. The agreement columns are
#     machine independent, so this is a confirmation rather than a new measurement -- but it is
#     worth having on the machine the work will actually be verified on.
#
# (2) Are the 8 local ctest failures environmental? Locally 8 of 37 failed, all segfaults in the
#     ssgmg driver: cvfem_restart, cvfem_cutback, cvfem_mg_transient, cvfem_richardson,
#     cvfem_mpi_guard, cvfem_distributed_setup (+2). None mentions merits or reduction, so the
#     new energy_or_potential_based definition is not obviously implicated.
#
#     The laptop's installed SFEM is AHEAD of the spike: it gained a pure virtual in af2a9b26f
#     (2026-09-19) that CVFEM never implemented, so the spike did not compile locally at all until
#     that method was added. Nobody has been able to run these tests locally since that install was
#     refreshed. Alps' installed SFEM predates af2a9b26f, so here the spike and its library are the
#     same vintage. If these tests pass here, the local failures are the skew and not the change.
#
# Only the failing subset is run, not all 37 -- the full suite exceeded 28 minutes on this
# partition in an earlier attempt, and the question is about these tests specifically.
#
# READ, committed before the run:
#   bench green + subset passes  -> Phase A is sound; the local failures are the install skew
#   bench green + subset fails   -> the failures are real and belong to this change or to the tree;
#                                   Phase A is NOT done and the cause must be found before Phase B
#   bench red                    -> the fix does not hold here and the local green was misleading

set -u
cd /ritom/scratch/cscs/zulianp/sfem-coarseop/spikes/cvfem

UENV="prgenv-gnu/24.11:v2"
echo "########## (1) the gate: full default sweep ##########"
srun -u --ntasks=1 --cpus-per-task=72 --uenv="$UENV" --view=default \
  bash -lc "OMP_NUM_THREADS=72 OMP_PROC_BIND=true ./build/cvfem_sshex8_bench" 2>&1 \
  | grep -aE "^L |^[0-9] |disagree|FAILED"
echo
echo "########## (2) the 8 tests that failed locally ##########"
srun -u --ntasks=1 --cpus-per-task=72 --uenv="$UENV" --view=default \
  bash -lc "cd build && OMP_NUM_THREADS=8 ctest --output-on-failure \
            -R 'cvfem_restart|cvfem_cutback|cvfem_mg_transient|cvfem_richardson|cvfem_mpi_guard|cvfem_distributed_setup'" \
  > /tmp/subset_$SLURM_JOB_ID.log 2>&1
echo "ctest rc=$?"
grep -aE "tests passed|tests failed" /tmp/subset_$SLURM_JOB_ID.log | tail -2
echo "--- failures here:"
grep -aE "\(Failed\)|\(Timeout\)" /tmp/subset_$SLURM_JOB_ID.log | head -10
echo "--- if no summary line printed, the suite did not report and this is NOT a pass"
