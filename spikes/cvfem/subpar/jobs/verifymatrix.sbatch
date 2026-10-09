#!/usr/bin/env bash
#SBATCH --account=c40
#SBATCH --job-name=cvfem-verifymatrix
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --time=02:00:00
#SBATCH --hint=nomultithread
#SBATCH --uenv-passthrough=ignore
#SBATCH --output=verifymatrix_%j.out
#SBATCH --error=verifymatrix_%j.err

# IS SERIAL BIT-IDENTICAL AFTER THE FOUR MPI FIXES?
#
# The enforcement rule for this workstream is explicit: after a change, DIFF the verification
# matrix output against the pre-change one -- not "compare conclusions", diff. Everything I
# have claimed about serial so far is spot values (158, 61, 43, 3, and the stencil checksums),
# which is not that.
#
# The claim under test is structural rather than empirical: each of the four fixes is behind a
# size() > 1 guard, so at one rank none of them executes.
#
#   (1) GlobalDenseLU's scatter-then-reduce gather   -- the whole function early-returns at
#                                                       one rank into make_dense_lu_from_bsr
#   (2) the fine Vanka stencil completion            -- guarded on mesh->comm()->size() > 1
#   (3) the level-operator / smoother parallel face  -- create_parallel_preconditioner is only
#                                                       applied when dist is true
#   (4) the coarse BSR completion                    -- same guard, per level
#
# If that holds, regenerating docs/CVFEM_Verification_Report.md and .html from this run's
# evidence leaves `git diff` on them EMPTY. A non-empty diff means a fix runs when it should
# not, and that is a defect regardless of how good the distributed numbers look.
#
# WHY THE NORMAL PARTITION. The matrix is serial but not short: the `unit` group alone shells
# out to ctest and ran past 540 s on the laptop before finishing two cases, and there are eight
# groups including an MMS ladder to 143,748 dof and an eight-point port sweep. That does not
# fit a 30-minute debug arm. Per the QOS rule, a normal-partition job does not consume the
# debug submit quota, so this can run alongside debug work.
#
# The script only produces evidence; the judgements live in the report generator, so the diff
# is taken afterwards:
#
#   python3 python/cvfem_verify_report.py <OUT> -o docs/CVFEM_Verification_Report.md --html
#   git diff --stat -- docs/CVFEM_Verification_Report.md docs/CVFEM_Verification_Report.html
#
# READ, committed before the run:
#
#   regenerated reports identical to the tracked ones -> serial is bit-identical and the four
#                                                        fixes are confined to the distributed
#                                                        path, as designed
#   any numeric field moved                           -> a fix executes at one rank; find which
#                                                        guard leaks before anything is committed
#   a case that fails here but passed before          -> a serial regression, and that outranks
#                                                        every distributed result in this branch

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

UENV_IMAGE="prgenv-gnu/24.11:v2"
UENV_VIEW="default"
BIN="${CVFEM_BIN:-./build/cvfem_hex8_ns_ssgmg}"
OUT="${OUT:-$SCRATCH/cvfem_verifymatrix/${SLURM_JOB_ID:-manual}}"
mkdir -p "$OUT"
[ -x "$BIN" ] || { echo "error: no binary at $BIN" >&2; exit 1; }

echo "spike root : $SPIKE_ROOT"
echo "driver     : $BIN"
echo "evidence   : $OUT"
echo

# One rank, the full team of threads: this is the SERIAL matrix, and the point is that none of
# the four MPI fixes executes. OMP_PROC_BIND per the standing benchmark convention.
srun -u --ntasks=1 --cpus-per-task=72 \
     --uenv="$UENV_IMAGE" --view="$UENV_VIEW" \
     bash -lc "OMP_NUM_THREADS=72 OMP_PROC_BIND=true DRIVER='$SPIKE_ROOT/$BIN' OUT='$OUT' \
               ./scripts/verify_report.sh"
rc=$?
echo
echo "verify_report exit: $rc"
echo "evidence directory: $OUT"
echo
echo "next, to take the diff that is the actual test:"
echo "  python3 python/cvfem_verify_report.py $OUT -o docs/CVFEM_Verification_Report.md --html"
echo "  git diff --stat -- docs/CVFEM_Verification_Report.md docs/CVFEM_Verification_Report.html"
