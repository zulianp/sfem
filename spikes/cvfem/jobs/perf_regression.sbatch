#!/usr/bin/env bash
#SBATCH --account=c40
#SBATCH --job-name=cvfem-perf-gate
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --time=00:25:00
#SBATCH --exclusive
#SBATCH --hint=nomultithread
#SBATCH --uenv-passthrough=ignore
#SBATCH --output=perf_gate_%j.out
#SBATCH --error=perf_gate_%j.err

# The performance-regression gate, on one Grace socket.
#
#   sbatch --export=ALL,REF_BIN=<old-binary> jobs/perf_regression.sbatch   <- the real gate
#   sbatch jobs/perf_regression.sbatch                        coarse baseline check
#   sbatch --export=ALL,RECORD=1 jobs/perf_regression.sbatch  re-record the baseline
#
# Run this whenever a change could touch code generation, compiler flags, memory layout or
# the build. What it gates is the packed matrix-free throughput -- see the header of
# scripts/perf_regression.sh for why that kernel and not another.
#
# Prefer REF_BIN: it measures the new binary against one built from the commit you are
# comparing to, both in this allocation with the order alternating, which is the only form
# that removes the node-to-node variation (5-11% between two Grace nodes) that makes the
# absolute baseline coarse.
#
# --exclusive is not optional here. The gate compares against numbers recorded on an idle
# socket, and a shared node would fail it for reasons that have nothing to do with the code.

set -u

# Locate the spike root -- the directory holding CMakeLists.txt, drivers/ and scripts/.
# BASH_SOURCE is NOT usable on its own: Slurm copies the batch script to
# /var/spool/slurmd/job<id>/slurm_script, so under sbatch the script's own path says
# nothing about where the tree is. SLURM_SUBMIT_DIR, the directory sbatch was invoked
# from, is the reliable one; the rest are for running the script directly.
_cvfem_probe=scripts/perf_regression.sh
SPIKE_ROOT=""
for _cand in "${SLURM_SUBMIT_DIR:-}" "${SLURM_SUBMIT_DIR:-}/.." "${CVFEM_DIR:-}" \
             "$(dirname "${BASH_SOURCE[0]}")/.."; do
    [ -n "${_cand}" ] && [ -f "${_cand}/${_cvfem_probe}" ] || continue
    SPIKE_ROOT="$(cd "${_cand}" && pwd)"; break
done
[ -n "${SPIKE_ROOT}" ] || {
    echo "error: could not locate spikes/cvfem; sbatch from there or set CVFEM_DIR" >&2; exit 1; }
cd "$SPIKE_ROOT"

UENV_IMAGE="${UENV_IMAGE:-prgenv-gnu/24.11:v2}"
UENV_VIEW="${UENV_VIEW:-default}"
NCORES="${NCORES:-${SLURM_CPUS_PER_TASK:-72}}"

# The tree on the compute side is usually an rsync copy with no .git, so `git rev-parse`
# inside the run returns nothing and a --record baseline says "commit: unknown". A baseline
# that cannot say what it measured gates nothing, so pass the commit in explicitly:
#   sbatch --export=ALL,RECORD=1,CVFEM_COMMIT=$(git rev-parse --short HEAD) jobs/perf_regression.sbatch
srun -u --ntasks=1 --cpus-per-task="${NCORES}" --uenv="${UENV_IMAGE}" --view="${UENV_VIEW}" \
    bash -lc "cd '${SPIKE_ROOT}' && THREADS=${NCORES} REPS=${REPS:-9} \
              CVFEM_COMMIT='${CVFEM_COMMIT:-unknown}' \
              stdbuf -oL ./scripts/perf_regression.sh \
                  ${REF_BIN:+--against '${REF_BIN}'} ${RECORD:+--record}"
rc=$?
echo "### gate exit: $rc"
exit $rc
