#!/usr/bin/env bash
#SBATCH --account=c40
#SBATCH --job-name=cvfem-rank-bisect
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=288
#SBATCH --time=00:30:00
#SBATCH --exclusive
#SBATCH --hint=nomultithread
#SBATCH --uenv-passthrough=ignore
#SBATCH --output=rank_bisect_%j.out
#SBATCH --error=rank_bisect_%j.err

# AT WHAT RANK COUNT DOES THE DISTRIBUTED SETUP CORRUPT THE HEAP?
#
# Job 4715015 ran cavity N=8 (4913 nodes, 512 macro elements, 19652 dof) at 288 ranks and
# every rank died before printing anything at all -- zero "nnodes:" lines, zero fgmres
# diagnostics, zero Newton lines. The first output was the allocator:
#
#     munmap_chunk(): invalid pointer
#     double free or corruption (!prev)
#     corrupted size vs. prev_size
#     srun: error: tasks 4-7,9-37,... : Segmentation fault (core dumped)
#
# So this is heap corruption during distributed SETUP, before any numerics. It is a
# different failure from the earlier 288-rank attempt, which was the deliberate guard
# "hex8_cube_create_distributed: cube too small (n_global_elements=108 ... comm_size=288)".
# Cavity N=8 clears that guard (512 elements and 4913 nodes both exceed 288), so the mesh is
# created and something after it corrupts memory.
#
# The same allocator signature appeared earlier in this campaign from coarse ids sourced
# from the wrong block -- arrays sized from one count and indexed by ids from another. At
# 288 ranks a rank owns about 1.8 elements and 17 nodes, so any site that assumes a minimum
# local size, or indexes a per-rank array by an id from a different numbering, will corrupt
# the heap here and nowhere else.
#
# This job finds the smallest rank count that reproduces it. Every arm is the same problem;
# only the decomposition changes. Arms are ordered ascending so the log shows the last good
# count immediately before the first bad one.
#
# READ: the first arm with no "nnodes:" line is where setup began failing. An arm that
# prints nnodes and then dies later is a DIFFERENT defect and must not be merged with this
# one. An arm that completes gives a usable iteration count for the invariance comparison
# against the serial reference, which is lin_it_total 173 (cavity N=8, mg + jacobi, 14
# Newton steps over 5 stages).
#
# SINGLE-THREADED: pure MPI, which is the configuration the 288-rank request is about, and
# which removes atomics and thread scheduling as explanations.
#
# SFEM_PARALLEL_MF_LEGACY=1 because the phased path is not available: an Op that has not
# overridden the scoped apply aborts at frontend/ops/sfem_Op.hpp:238 with "does not support
# ElementScope other than ALL", and the CVFEM operator has no ElementScope implementation at
# all. Job 4715015's phased arms died there on every multi-rank arm, which is the plan's
# S9/S13 work, not a measurement.

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
OUT="${OUT:-$SCRATCH/cvfem_rank_bisect/${SLURM_JOB_ID:-manual}}"
RAW="$OUT/raw"
mkdir -p "$RAW"
[ -x "$BIN" ] || { echo "error: no binary at $BIN" >&2; exit 1; }

# Per-arm wall budget, in seconds.
#
# A hung arm must not be allowed to consume the allocation. These sweeps run the same
# problem at every rank count inside one job, so an arm that enters a collective its peers
# never reach will sit there until Slurm kills the whole job, taking every arm still queued
# behind it with it -- far more expensive than the single arm that hung. One measured
# example from this very sweep: serial finished in 18 s, two ranks in 92 s, four ranks in
# 214 s, and the eight-rank arm then produced no output at all and was still holding the
# node minutes later.
#
# The budget is sized against the largest LEGITIMATE arm rather than against the serial
# time, because multi-rank arms genuinely do more Krylov work here; a cap of "serial times
# a small factor" would clip real runs. ARM_TIMEOUT bounds a hang, it does not trim honest
# work.
#
# Both layers are deliberate: srun --time lets Slurm reclaim the step, and the timeout
# wrapper catches the case where srun itself wedges rather than the application. An arm that
# hits either is reported as TIMEOUT and the sweep moves to the next rank count, because a
# sweep that reports "8 hung, 16 through 288 clean" localises the fault, while one that dies
# at its third arm says nothing.
ARM_TIMEOUT="${ARM_TIMEOUT:-120}"

run_arm() {
    local tag="$1" nranks="$2"; shift 2
    echo "### ${tag} ranks=${nranks} ###"
    timeout --signal=TERM --kill-after=30 "${ARM_TIMEOUT}" \
        srun -u --ntasks="${nranks}" --cpus-per-task=1 --kill-on-bad-exit=1 \
            --time="00:$(printf '%02d' $(( (ARM_TIMEOUT + 59) / 60 ))):00" \
            --uenv="$UENV_IMAGE" --view="$UENV_VIEW" \
            bash -lc "$* stdbuf -oL ${BIN} ${OUT}/${tag}_n${nranks}" 2>&1 \
        | tee "$RAW/${tag}_n${nranks}.log" \
        | stdbuf -oL grep -aE "nnodes:|global:|^fgmres: distributed|newton_converged|lin_it_total|munmap|double free|corrupt|Segmentation|cube too small"
    local rc=${PIPESTATUS[0]}
    if [ "$rc" = "124" ] || [ "$rc" = "137" ]; then
        echo "  (arm exit: ${rc} -- TIMEOUT after ${ARM_TIMEOUT}s, arm did not finish)"
    else
        echo "  (arm exit: ${rc})"
    fi
    echo
}

# The question this sweep asks is answered during distributed SETUP, not by the solve:
# every corrupt arm died with zero "nnodes:" lines, before any numerics ran. So the arms are
# cut short deliberately -- one Newton step, a small linear cap, and a verification
# tolerance that cannot fail the run -- which takes each arm from minutes to seconds and
# lets all eleven rank counts land inside one debug allocation.
#
# Cutting them short is not a shortcut around the measurement, it IS the measurement: an arm
# that prints its mesh line and its fgmres diagnostic has completed distributed setup, which
# is exactly what the heap corruption used to prevent. The iteration-invariance comparison
# is a different question and belongs in its own job, on the rank counts cheap enough to run
# it -- pre-fix those totals were 173, 8133, 28116, 44058 and 12000 linear iterations at
# 1, 2, 4, 8 and 16 ranks, which is why the full solve cannot be swept here.
BASE="OMP_NUM_THREADS=1 SFEM_FGMRES=1 SFEM_FGMRES_DIAG=1 SFEM_ENABLE_OUTPUT=0 \\
      SFEM_PARALLEL_MF_LEGACY=1 SFEM_ALLOW_MPI=1 \\
      SFEM_NL_MAX_IT=1 SFEM_LSOLVE_MAX_IT=40 SFEM_VERIFY_TOL=1e9 \\
      SFEM_CASE=cavity SFEM_N=8 SFEM_ELEMENT_REFINE_LEVEL=2 \\
      SFEM_GMG=1 SFEM_GMG_GALERKIN=2 SFEM_SMOOTHER=bjacobi"

for n in 1 2 4 8 16 32 64 128 192 256 288; do
    run_arm "bisect" "$n" "${BASE}"
done

echo "raw logs: $RAW"
