#!/usr/bin/env bash
#SBATCH --account=c40
#SBATCH --job-name=cvfem-inv288
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=288
#SBATCH --time=00:28:00
#SBATCH --exclusive
#SBATCH --hint=nomultithread
#SBATCH --uenv-passthrough=ignore
#SBATCH --output=inv288_%j.out
#SBATCH --error=inv288_%j.err

# DOES INVARIANCE STILL HOLD AT A FULL NODE, WITH THE COARSE-PACKING FIX?
#
# The headline of this workstream is that MG + Vanka gives the SAME linear iteration count from
# 1 to 288 ranks on a four-level, 1,098,500-dof problem -- lin_it 6 -- after four decomposition
# defects were fixed. That was measured with coarse levels PACKED (job 4719099).
#
# derefine_op now clears pack_size, so no coarse level packs or renumbers. Job 4720014 re-ran
# the Vanka sweep and got 6 at 1, 8, 32 and 64 ranks -- but it stopped at 64. The top of the
# range is precisely where the headline lives, and 288 ranks is where a rank owns so little of
# the mesh that setup effects dominate, so it is the arm most likely to expose a coarse-level
# change. Claiming invariance "to 288" on the strength of a sweep that stopped at 64 would be
# quoting a result I do not have.
#
# Rank 1 is repeated inside this job rather than cited from another, so the comparison is
# self-contained and does not rest on matching two runs' conditions by memory.
#
# Pure MPI, one thread per rank: no OpenMP, so atomics and thread scheduling are removed as
# explanations and whatever survives is genuinely about the decomposition.
#
# READ, committed before the run:
#
#   6 at 1 / 128 / 288        -> invariance survives the fix across the full node; the headline
#                                stands as stated
#   6 low, moving at 288      -> a coarse-level setup effect the fix introduced, and it outranks
#                                the serial win
#   anything at the cap       -> did NOT converge; that count is a floor, not a result
#
# newton_converged is printed for every arm and must be read before any count is compared.
# Timings carry the dof count.

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
OUT="${OUT:-$SCRATCH/cvfem_inv288/${SLURM_JOB_ID:-manual}}"
RAW="$OUT/raw"
mkdir -p "$RAW"
[ -x "$BIN" ] || { echo "error: no binary at $BIN" >&2; exit 1; }

# Serial on this case measured 17.8 s; 240 s per arm is generous and bounds a deadlock rather
# than a slow run, so an arm that is merely slow is never recorded as a timeout.
ARM_TIMEOUT="${ARM_TIMEOUT:-240}"

COMMON="OMP_NUM_THREADS=1 SFEM_FGMRES=1 SFEM_ENABLE_OUTPUT=0 \
        SFEM_PARALLEL_MF_LEGACY=1 SFEM_ALLOW_MPI=1 \
        SFEM_LSOLVE_MAX_IT=4000 SFEM_LSOLVE_RTOL=1e-4 SFEM_LSOLVE_ATOL=1e-14"
CASE="SFEM_CASE=cavity SFEM_N=8 SFEM_ELEMENT_REFINE_LEVEL=8"
ONESTEP="SFEM_NL_MAX_IT=1 SFEM_VERIFY_TOL=1e9"

echo "cavity N=8 refine 8 -- 4 levels, 1,098,500 dof, MG + Vanka, pure MPI."
echo "Pre-fix: lin_it 6 at 1..288.  Post-fix so far: 6 at 1/8/32/64 (job 4720014)."
echo
printf "  %-8s %-12s %-12s %s\n" "ranks" "lin_it" "converged" "seconds"
for n in 1 128 288; do
    tag="inv_n${n}"
    timeout --signal=TERM --kill-after=30 "${ARM_TIMEOUT}" \
        srun -u --ntasks="${n}" --cpus-per-task=1 --kill-on-bad-exit=1 \
            --uenv="$UENV_IMAGE" --view="$UENV_VIEW" \
            bash -lc "${COMMON} ${CASE} ${ONESTEP} SFEM_GMG=1 SFEM_GMG_GALERKIN=2 \
                      SFEM_SMOOTHER=vanka stdbuf -oL ${BIN} ${OUT}/${tag}" \
        > "$RAW/${tag}.log" 2>&1
    rc=$?
    it=$(grep -a lin_it_total "$RAW/${tag}.log" | head -1 | sed 's/.*lin_it_total: //')
    cv=$(grep -ao "newton_converged: [01]" "$RAW/${tag}.log" | head -1 | awk '{print $2}')
    sec=$(grep -a "ns_ssgmg: .* seconds" "$RAW/${tag}.log" | head -1 | sed 's/.*ssgmg: //')
    [ "$rc" = "124" ] && it="TIMEOUT"
    [ "$it" = "4000" ] && it="4000(cap)"
    printf "  %-8s %-12s %-12s %s\n" "$n" "${it:-NONE}" "${cv:-?}" "${sec:-n/a}"
done
echo
echo "sizes are per-rank locals below; the global problem is 274,625 nodes / 1,098,500 dof"
grep -ah "^nnodes:" "$RAW"/inv_n1.log | head -1
echo "raw logs: $RAW"
