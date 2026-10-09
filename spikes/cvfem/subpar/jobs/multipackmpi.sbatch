#!/usr/bin/env bash
#SBATCH --account=c40
#SBATCH --job-name=cvfem-mpmpi
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=288
#SBATCH --time=00:28:00
#SBATCH --exclusive
#SBATCH --hint=nomultithread
#SBATCH --uenv-passthrough=ignore
#SBATCH --output=mpmpi_%j.out
#SBATCH --error=mpmpi_%j.err

# THE ONE CONFIGURATION WHERE THIS SESSION'S TWO HALVES MEET.
#
# Every MPI invariance measurement in this workstream -- before and after the coarse-packing fix
# -- used cavity N=8 (512 macro elements). Its coarse levels are a single pack at any sensible
# pack size, so the multi-pack defect could never appear there. Meanwhile every measurement OF
# the multi-pack defect was serial.
#
# cavity N=16 refine 4 is the case that actually triggers it: 4096 macro elements, two packs at
# the stock SFEM_PACK_SIZE=2048, 274,625 nodes, 1,098,500 dof, 3 levels. Serially it went
# 7 -> 22 linear iterations before the fix and back to 7 after. It has never been run
# multi-rank at all.
#
# So this asks the question neither half has answered: does decomposition invariance hold on the
# case that triggers the packing defect, now that coarse levels no longer pack?
#
# Block-Jacobi is carried as the control. It inverts one 4x4 block per node, so its owned output
# depends only on owned input and it cannot be partition dependent by construction -- any
# movement there is a defect rather than tuning.
#
# READ, committed before the run:
#
#   vanka flat across ranks        -> invariance holds on the multi-pack case; the fix and the
#                                     decomposition work are consistent with each other
#   vanka flat but != 7            -> invariance holds, but the parallel path reaches a
#                                     different operator than serial; that is a defect
#   vanka moving with rank count   -> a decomposition defect this case exposes and the 512-element
#                                     cases did not, and it outranks every serial result here
#   bjacobi moving                 -> more serious still: that arm cannot be partition dependent
#
# Serial reference, measured: vanka 7 at 5.82 s, 1,098,500 dof, 72 threads (job 4719975).
# Pure MPI here, one thread per rank, so atomics and thread scheduling are out of the picture.
# A count at the cap did NOT converge and is a floor; newton_converged is printed for every arm.

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
OUT="${OUT:-$SCRATCH/cvfem_mpmpi/${SLURM_JOB_ID:-manual}}"
RAW="$OUT/raw"
mkdir -p "$RAW"
[ -x "$BIN" ] || { echo "error: no binary at $BIN" >&2; exit 1; }

# Serial on this case measured 5.82 s at 72 threads; one thread per rank is slower, so 300 s
# bounds a deadlock rather than a slow run.
ARM_TIMEOUT="${ARM_TIMEOUT:-300}"

COMMON="OMP_NUM_THREADS=1 SFEM_FGMRES=1 SFEM_ENABLE_OUTPUT=0 \
        SFEM_PARALLEL_MF_LEGACY=1 SFEM_ALLOW_MPI=1 \
        SFEM_LSOLVE_MAX_IT=4000 SFEM_LSOLVE_RTOL=1e-4 SFEM_LSOLVE_ATOL=1e-14"
CASE="SFEM_CASE=cavity SFEM_N=16 SFEM_ELEMENT_REFINE_LEVEL=4"
ONESTEP="SFEM_NL_MAX_IT=1 SFEM_VERIFY_TOL=1e9"

echo "cavity N=16 refine 4 -- 4096 macro elements, 1,098,500 dof, 3 levels."
echo "Stock SFEM_PACK_SIZE=2048, i.e. TWO packs: the configuration that triggered the defect."
echo "Serial reference with the fix: vanka 7 at 5.82 s."
echo
for sm in vanka jacobi; do
    echo "--- smoother=$sm"
    printf "  %-8s %-12s %-12s %s\n" "ranks" "lin_it" "converged" "seconds"
    for n in 1 8 32 128 288; do
        tag="mp_${sm}_n${n}"
        timeout --signal=TERM --kill-after=30 "${ARM_TIMEOUT}" \
            srun -u --ntasks="${n}" --cpus-per-task=1 --kill-on-bad-exit=1 \
                --uenv="$UENV_IMAGE" --view="$UENV_VIEW" \
                bash -lc "${COMMON} ${CASE} ${ONESTEP} SFEM_GMG=1 SFEM_GMG_GALERKIN=2 \
                          SFEM_SMOOTHER=${sm} stdbuf -oL ${BIN} ${OUT}/${tag}" \
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
done
echo "the global problem is 274,625 nodes / 1,098,500 dof; lines below are per-rank locals"
grep -ah "^nnodes:" "$RAW"/mp_vanka_n1.log | head -1
echo "raw logs: $RAW"
