#!/usr/bin/env bash
#SBATCH --account=c40
#SBATCH --job-name=cvfem-coarseconv
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=288
#SBATCH --time=00:20:00
#SBATCH --exclusive
#SBATCH --hint=nomultithread
#SBATCH --uenv-passthrough=ignore
#SBATCH --output=coarseconv_%j.out
#SBATCH --error=coarseconv_%j.err

# DOES THE COARSE BiCGStab CONVERGE, OR IS IT STAGNATING?
#
# Job 4720451 put the coarsest level on BiCGStab (SFEM_GMG_DENSE_LU_BELOW=4096) and the OUTER
# iteration count stopped being shape independent:
#
#     4x72  15      8x36  12      12x24 12      18x16 19      36x8 33      72x4 39
#
# against a flat 7 at every shape with the dense LU. Those timings were therefore NOT ranked:
# the shapes are not solving with the same preconditioner.
#
# Two explanations remain, and they lead to different conclusions:
#
#   (a) the coarse solve CONVERGES to its 1e-8 tolerance everywhere -> a well converged
#       iterative coarse solve still makes the outer count swing 12 -> 39, so this
#       preconditioner is genuinely decomposition dependent and the dense LU's exactness is
#       what bought the flat 7
#   (b) the coarse solve STAGNATES or hits its 2000 iteration budget -> the finding is only
#       that BiCGStab cannot solve THIS coarse system, and nothing general follows
#
# The logs from 4720451 cannot tell these apart: SFEM_GMG_COARSE_VERBOSE defaults to 0, so they
# carry no coarse convergence data at all. The per-call cost (37.4 / 12.8 / 4.2 ms at
# 4x72 / 12x24 / 72x4) is sublinear against an 18x rank increase, which is consistent with
# parallel speedup rather than with fewer iterations -- suggestive, not decisive.
#
# Three shapes: the baseline, the apparent minimum, and the lean extreme.
#
# READ, committed before the run:
#
#   coarse converges to 1e-8 at every shape, iteration counts similar -> (a). The dense LU is
#       the right default for invariance, and an iterative coarse solve trades it away.
#   coarse iteration counts differ sharply with shape                 -> (a) with a mechanism:
#       the coarse Krylov method is itself decomposition sensitive.
#   coarse hits 2000 / residual far above 1e-8                        -> (b). Narrow finding;
#       do not generalise about iterative coarse solves from it.
#
# Verbose output is one line per coarse iteration times one coarse solve per V-cycle, so it is
# summarised rather than dumped: calls, total coarse iterations, worst final residual.

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
BIN="${CVFEM_BIN:-./build/cvfem_hex8_ns_ssgmg}"
OUT="${OUT:-$SCRATCH/cvfem_coarseconv/${SLURM_JOB_ID:-manual}}"; RAW="$OUT/raw"; mkdir -p "$RAW"
[ -x "$BIN" ] || { echo "error: no binary at $BIN" >&2; exit 1; }
ARM_TIMEOUT="${ARM_TIMEOUT:-220}"

COMMON="SFEM_FGMRES=1 SFEM_ENABLE_OUTPUT=0 SFEM_PARALLEL_MF_LEGACY=1 SFEM_ALLOW_MPI=1 \
        SFEM_LSOLVE_MAX_IT=4000 SFEM_LSOLVE_RTOL=1e-4 SFEM_LSOLVE_ATOL=1e-14 \
        SFEM_GMG_DENSE_LU_BELOW=4096 SFEM_GMG_COARSE_MAX_IT=2000 SFEM_GMG_COARSE_VERBOSE=1"
CASE="SFEM_CASE=cavity SFEM_N=16 SFEM_ELEMENT_REFINE_LEVEL=4"
ONESTEP="SFEM_NL_MAX_IT=1 SFEM_VERIFY_TOL=1e9"

echo "cavity N=16 refine 4 -- 1,098,500 dof, 3 levels, coarsest 19,652 dofs on BiCGStab."
echo "Coarse tolerance rtol 1e-8, budget 2000.  Dense-LU reference: lin_it 7 at EVERY shape."
echo
printf "  %-9s %-9s %-12s %-14s %s\n" "shape" "lin_it" "coarse_calls" "coarse_its_tot" "worst final resid"
for shape in "4 72" "12 24" "72 4"; do
    set -- $shape; n=$1; t=$2; tag="s${n}x${t}"
    timeout --signal=TERM --kill-after=30 "${ARM_TIMEOUT}" \
        srun -u --ntasks="${n}" --cpus-per-task="${t}" --kill-on-bad-exit=1 \
            --uenv="$UENV_IMAGE" --view="$UENV_VIEW" \
            bash -lc "OMP_NUM_THREADS=${t} OMP_PROC_BIND=true ${COMMON} ${CASE} ${ONESTEP} \
                      SFEM_GMG=1 SFEM_GMG_GALERKIN=2 SFEM_SMOOTHER=vanka \
                      stdbuf -oL ${BIN} ${OUT}/${tag}" > "$RAW/${tag}.log" 2>&1
    rc=$?
    it=$(grep -a lin_it_total "$RAW/${tag}.log" | head -1 | sed 's/.*lin_it_total: //')
    calls=$(grep -am1 "coarse_solve" "$RAW/${tag}.log" | awk '{print $3}')
    # every verbose coarse iteration line, however the solver spells it
    cits=$(grep -acE "^ *(bcgs|BiCGStab|cg)[^a-z]" "$RAW/${tag}.log")
    worst=$(grep -aoE "(resid|residual|rnorm)[^0-9]*[0-9.]+e?[-+0-9]*" "$RAW/${tag}.log" \
            | grep -oE "[0-9.]+e?[-+0-9]*$" | sort -g | tail -1)
    [ "$rc" = "124" ] && it="TIMEOUT"
    printf "  %-9s %-9s %-12s %-14s %s\n" "${n}x${t}" "${it:-NONE}" "${calls:-n/a}" \
        "${cits:-0}" "${worst:-none reported}"
done
echo
echo "if coarse_its_tot reads 0 the solver prints nothing under VERBOSE=1 and this question"
echo "needs a different instrument, not a re-run."
echo "sample of whatever the coarse solve does print, from the 4x72 arm:"
grep -aiE "bcgs|coarse" "$RAW"/s4x72.log | grep -viE "coarse_solve|coarse_factor|coarse LU|coarse levels" | head -8
echo "raw logs: $RAW"
