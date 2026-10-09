#!/usr/bin/env bash
#SBATCH --account=c40
#SBATCH --job-name=cvfem-shapeiter
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=288
#SBATCH --time=00:29:00
#SBATCH --exclusive
#SBATCH --hint=nomultithread
#SBATCH --uenv-passthrough=ignore
#SBATCH --output=shapeiter_%j.out
#SBATCH --error=shapeiter_%j.err

# DOES THE SHAPE OPTIMUM MOVE ONCE THE COARSE FACTORISATION IS GONE?
#
# The dense-LU sweep (job 4720450, same case) found 4x72 fastest and MONOTONE -- no interior
# minimum -- at lin_it 7 for every shape:
#
#     4x72   9.31 s   coarse_factor  5.351 (57%)   t_solve 1.290
#     8x36  11.59 s                  7.824         t_solve 1.326
#    12x24  14.66 s                 10.864         t_solve 1.385
#    18x16  20.24 s                 16.214         t_solve 1.530
#     36x8  28.58 s                 24.602         t_solve 1.545
#     72x4  49.99 s                 45.778 (92%)   t_solve 1.727
#
# coarse_factor grows 8.6x across the sweep while t_solve moves only 34%. So the curve is the
# coarse solve, not the shape: every rank holds the whole 19,652-dof dense factor, threaded
# getrf slows as threads shrink, and the n^2 allreduce dears as ranks grow. The shape trade the
# sweep was meant to expose is buried under a 57-92% overhead.
#
# SFEM_GMG_DENSE_LU_BELOW (dofs, gating local AND global, default 1<<30) diverts the coarsest
# level to BiCGStab. That CHANGES THE PRECONDITIONER, so this is not comparable to the numbers
# above as a like-for-like; lin_it is reported for every shape precisely so any change is
# visible rather than assumed away. The driver's own comment warns that an iterative coarse
# solve "hands up a correction that is noise while reporting success", which is why the coarse
# tolerance stays tight and the iteration budget is generous.
#
# READ, committed before the run:
#
#   lin_it still 7, and an INTERIOR minimum appears -> the real shape sweet spot, previously
#                                                      masked by the factorisation
#   lin_it still 7, still monotone toward 4x72      -> fat ranks win on their own merits, not
#                                                      just because of the coarse solve
#   lin_it moves with shape                         -> the iterative coarse solve is
#                                                      shape-sensitive; these timings are then
#                                                      NOT comparable and must not be ranked
#   lin_it much worse than 7                        -> BiCGStab is a poor coarse solve here and
#                                                      the dense LU is the right default

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
OUT="${OUT:-$SCRATCH/cvfem_shapeiter/${SLURM_JOB_ID:-manual}}"; RAW="$OUT/raw"; mkdir -p "$RAW"
[ -x "$BIN" ] || { echo "error: no binary at $BIN" >&2; exit 1; }
ARM_TIMEOUT="${ARM_TIMEOUT:-220}"

COMMON="SFEM_FGMRES=1 SFEM_ENABLE_OUTPUT=0 SFEM_PARALLEL_MF_LEGACY=1 SFEM_ALLOW_MPI=1 \
        SFEM_LSOLVE_MAX_IT=4000 SFEM_LSOLVE_RTOL=1e-4 SFEM_LSOLVE_ATOL=1e-14 \
        SFEM_GMG_DENSE_LU_BELOW=4096 SFEM_GMG_COARSE_MAX_IT=2000"
CASE="SFEM_CASE=cavity SFEM_N=16 SFEM_ELEMENT_REFINE_LEVEL=4"
ONESTEP="SFEM_NL_MAX_IT=1 SFEM_VERIFY_TOL=1e9"

echo "cavity N=16 refine 4 -- 4096 macro elements, 274,625 nodes, 1,098,500 dof, 3 levels."
echo "Coarsest level 19,652 dofs pushed onto BiCGStab (DENSE_LU_BELOW=4096). Fixed 288 cores."
echo "Dense-LU reference at 4x72: 9.31 s total, lin_it 7."
echo
printf "  %-10s %-11s %-12s %-13s %-12s %s\n" "shape" "lin_it" "total (s)" "coarse_solve" "t_solve" "denseLU?"
for shape in "4 72" "8 36" "12 24" "18 16" "36 8" "72 4"; do
    set -- $shape; n=$1; t=$2; tag="s${n}x${t}"
    timeout --signal=TERM --kill-after=30 "${ARM_TIMEOUT}" \
        srun -u --ntasks="${n}" --cpus-per-task="${t}" --kill-on-bad-exit=1 \
            --uenv="$UENV_IMAGE" --view="$UENV_VIEW" \
            bash -lc "OMP_NUM_THREADS=${t} OMP_PROC_BIND=true ${COMMON} ${CASE} ${ONESTEP} \
                      SFEM_GMG=1 SFEM_GMG_GALERKIN=2 SFEM_SMOOTHER=vanka \
                      stdbuf -oL ${BIN} ${OUT}/${tag}" > "$RAW/${tag}.log" 2>&1
    rc=$?
    it=$(grep -a lin_it_total "$RAW/${tag}.log" | head -1 | sed 's/.*lin_it_total: //')
    sec=$(grep -a "ns_ssgmg: .* seconds" "$RAW/${tag}.log" | head -1 | sed 's/.*ssgmg: \([0-9.]*\).*/\1/')
    csv=$(grep -am1 "coarse_solve" "$RAW/${tag}.log" | awk '{print $2}')
    ts=$(grep -am1 "t_solve" "$RAW/${tag}.log" | sed 's/.*t_solve: \([0-9.]*\).*/\1/')
    lu=$(grep -ac "^coarse LU:" "$RAW/${tag}.log")
    [ "$rc" = "124" ] && it="TIMEOUT"; [ "$it" = "4000" ] && it="4000(cap)"
    grep -aq "oom_kill" "$RAW/${tag}.log" && it="OOM-KILLED"
    printf "  %-10s %-11s %-12s %-13s %-12s %s\n" "${n}x${t}" "${it:-NONE}" "${sec:-n/a}" \
        "${csv:-n/a}" "${ts:-n/a}" "$([ "${lu:-0}" -gt 0 ] && echo "yes(GATE FAILED)" || echo no)"
done
echo
echo "denseLU must read 'no' on every arm, else the gate did not divert and this is the old run."
echo "problem size is 1,098,500 dof GLOBAL; any nnodes: line in the raw logs is a per-rank local."
echo "raw logs: $RAW"
