#!/usr/bin/env bash
#SBATCH --account=c40
#SBATCH --job-name=cvfem-shapesat
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=288
#SBATCH --time=00:29:00
#SBATCH --exclusive
#SBATCH --hint=nomultithread
#SBATCH --uenv-passthrough=ignore
#SBATCH --output=shapesat_%j.out
#SBATCH --error=shapesat_%j.err

# THE SHAPE SWEEP AT A SIZE THAT ACTUALLY FILLS THE NODE.
#
# The first sweep (job 4720450) ran 1,098,500 dof over 288 cores -- about 3,800 dof per core.
# That is an under-filled problem, and the standing rule is to size until throughput saturates
# before comparing anything. Its verdict (4x72 fastest, monotone) was also dominated by the
# coarse dense factorisation at 57-92% of the run, so the shape trade was never visible.
#
# This raises the fine work 8x while leaving the coarsest level exactly where it was:
#
#     cavity N=16 refine 8 -> 4096 macro elements, 129^3 = 2,146,689 nodes, 8,586,756 dof
#                             4 levels (8 -> 4 -> 2 -> 1)
#                             coarsest still 17^3 = 4,913 nodes = 19,652 dofs
#                             ~29,800 dof per core at 288 cores
#
# So the dense factorisation becomes a fixed ~5 s overhead against a much larger solve, and
# whatever shape preference the solver itself has should emerge instead of being masked.
#
# Four shapes rather than six, because each arm is roughly eight times the work and the debug
# partition allows 29 minutes. They span the range: one rank per socket, then progressively
# leaner ranks.
#
# READ, committed before the run:
#
#   lin_it identical across shapes   -> same problem solved; wall clocks comparable
#   an interior minimum              -> the genuine sweet spot at saturation, and it is NOT the
#                                       baseline; report it with the phase split
#   still monotone toward 4x72       -> the baseline is right for the real reason, not just
#                                       because of the coarse solve
#   any arm OOM / at the cap         -> not a result; OOM is a resource limit, a capped count a
#                                       floor
#
# Timings carry the dof count. Every nnodes: line in a raw log is a PER-RANK local, not the
# problem size -- 8,586,756 dof is the global figure.

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
OUT="${OUT:-$SCRATCH/cvfem_shapesat/${SLURM_JOB_ID:-manual}}"; RAW="$OUT/raw"; mkdir -p "$RAW"
[ -x "$BIN" ] || { echo "error: no binary at $BIN" >&2; exit 1; }

# Eight times the fine work of the 9-50 s sweep, so 400 s bounds a deadlock rather than a slow
# arm; an arm that is merely slow must never be recorded as a timeout.
ARM_TIMEOUT="${ARM_TIMEOUT:-400}"

COMMON="SFEM_FGMRES=1 SFEM_ENABLE_OUTPUT=0 SFEM_PARALLEL_MF_LEGACY=1 SFEM_ALLOW_MPI=1 \
        SFEM_LSOLVE_MAX_IT=4000 SFEM_LSOLVE_RTOL=1e-4 SFEM_LSOLVE_ATOL=1e-14"
CASE="SFEM_CASE=cavity SFEM_N=16 SFEM_ELEMENT_REFINE_LEVEL=8"
ONESTEP="SFEM_NL_MAX_IT=1 SFEM_VERIFY_TOL=1e9"

echo "cavity N=16 refine 8 -- 4096 macro elements, 2,146,689 nodes, 8,586,756 dof, 4 levels."
echo "~29,800 dof per core at 288 cores.  MG + Vanka, GALERKIN=2, OMP_PROC_BIND=true."
echo "Under-filled reference (1,098,500 dof): 4x72 was 9.31 s, monotone, coarse_factor 57%."
echo
printf "  %-10s %-11s %-12s %-12s %-12s %s\n" "shape" "lin_it" "total (s)" "coarse_fac" "t_solve" "converged"
for shape in "4 72" "8 36" "18 16" "72 4"; do
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
    cf=$(grep -am1 "coarse_factor" "$RAW/${tag}.log" | awk '{print $2}')
    ts=$(grep -am1 "t_solve" "$RAW/${tag}.log" | sed 's/.*t_solve: \([0-9.]*\).*/\1/')
    cv=$(grep -ao "newton_converged: [01]" "$RAW/${tag}.log" | head -1 | awk '{print $2}')
    [ "$rc" = "124" ] && it="TIMEOUT"; [ "$it" = "4000" ] && it="4000(cap)"
    grep -aq "oom_kill" "$RAW/${tag}.log" && it="OOM-KILLED"
    printf "  %-10s %-11s %-12s %-12s %-12s %s\n" "${n}x${t}" "${it:-NONE}" "${sec:-n/a}" \
        "${cf:-n/a}" "${ts:-n/a}" "${cv:-?}"
done
echo
echo "global problem: 8,586,756 dof.  nnodes: lines in raw logs are PER-RANK locals."
grep -ah "^gmg: .* levels" "$RAW"/s4x72.log | head -1
grep -ah "^coarse LU:" "$RAW"/s4x72.log | head -1
echo "raw logs: $RAW"
