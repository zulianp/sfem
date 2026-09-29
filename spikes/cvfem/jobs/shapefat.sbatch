#!/usr/bin/env bash
#SBATCH --account=c40
#SBATCH --job-name=cvfem-shapefat
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=288
#SBATCH --time=00:25:00
#SBATCH --exclusive
#SBATCH --hint=nomultithread
#SBATCH --uenv-passthrough=ignore
#SBATCH --output=shapefat_%j.out
#SBATCH --error=shapefat_%j.err

# BELOW THE BASELINE: DO FEWER, FATTER RANKS WIN FURTHER?
#
# Every shape sweep so far went UPWARD from 4x72 and found it fastest, monotonically, at both
# problem sizes. At 8,586,756 dof:
#
#     4x72  11.44 s   coarse_factor 5.414 (47%)   t_solve 1.669   lin_it 5
#     8x36  12.97 s                 7.779         t_solve 1.787   lin_it 5
#    18x16  22.61 s                16.945         t_solve 2.457   lin_it 5
#     72x4  50.74 s                45.385 (89%)   t_solve 2.400   lin_it 5
#
# The dominant term is a dense coarse factorisation that EVERY rank holds in full and assembles
# through an n^2 allreduce. That cost falls as ranks fall, and at ONE rank it changes character
# entirely: the serial branch (make_dense_lu_from_bsr) factorises once, with no replication and
# no allreduce at all. So the untested direction is downward.
#
# The counter-pressure is NUMA. The node is 4 sockets / 36 NUMA domains, and state vectors are
# std::vector<real_t> zero-filled by the constructing thread, so pages land where that thread
# ran. One rank spanning 288 threads crosses every domain; 4x72 keeps each team inside a socket.
# That is the trade this measures, and it is genuinely two-sided.
#
# READ, committed before the run:
#
#   1x288 or 2x144 faster than 4x72  -> the baseline is not the optimum and the coarse solve is
#                                       still deciding; report the phase split, since the win
#                                       would be in coarse_factor rather than in t_solve
#   t_solve WORSE at 1x288 while total is better -> first-touch NUMA is biting the solve exactly
#                                       as predicted, and the win is bought from the coarse
#                                       solve alone; that is a coarse-solver argument, not a
#                                       shape argument
#   4x72 still fastest               -> the baseline is the optimum across the whole range and
#                                       the tuning question is settled
#   lin_it not 5 everywhere          -> shapes are not solving the same problem; do NOT rank
#                                       the timings
#
# Saturated size, so the comparison is made where throughput actually matters. Timings carry the
# dof count. nnodes: lines in raw logs are PER-RANK locals, not the problem size.

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
OUT="${OUT:-$SCRATCH/cvfem_shapefat/${SLURM_JOB_ID:-manual}}"; RAW="$OUT/raw"; mkdir -p "$RAW"
[ -x "$BIN" ] || { echo "error: no binary at $BIN" >&2; exit 1; }
ARM_TIMEOUT="${ARM_TIMEOUT:-400}"

COMMON="SFEM_FGMRES=1 SFEM_ENABLE_OUTPUT=0 SFEM_PARALLEL_MF_LEGACY=1 SFEM_ALLOW_MPI=1 \
        SFEM_LSOLVE_MAX_IT=4000 SFEM_LSOLVE_RTOL=1e-4 SFEM_LSOLVE_ATOL=1e-14"
CASE="SFEM_CASE=cavity SFEM_N=16 SFEM_ELEMENT_REFINE_LEVEL=8"
ONESTEP="SFEM_NL_MAX_IT=1 SFEM_VERIFY_TOL=1e9"

echo "cavity N=16 refine 8 -- 2,146,689 nodes, 8,586,756 dof, 4 levels, ~29,800 dof/core."
echo "MG + Vanka, GALERKIN=2, dense LU coarse.  Reference: 4x72 = 11.44 s, lin_it 5."
echo "1x288 uses the SERIAL coarse branch: one factorisation, no replication, no allreduce."
echo
printf "  %-9s %-9s %-12s %-12s %-12s %s\n" "shape" "lin_it" "total (s)" "coarse_fac" "t_solve" "converged"
for shape in "1 288" "2 144" "4 72"; do
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
    printf "  %-9s %-9s %-12s %-12s %-12s %s\n" "${n}x${t}" "${it:-NONE}" "${sec:-n/a}" \
        "${cf:-n/a}" "${ts:-n/a}" "${cv:-?}"
done
echo
echo "lin_it must read 5 on every arm, else the shapes are not comparable."
echo "global problem: 8,586,756 dof."
echo "raw logs: $RAW"
