#!/usr/bin/env bash
#SBATCH --account=c40
#SBATCH --job-name=cvfem-bjcontrol
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=288
#SBATCH --time=00:28:00
#SBATCH --exclusive
#SBATCH --hint=nomultithread
#SBATCH --uenv-passthrough=ignore
#SBATCH --output=bjcontrol_%j.out
#SBATCH --error=bjcontrol_%j.err

# CORRECTING AN INVALID CONTROL OF MY OWN MAKING.
#
# Job 4720014 ran the block-Jacobi control on cavity N=8 refine 8 (4 levels, 1,098,500 dof) and
# compared it against "before the fix: block-Jacobi 11". That 11 was never measured on that
# case -- it belongs to cavity N=8 refine 4 (3 levels, 143,748 dof), job 4719099. So the
# TIMEOUT / 4000 results there are compared against a baseline that does not belong to them and
# establish nothing. The Vanka half of that job IS valid: lin_it 6 at 1/8/32/64 ranks, matching
# the pre-fix result exactly.
#
# One detail from that run is worth keeping: it was a convergence failure, not a hang. 64 ranks
# COMPLETED in 198.13 s against a 240 s arm cap, so the lower rank counts were doing the same
# non-converging work and simply ran out of wall clock.
#
# (1) THE CONTROL AGAINST A NUMBER THAT EXISTS. Block-Jacobi on the case where flat 11 was
#     recorded. Block-Jacobi inverts one 4x4 block per node, so its owned output depends only on
#     owned input and it is structurally partition independent: any movement here is a defect,
#     not tuning.
#
# (2) IS THE 4000 ON THE HARDER CASE THE SMOOTHER, OR THE FIX? With SFEM_PACK_SIZE=0 nothing
#     packs and the fix is inert -- ret->pack_size > 0 is false -- so that arm behaves
#     identically before and after. Comparing it against the stock 2048 therefore isolates
#     packing from the smoother. Run at a LOOSER rtol on purpose: two arms both sitting at an
#     iteration cap are censored, not equal, and would prove nothing.
#
# READ, committed before the run:
#
#   (1) flat across ranks            -> invariance intact; the fix did not disturb it
#   (1) moving                       -> a defect, and it outranks the serial win
#   (2) pk=0 and pk=2048 agree       -> the hard case simply defeats block-Jacobi; packing and
#                                       the fix are exonerated
#   (2) pk=0 and pk=2048 differ      -> packing still changes this path and the fix is incomplete
#
# A count at the cap did not converge and is a FLOOR. newton_converged is printed for every arm.
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
OUT="${OUT:-$SCRATCH/cvfem_bjcontrol/${SLURM_JOB_ID:-manual}}"
RAW="$OUT/raw"
mkdir -p "$RAW"
[ -x "$BIN" ] || { echo "error: no binary at $BIN" >&2; exit 1; }

ARM_TIMEOUT="${ARM_TIMEOUT:-240}"
ONESTEP="SFEM_NL_MAX_IT=1 SFEM_VERIFY_TOL=1e9"

echo "########## (1) BLOCK-JACOBI INVARIANCE, ON THE CASE WHERE 11 WAS RECORDED ##########"
echo "cavity N=8 refine 4 -- 3 levels, 143,748 dof. Pre-fix baseline: flat 11 at 1..288 ranks."
echo
MPI_COMMON="OMP_NUM_THREADS=1 SFEM_FGMRES=1 SFEM_ENABLE_OUTPUT=0 \
            SFEM_PARALLEL_MF_LEGACY=1 SFEM_ALLOW_MPI=1 \
            SFEM_LSOLVE_MAX_IT=4000 SFEM_LSOLVE_RTOL=1e-4 SFEM_LSOLVE_ATOL=1e-14"
printf "  %-8s %-12s %-12s %s\n" "ranks" "lin_it" "converged" "seconds"
for n in 1 8 32 64; do
    tag="bj_n${n}"
    timeout --signal=TERM --kill-after=30 "${ARM_TIMEOUT}" \
        srun -u --ntasks="${n}" --cpus-per-task=1 --kill-on-bad-exit=1 \
            --uenv="$UENV_IMAGE" --view="$UENV_VIEW" \
            bash -lc "${MPI_COMMON} SFEM_CASE=cavity SFEM_N=8 SFEM_ELEMENT_REFINE_LEVEL=4 \
                      ${ONESTEP} SFEM_GMG=1 SFEM_GMG_GALERKIN=2 SFEM_SMOOTHER=jacobi \
                      stdbuf -oL ${BIN} ${OUT}/${tag}" \
        > "$RAW/${tag}.log" 2>&1
    rc=$?
    it=$(grep -a lin_it_total "$RAW/${tag}.log" | head -1 | sed 's/.*lin_it_total: //')
    cv=$(grep -ao "newton_converged: [01]" "$RAW/${tag}.log" | head -1 | awk '{print $2}')
    sec=$(grep -a "ns_ssgmg: .* seconds" "$RAW/${tag}.log" | head -1 | sed 's/.*ssgmg: //')
    [ "$rc" = "124" ] && it="TIMEOUT"
    printf "  %-8s %-12s %-12s %s\n" "$n" "${it:-NONE}" "${cv:-?}" "${sec:-n/a}"
done
echo
grep -ah "^nnodes:" "$RAW"/bj_n1.log | head -1
echo

echo "########## (2) ON THE HARD CASE: IS IT THE SMOOTHER OR THE PACKING? ##########"
echo "cavity N=8 refine 8 -- 4 levels, 1,098,500 dof, SERIAL, 72 threads, rtol 1e-2 so it"
echo "converges and the counts are real rather than two censored caps."
echo "pk=0 leaves the fix inert and nothing packed, so it reads the same before and after."
echo
SER="OMP_NUM_THREADS=72 OMP_PROC_BIND=true SFEM_FGMRES=1 SFEM_ENABLE_OUTPUT=0 \
     SFEM_PARALLEL_MF_LEGACY=1 SFEM_ALLOW_MPI=1 \
     SFEM_LSOLVE_MAX_IT=2000 SFEM_LSOLVE_RTOL=1e-2 SFEM_LSOLVE_ATOL=1e-14"
printf "  %-10s %-8s %-12s %-12s %s\n" "smoother" "pack" "lin_it" "converged" "seconds"
for sm in jacobi vanka; do
    for pk in 0 2048; do
        tag="hard_${sm}_pk${pk}"
        timeout --signal=TERM --kill-after=30 "${ARM_TIMEOUT}" \
            srun -u --ntasks=1 --cpus-per-task=72 --kill-on-bad-exit=1 \
                --uenv="$UENV_IMAGE" --view="$UENV_VIEW" \
                bash -lc "${SER} SFEM_CASE=cavity SFEM_N=8 SFEM_ELEMENT_REFINE_LEVEL=8 \
                          ${ONESTEP} SFEM_GMG=1 SFEM_GMG_GALERKIN=2 SFEM_SMOOTHER=${sm} \
                          SFEM_PACK_SIZE=${pk} stdbuf -oL ${BIN} ${OUT}/${tag}" \
            > "$RAW/${tag}.log" 2>&1
        rc=$?
        it=$(grep -a lin_it_total "$RAW/${tag}.log" | head -1 | sed 's/.*lin_it_total: //')
        cv=$(grep -ao "newton_converged: [01]" "$RAW/${tag}.log" | head -1 | awk '{print $2}')
        sec=$(grep -a "ns_ssgmg: .* seconds" "$RAW/${tag}.log" | head -1 | sed 's/.*ssgmg: //')
        [ "$rc" = "124" ] && it="TIMEOUT"
        [ "$it" = "2000" ] && it="2000(cap)"
        printf "  %-10s %-8s %-12s %-12s %s\n" "$sm" "$pk" "${it:-NONE}" "${cv:-?}" "${sec:-n/a}"
    done
done
echo
grep -ah "^nnodes:" "$RAW"/hard_vanka_pk0.log | head -1
echo "raw logs: $RAW"
