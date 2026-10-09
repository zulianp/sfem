#!/usr/bin/env bash
#SBATCH --account=c40
#SBATCH --job-name=cvfem-fixcheck
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=288
#SBATCH --time=00:28:00
#SBATCH --exclusive
#SBATCH --hint=nomultithread
#SBATCH --uenv-passthrough=ignore
#SBATCH --output=fixcheck_%j.out
#SBATCH --error=fixcheck_%j.err

# DOES THE COARSE-PACKING FIX HOLD UNDER MPI, AND IS THE ONE SLOWER ARM NOISE?
#
# The fix: derefine_op clears pack_size, so a coarse level never packs and therefore never
# renumbers the mesh that the hierarchy's transfers index by fine macro-corner id. Confirmed
# serially at 1,098,500 dof with the stock SFEM_PACK_SIZE=2048:
#
#     GALERKIN=2   22 -> 7 linear its, 8.97 s -> 5.82 s
#     GALERKIN=0   1000(cap, NOT converged) -> 17,  222.65 s -> 12.97 s
#
# TWO THINGS THAT ARE NOT YET ESTABLISHED.
#
# (1) DECOMPOSITION INVARIANCE. Earlier in this workstream four decomposition defects were
#     fixed and MG+Vanka was shown to give the SAME linear iteration count from 1 to 288 ranks
#     on this case (lin_it 6, 4 levels, 1,098,500 dof), with a block-Jacobi control flat at 11.
#     That was measured with coarse levels PACKED. They are now unpacked, which changes coarse
#     setup on every rank, so the invariance has to be re-measured rather than assumed to carry
#     over. Block-Jacobi inverts one 4x4 block per node and is structurally partition
#     independent, so it is the control: any movement there is a defect, not tuning.
#
# (2) THE ONE ARM THAT GOT SLOWER. GALERKIN=2 at pk=4096 (one pack) went 5.65 s -> 5.87 s,
#     about 3.9%. Every other arm was unchanged or much faster. The standing rule is no
#     performance regressions, so this is repeated to establish the run-to-run band instead of
#     being explained away. If the spread across repeats covers 0.22 s it is noise; if the
#     repeats cluster tightly above 5.65 s it is a real cost and must be reported as one.
#
# READ, committed before the run:
#
#   vanka lin_it identical at every rank count -> invariance survives the fix
#   vanka lin_it moving                        -> the fix broke the decomposition work, and
#                                                 that outranks the serial win
#   bjacobi moving                             -> same, and more serious: that arm cannot be
#                                                 partition dependent by construction
#   repeat spread >= 0.22 s                    -> the 3.9% is noise
#
# A count at the iteration cap did not converge and is a FLOOR. newton_converged is printed for
# every arm. Timings carry the dof count.

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
OUT="${OUT:-$SCRATCH/cvfem_fixcheck/${SLURM_JOB_ID:-manual}}"
RAW="$OUT/raw"
mkdir -p "$RAW"
[ -x "$BIN" ] || { echo "error: no binary at $BIN" >&2; exit 1; }

# Serial on this case ran ~21 s before; 240 s per arm is generous and bounds a deadlock.
ARM_TIMEOUT="${ARM_TIMEOUT:-240}"

MPI_COMMON="OMP_NUM_THREADS=1 SFEM_FGMRES=1 SFEM_ENABLE_OUTPUT=0 \
            SFEM_PARALLEL_MF_LEGACY=1 SFEM_ALLOW_MPI=1 \
            SFEM_LSOLVE_MAX_IT=4000 SFEM_LSOLVE_RTOL=1e-4 SFEM_LSOLVE_ATOL=1e-14"
INV_CASE="SFEM_CASE=cavity SFEM_N=8 SFEM_ELEMENT_REFINE_LEVEL=8"
ONESTEP="SFEM_NL_MAX_IT=1 SFEM_VERIFY_TOL=1e9"

echo "########## (1) DECOMPOSITION INVARIANCE WITH THE FIX ##########"
echo "cavity N=8 refine 8 -- 4 levels, 1,098,500 dof.  Before the fix: vanka 6 at 1..288 ranks,"
echo "block-Jacobi 11.  Pure MPI, one thread per rank."
echo
for sm in vanka jacobi; do
    echo "--- smoother=$sm"
    printf "  %-8s %-12s %-12s %s\n" "ranks" "lin_it" "converged" "seconds"
    for n in 1 8 32 64; do
        tag="inv_${sm}_n${n}"
        timeout --signal=TERM --kill-after=30 "${ARM_TIMEOUT}" \
            srun -u --ntasks="${n}" --cpus-per-task=1 --kill-on-bad-exit=1 \
                --uenv="$UENV_IMAGE" --view="$UENV_VIEW" \
                bash -lc "${MPI_COMMON} ${INV_CASE} ${ONESTEP} SFEM_GMG=1 SFEM_GMG_GALERKIN=2 \
                          SFEM_SMOOTHER=${sm} stdbuf -oL ${BIN} ${OUT}/${tag}" \
            > "$RAW/${tag}.log" 2>&1
        rc=$?
        it=$(grep -a lin_it_total "$RAW/${tag}.log" | head -1 | sed 's/.*lin_it_total: //')
        cv=$(grep -ao "newton_converged: [01]" "$RAW/${tag}.log" | head -1 | awk '{print $2}')
        sec=$(grep -a "ns_ssgmg: .* seconds" "$RAW/${tag}.log" | head -1 | sed 's/.*ssgmg: //')
        [ "$rc" = "124" ] && it="TIMEOUT"
        printf "  %-8s %-12s %-12s %s\n" "$n" "${it:-NONE}" "${cv:-?}" "${sec:-n/a}"
    done
    echo
done

echo "########## (2) NOISE BAND FOR THE ONE ARM THAT GOT SLOWER ##########"
echo "cavity N=16 refine 4 -- 1,098,500 dof, GALERKIN=2, pk=4096 (one pack), 72 threads."
echo "Before the fix this arm was 5.65 s; with the fix one measurement read 5.87 s (+3.9%)."
echo
SER="OMP_NUM_THREADS=72 OMP_PROC_BIND=true SFEM_FGMRES=1 SFEM_ENABLE_OUTPUT=0 \
     SFEM_PARALLEL_MF_LEGACY=1 SFEM_ALLOW_MPI=1 \
     SFEM_LSOLVE_MAX_IT=1000 SFEM_LSOLVE_RTOL=1e-4 SFEM_LSOLVE_ATOL=1e-14"
printf "  %-10s %-10s %s\n" "repeat" "lin_it" "seconds"
for r in 1 2 3 4; do
    tag="noise_r${r}"
    timeout --signal=TERM --kill-after=30 "${ARM_TIMEOUT}" \
        srun -u --ntasks=1 --cpus-per-task=72 --kill-on-bad-exit=1 \
            --uenv="$UENV_IMAGE" --view="$UENV_VIEW" \
            bash -lc "${SER} SFEM_CASE=cavity SFEM_N=16 SFEM_ELEMENT_REFINE_LEVEL=4 ${ONESTEP} \
                      SFEM_GMG=1 SFEM_GMG_GALERKIN=2 SFEM_SMOOTHER=vanka SFEM_PACK_SIZE=4096 \
                      stdbuf -oL ${BIN} ${OUT}/${tag}" \
        > "$RAW/${tag}.log" 2>&1
    it=$(grep -a lin_it_total "$RAW/${tag}.log" | head -1 | sed 's/.*lin_it_total: //')
    sec=$(grep -a "ns_ssgmg: .* seconds" "$RAW/${tag}.log" | head -1 | sed 's/.*ssgmg: //')
    printf "  %-10s %-10s %s\n" "$r" "${it:-NONE}" "${sec:-n/a}"
done
echo
grep -ah "^nnodes:" "$RAW"/noise_r1.log | head -1
echo "raw logs: $RAW"
