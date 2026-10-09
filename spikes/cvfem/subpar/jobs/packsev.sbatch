#!/usr/bin/env bash
#SBATCH --account=c40
#SBATCH --job-name=cvfem-packsev
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --time=00:25:00
#SBATCH --hint=nomultithread
#SBATCH --uenv-passthrough=ignore
#SBATCH --output=packsev_%j.out
#SBATCH --error=packsev_%j.err

# WHAT DOES THIS DEFECT DO AT THE DEFAULT PACK SIZE, ON A REAL PROBLEM SIZE?
#
# Everything measured so far forced SFEM_PACK_SIZE=256 on a 512-element mesh to get two packs.
# That is a reproducer, not a severity statement. The configuration that actually ships is the
# DEFAULT 2048, and it bites on any mesh with more than 2048 macro elements.
#
#     cavity N=16, refine 4   ->  4096 macro elements, 274,625 nodes, 1,098,500 dof
#                                 2 packs at the default, 1 pack at 4096, none at 0
#
# Established on the 512-element reproducer (serial, cd8db4f32 bisected as the first bad commit):
#
#     GALERKIN=2   7 -> 14      GALERKIN=0  12 -> 4000 (cap, NOT converged)
#
# GALERKIN=0 removes the Galerkin fold entirely, so the only thing cd8db4f32 leaves in play
# there is the prolongation's coarse columns -- and that is the arm that goes to the cap.
#
# READ, committed before the run:
#
#   pk=2048 matches pk=0        -> the default is safe at this size and the defect needs an
#                                  unusual pack size; severity drops sharply
#   pk=2048 matches pk=4096(1 pack) but both differ from 0 -> packing itself is implicated, not
#                                  the pack COUNT, and my whole reading is wrong
#   pk=2048 differs, pk=4096 agrees with 0 -> the default silently takes the broken path on any
#                                  problem past 2048 macro elements; that is the headline
#
# A count at the iteration cap did NOT converge and is a FLOOR, not a result -- newton_converged
# is printed for every arm. LSOLVE_MAX_IT is lowered to 1000 here purely to bound wall clock;
# any arm reporting 1000 is censored, not measured, and must be quoted that way.
# Timings carry the dof count, per the standing rule.

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
OUT="${OUT:-$SCRATCH/cvfem_packsev/${SLURM_JOB_ID:-manual}}"
RAW="$OUT/raw"
mkdir -p "$RAW"
[ -x "$BIN" ] || { echo "error: no binary at $BIN" >&2; exit 1; }

ARM_TIMEOUT="${ARM_TIMEOUT:-300}"

COMMON="OMP_NUM_THREADS=72 OMP_PROC_BIND=true SFEM_FGMRES=1 SFEM_ENABLE_OUTPUT=0 \
        SFEM_PARALLEL_MF_LEGACY=1 SFEM_ALLOW_MPI=1 \
        SFEM_LSOLVE_MAX_IT=1000 SFEM_LSOLVE_RTOL=1e-4 SFEM_LSOLVE_ATOL=1e-14"
CASE="SFEM_CASE=cavity SFEM_N=16 SFEM_ELEMENT_REFINE_LEVEL=4"
ONESTEP="SFEM_NL_MAX_IT=1 SFEM_VERIFY_TOL=1e9"

echo "cavity N=16 refine 4 -- 4096 macro elements, 1,098,500 dof, SERIAL, 72 threads"
echo "pk=0 atomic / pk=2048 THE DEFAULT (2 packs) / pk=4096 one pack"
echo
printf "  %-10s %-8s %-10s %-10s %s\n" "GALERKIN" "pack" "lin_it" "converged" "seconds"
for gal in 0 2; do
    for pk in 0 2048 4096; do
        tag="gal${gal}_pk${pk}"
        timeout --signal=TERM --kill-after=30 "${ARM_TIMEOUT}" \
            srun -u --ntasks=1 --cpus-per-task=72 --kill-on-bad-exit=1 \
                --uenv="$UENV_IMAGE" --view="$UENV_VIEW" \
                bash -lc "${COMMON} ${CASE} ${ONESTEP} SFEM_GMG=1 SFEM_SMOOTHER=vanka \
                          SFEM_GMG_GALERKIN=${gal} SFEM_PACK_SIZE=${pk} \
                          stdbuf -oL ${BIN} ${OUT}/${tag}" \
            > "$RAW/${tag}.log" 2>&1
        rc=$?
        it=$(grep -a lin_it_total "$RAW/${tag}.log" | head -1 | sed 's/.*lin_it_total: //')
        cv=$(grep -ao "newton_converged: [01]" "$RAW/${tag}.log" | head -1 | awk '{print $2}')
        sec=$(grep -a "ns_ssgmg: .* seconds" "$RAW/${tag}.log" | head -1 | sed 's/.*ssgmg: //')
        [ "$rc" = "124" ] && it="TIMEOUT"
        printf "  %-10s %-8s %-10s %-10s %s\n" "$gal" "$pk" "${it:-NONE}" "${cv:-?}" "${sec:-n/a}"
    done
    echo
done
grep -ah "^nnodes:" "$RAW"/gal2_pk0.log | head -1
echo "raw logs: $RAW"
