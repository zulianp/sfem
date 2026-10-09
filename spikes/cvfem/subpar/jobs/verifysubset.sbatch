#!/usr/bin/env bash
#SBATCH --account=c40
#SBATCH --job-name=cvfem-vsubset
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --time=00:28:00
#SBATCH --hint=nomultithread
#SBATCH --uenv-passthrough=ignore
#SBATCH --output=vsubset_%j.out
#SBATCH --error=vsubset_%j.err

# AN EARLY SIGNAL ON THE FIX, HOURS BEFORE THE FULL MATRIX CAN RUN.
#
# The full serial verification matrix (job 4720012) is queued behind 1,130 pending jobs on the
# normal partition and is estimated to start TOMORROW 01:36. It gates the commit and it gates
# the source cleanup being held back. That is too long to learn nothing in the meantime.
#
# verify_report.sh takes VERIFY_GROUPS (default "unit mms bc port budget step pump nozzle").
# Only two of those groups set SFEM_GMG=1 and therefore DEREFINE -- `port` (line 304) and
# `budget` (line 350, ladder "2:4 2:8 4:8") -- and derefinement is the only thing the
# coarse-packing fix touches: derefine_op clears pack_size so a coarse level never packs and
# never renumbers. Every other group runs SFEM_GMG=0 and cannot reach the changed code at all.
#
# WHAT THIS IS AND IS NOT. It is an early signal: do the derefining cases still run, converge,
# and land on the values the tracked report records? It is NOT the byte-identical diff. That
# still requires the full matrix over all eight groups, because the report generator builds its
# tables from the complete evidence set and a partial run would regenerate a partial report.
# Nothing tracked gets regenerated from this.
#
# EXPECTED. Those coarse meshes are 8, 27 and 64 elements against a 2048 pack size, so they
# were single-pack before the fix, where the permutation is the identity. The fix should
# therefore change no number here; what it will add is one printed line per run announcing that
# coarse levels are not packed, which the report generator cannot pick up because it extracts
# named fields by regex rather than embedding log text.
#
# READ, committed before the run:
#
#   both groups run and their values match the tracked report -> the fix is inert where it is
#       reachable serially, as designed; the full matrix should come back with an empty diff
#   a value moved                                             -> the fix is NOT inert serially
#       and the commit is blocked until that is understood
#   a case fails that previously passed                       -> a serial regression, which
#       outranks every distributed and throughput result in this branch

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
OUT="${OUT:-$SCRATCH/cvfem_vsubset/${SLURM_JOB_ID:-manual}}"
mkdir -p "$OUT"
[ -x "$BIN" ] || { echo "error: no binary at $BIN" >&2; exit 1; }

echo "groups: port budget  (the only two that derefine)"
echo "evidence: $OUT"
echo

srun -u --ntasks=1 --cpus-per-task=72 \
     --uenv="$UENV_IMAGE" --view="$UENV_VIEW" \
     bash -lc "OMP_NUM_THREADS=72 OMP_PROC_BIND=true VERIFY_GROUPS='port budget' \
               DRIVER='$SPIKE_ROOT/$BIN' OUT='$OUT' ./scripts/verify_report.sh"
rc=$?
echo
echo "verify_report exit: $rc"
echo
echo "=== did the fix announce itself? (it should, once per run that derefines)"
grep -rl "coarse levels are not packed" "$OUT" 2>/dev/null | wc -l
echo "=== runs that did NOT converge"
grep -rl "newton_converged: 0" "$OUT"/*.log 2>/dev/null | wc -l
echo
echo "=== the values to eyeball against the tracked report"
for f in "$OUT"/port_*.log "$OUT"/budget_*.log; do
    [ -f "$f" ] || continue
    printf "  %-34s %s\n" "$(basename "$f")" \
        "$(grep -am1 -E 'u_linf:|sum of continuity' "$f" | cut -c1-70)"
done
echo
echo "evidence directory: $OUT"
