#!/usr/bin/env bash
#SBATCH --account=c40
#SBATCH --job-name=cvfem-diagsum8
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=288
#SBATCH --time=00:20:00
#SBATCH --exclusive
#SBATCH --hint=nomultithread
#SBATCH --uenv-passthrough=ignore
#SBATCH --output=diagsum8_%j.out
#SBATCH --error=diagsum8_%j.err

# IS THE COARSE BLOCK DIAGONAL COMPLETE ON OWNED NODES AT EIGHT RANKS?
#
# The reframe this job acts on. Multigrid + block-Jacobi measures
#
#     1 rank 158    2 ranks 158    4 ranks 158    8 ranks 726
#
# so it is decomposition INVARIANT through four ranks and breaks only at eight. And with the
# coarse correction scaled to zero the sweep gives 320 at four ranks and 320 at eight -- the
# same number -- so with no coarse grid in the cycle the solver is invariant at eight ranks
# too. That localises the fault to the coarse-grid correction, at eight ranks specifically,
# rather than to a general rank dependence.
#
# galerkin_block_diag accumulates each coarse node's 4x4 centre block over the local reduce
# table and nothing exchanges the result, so a coarse node shared between ranks holds only
# its local contributions. That diagonal is what the coarse block-Jacobi smoother inverts.
# The owned sums were identical to 16 significant figures at 1, 2 and 4 ranks -- the owner
# was authoritative there -- and were never measured at eight, which is the only rank count
# where the cycle breaks.
#
# THIS RUNS PROBE FREE. SFEM_GMG_DIAG_OWNED_SUM is not gated on SFEM_GMG_CHECK, so none of
# the probing path runs: no colouring, no composite applies, no probed composite. That
# matters twice over. It avoids the deadlock that hung every GMG_CHECK arm at four ranks and
# above -- the colour count is derived from the rank-local sparsity pattern while the probe
# loop's composite() is collective, so ranks issue different numbers of collectives -- and it
# avoids every index-keyed probe vector, none of which is a valid distributed quantity.
#
# READ, committed before the run:
#
#   owned sums identical at 8 as at 1/2/4  -> the owner is authoritative, the coarse diagonal
#                                             is NOT the fault, and the coarse operator's
#                                             action or the coarse solve is the next target
#   owned sums differ at 8                 -> the partial sums are genuinely split across
#                                             ranks, the smoother inverts a wrong diagonal,
#                                             and the fix is a scatter_add before the gather
#
# gidsum is the global-id-weighted sum: it moves if the same total is distributed over
# different nodes, which a plain sum would hide. A TIMEOUT is unmeasured, not a result.
#
# Per-arm timeout, sized against the serial run: the serial arm is 0.16 s of solve, and the
# 8-rank arm after the nodal-gradient fix was about 3 s, so 120 s is far above any arm that
# is progressing and far below the allocation.

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
OUT="${OUT:-$SCRATCH/cvfem_diagsum8/${SLURM_JOB_ID:-manual}}"
RAW="$OUT/raw"
mkdir -p "$RAW"
[ -x "$BIN" ] || { echo "error: no binary at $BIN" >&2; exit 1; }

ARM_TIMEOUT="${ARM_TIMEOUT:-120}"

run_arm() {
    local tag="$1" nranks="$2"; shift 2
    echo "### ${tag} ranks=${nranks} ###"
    timeout --signal=TERM --kill-after=30 "${ARM_TIMEOUT}" \
        srun -u --ntasks="${nranks}" --cpus-per-task=1 --kill-on-bad-exit=1 \
            --time="00:$(printf '%02d' $(( (ARM_TIMEOUT + 59) / 60 ))):00" \
            --uenv="$UENV_IMAGE" --view="$UENV_VIEW" \
            bash -lc "$* stdbuf -oL ${BIN} ${OUT}/${tag}_n${nranks}" 2>&1 \
        | tee "$RAW/${tag}_n${nranks}.log" \
        | stdbuf -oL grep -aE "^diagsum |^  lin_it: |newton_converged|lin_it_total|nnodes:"
    local rc=${PIPESTATUS[0]}
    if [ "$rc" = "124" ] || [ "$rc" = "137" ]; then
        echo "  (arm exit: ${rc} -- TIMEOUT after ${ARM_TIMEOUT}s)"
    else
        echo "  (arm exit: ${rc})"
    fi
    echo
}

COMMON="OMP_NUM_THREADS=1 SFEM_FGMRES=1 SFEM_ENABLE_OUTPUT=0 \
        SFEM_PARALLEL_MF_LEGACY=1 SFEM_ALLOW_MPI=1 \
        SFEM_LSOLVE_MAX_IT=4000 SFEM_LSOLVE_ATOL=1e-12"
CAVITY="SFEM_CASE=cavity SFEM_N=4 SFEM_ELEMENT_REFINE_LEVEL=2"
MG="SFEM_GMG=1 SFEM_GMG_GALERKIN=2 SFEM_SMOOTHER=bjacobi"
DIAG="SFEM_GMG_DIAG_OWNED_SUM=1"

# The coarse diagonal, and the iteration count that goes with it, in the same run.
for n in 1 2 4 8; do
    run_arm "diagsum" "$n" "${COMMON} ${CAVITY} ${MG} ${DIAG}"
done

# The coarse correction switched off, at the two rank counts that disagree. This is the
# control: it already gave 320 and 320, and repeating it in the same job with the same binary
# keeps the comparison honest rather than resting on a number from an earlier build.
for n in 4 8; do
    run_arm "cgc0" "$n" "${COMMON} ${CAVITY} ${MG} ${DIAG} SFEM_GMG_CGC=0"
done

echo "raw logs: $RAW"
