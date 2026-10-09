#!/usr/bin/env bash
#SBATCH --account=c40
#SBATCH --job-name=cvfem-stencil
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=288
#SBATCH --time=00:20:00
#SBATCH --exclusive
#SBATCH --hint=nomultithread
#SBATCH --uenv-passthrough=ignore
#SBATCH --output=stencil_%j.out
#SBATCH --error=stencil_%j.err

# ARE THE ASSEMBLED VANKA STENCILS DECOMPOSITION INVARIANT?
#
# Measured locally: they are not, and entries are going missing.
#
#     ranks         1          2          4          5          6          8
#     owned_slots  1728       1728       1728       1728       1728       1728
#     abs        81.7219    80.3268    77.5365    76.2905    75.9852    76.0782
#     lin_it        6          6          6          8         10         14
#
# owned_slots is the self-check: the sum runs over OWNED ELEMENTS, every macro-element is
# owned by exactly one rank, and the driver's global gate confirms the owned counts sum to the
# 64 elements of this case. So the probe visits the same (element, lattice node, slot) triples
# at every rank count -- 64 x 27 = 1728 -- and it does. The VALUES still move, and abs falls
# monotonically, which is what a sum LOSING contributions looks like. Reordering would leave
# abs fixed.
#
# WHY THE EARLIER VERSION OF THIS PROBE SAID THE OPPOSITE. It filtered on the row node being
# owned, and reported sum and abs bit-identical at every rank count. That filter discards
# exactly the rows that are wrong, so the all-clear was an artefact of the probe rather than a
# property of the code. Recorded here because the same blind spot applies to any owned-node
# keyed probe of an intermediate quantity.
#
# THE MECHANISM. build_fold_map discards its GalerkinLevel -- it ends with (void)gl -- and
# completes a node pair by summing every macro-element THIS RANK holds that carries the pair,
# writing the full sum back to each member so that a sweep reading only its own stencils still
# sees the whole entry. For a pair involving an owned node that is complete, because every
# element touching an owned node is local. The pairs that can be short are those between two
# NON-owned nodes of an aura element, and those entries sit in the 8x8 patch that is inverted
# to produce the correction at an owned node. Both sweeps share the fold, which is why additive
# and multiplicative Vanka both drift, and block-Jacobi -- which folds nothing -- does not.
#
# Everything upstream is already eliminated by measurement: the linearisation state, the patch
# multiplicities (1728 element and 4096 patch incidences, identical at every rank count), the
# input gather (the wrapper is correct, GhostsAndAura, and its no-MG counterpart has since been
# added), the coarse correction (the drift persists at SFEM_GMG_CGC=0 and with no multigrid at
# all), the sweep variant, and the nodal pressure-gradient reconstruction (invariant to
# fourteen figures under block-Jacobi, so its drift under Vanka is a consequence and not a
# cause).
#
# READ, committed before the run:
#
#   owned_slots 1728 everywhere and abs falling -> confirms the local result on cray-mpich; the
#                                                  fold owes a cross-rank completion
#   owned_slots NOT 1728 somewhere              -> the element-ownership restriction is not a
#                                                  partition on that cut and this probe is void
#                                                  too; fix the probe before reading anything
#   abs flat on cray-mpich                      -> the local OpenMPI result does not reproduce
#                                                  and the whole finding needs re-taking
#
# A TIMEOUT is unmeasured, not a result.

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
OUT="${OUT:-$SCRATCH/cvfem_stencil/${SLURM_JOB_ID:-manual}}"
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
        | stdbuf -oL grep -aE "^stencilsum|^smoothsum |lin_it_total|newton_converged|nnodes:"
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
MGV="SFEM_GMG=1 SFEM_GMG_GALERKIN=2 SFEM_SMOOTHER=vanka"
ONESTEP="SFEM_NL_MAX_IT=1 SFEM_VERIFY_TOL=1e9"

# The stencils and the smoother they produce, at every rank count across the transition.
for n in 1 2 3 4 5 6 7 8; do
    run_arm "sten" "$n" \
        "${COMMON} ${CAVITY} ${MGV} ${ONESTEP} SFEM_GMG_CGC=0 \
         SFEM_VANKA_STENCIL_SUM=1 SFEM_SMOOTHER_APPLY_SUM=1"
done

echo "raw logs: $RAW"
