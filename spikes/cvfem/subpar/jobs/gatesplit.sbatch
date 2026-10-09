#!/usr/bin/env bash
#SBATCH --account=c40
#SBATCH --job-name=cvfem-gatesplit
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=288
#SBATCH --time=00:20:00
#SBATCH --exclusive
#SBATCH --hint=nomultithread
#SBATCH --uenv-passthrough=ignore
#SBATCH --output=gatesplit_%j.out
#SBATCH --error=gatesplit_%j.err

# ONE DEFECT, TWO SYMPTOMS: THE RANK-LOCAL FOLD
#
# build_fold_map discards its GalerkinLevel -- it ends with (void)gl -- and galerkin_build_inverse
# groups slots by destination block over [kbegin, kend), the slots THIS RANK holds. So a node pair
# carried by two macro-elements folds to the full sum only when both elements are local. That is
# guaranteed for pairs touching an owned node, because every element touching an owned node is
# local; it is not guaranteed for pairs between two NON-owned nodes of an aura element.
#
# SYMPTOM ONE, the Vanka stencils. Confirmed on cray-mpich at eight rank counts:
#
#     ranks     1       2       3       4       5       6       7       8
#     slots   1728    1728    1728    1728    1728    1728    1728    1728
#     abs    81.722  80.327  78.538  77.537  76.291  75.985  75.445  76.078
#     lin_it    6       6       6       6       8      10      12      14
#
# owned_slots is the self-check: summed over OWNED ELEMENTS, and every macro-element is owned by
# exactly one rank, so the probe visits identical triples at every rank count. abs falls, which is
# a sum LOSING contributions; reordering would leave it fixed. Note abs already drops at two ranks
# while lin_it is still 6 -- the loss starts at any cut and only shows in the count once it is big
# enough.
#
# SYMPTOM TWO, the folded coarse BSR. Measured locally with the assembly gate, after BOTH of its
# own flaws were repaired (the probe direction was keyed on the local dof index; the comparison ran
# over the whole local range including ghost rows that were never meant to agree):
#
#     gate rel     1 rank 4.1e-16 OK    2 ranks 0.299    4 ranks 0.367    8 ranks 0.514
#
# and the split of the two sides says which one moves:
#
#     composite R.A.P   bit-identical at 1/2/4/8 (abs 3.2882378949663007 throughout)
#     assembled apply   sum 0.827 -> 0.807 -> 0.282 -> 0.337, abs 3.988 -> 3.825
#
# So the composite is invariant and the assembled matrix is not. That does NOT contradict applysum,
# which measured invariance to sixteen figures: applysum applies a_c, the element-matrix operator,
# while the gate applies the probed/folded BSR. Different objects, one root cause.
#
# WHY THE COARSE SOLVE SURVIVES IT ANYWAY. make_dense_lu_from_bsr_global reads OWNED ROWS ONLY,
# and owned rows are complete, so the short rows never reach the factorisation. Multigrid with
# block-Jacobi measures 158 linear iterations at 1, 2, 4, 5, 6 and 8 ranks. Vanka does not survive
# it, because a patch inverts a block that contains the incomplete entries.
#
# READ, committed before the run:
#
#   gate OK at 1 and MISMATCH growing with ranks, composite flat -> confirms the local result; the
#                                                                   fold owes a cross-rank completion
#   composite also moving on cray-mpich -> the local OpenMPI split does not reproduce and the
#                                          attribution has to be re-taken
#   bjacobi not 158 somewhere           -> a regression from the four edits since the last Alps
#                                          build, and that comes first
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
OUT="${OUT:-$SCRATCH/cvfem_gatesplit/${SLURM_JOB_ID:-manual}}"
RAW="$OUT/raw"
mkdir -p "$RAW"
[ -x "$BIN" ] || { echo "error: no binary at $BIN" >&2; exit 1; }

ARM_TIMEOUT="${ARM_TIMEOUT:-180}"

run_arm() {
    local tag="$1" nranks="$2"; shift 2
    echo "### ${tag} ranks=${nranks} ###"
    timeout --signal=TERM --kill-after=30 "${ARM_TIMEOUT}" \
        srun -u --ntasks="${nranks}" --cpus-per-task=1 --kill-on-bad-exit=1 \
            --time="00:$(printf '%02d' $(( (ARM_TIMEOUT + 59) / 60 ))):00" \
            --uenv="$UENV_IMAGE" --view="$UENV_VIEW" \
            bash -lc "$* stdbuf -oL ${BIN} ${OUT}/${tag}_n${nranks}" 2>&1 \
        | tee "$RAW/${tag}_n${nranks}.log" \
        | stdbuf -oL grep -aE "^gatesplit|^assembly gate|^stencilsum|lin_it_total|newton_converged"
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
MGJ="SFEM_GMG=1 SFEM_GMG_GALERKIN=2 SFEM_SMOOTHER=bjacobi"
ONESTEP="SFEM_NL_MAX_IT=1 SFEM_VERIFY_TOL=1e9"

# (1) The gate and its split. SYMBOLIC_PATTERN=0 so the too-narrow pattern is not the cause.
for n in 1 2 4 5 6 8; do
    run_arm "gate" "$n" \
        "${COMMON} ${CAVITY} ${MGJ} ${ONESTEP} SFEM_GMG_CHECK=1 \
         SFEM_GMG_SYMBOLIC_PATTERN=0 SFEM_GMG_GATE_SPLIT=1"
done

# (2) The regression gate. This must be 158 at every rank count.
for n in 1 2 4 5 6 8; do
    run_arm "full_jac" "$n" "${COMMON} ${CAVITY} ${MGJ}"
done

echo "raw logs: $RAW"
