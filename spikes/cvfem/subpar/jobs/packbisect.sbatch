#!/usr/bin/env bash
#SBATCH --account=c40
#SBATCH --job-name=cvfem-packbisect
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --time=00:25:00
#SBATCH --hint=nomultithread
#SBATCH --uenv-passthrough=ignore
#SBATCH --output=packbisect_%j.out
#SBATCH --error=packbisect_%j.err

# WHERE DOES THE MULTI-PACK DEFECT LIVE?
#
# Established locally on a two-second mesh (cavity N=8, refine 4, 512 macro elements):
# the packed operator is WRONG WHENEVER THERE IS MORE THAN ONE PACK, and only then.
#
#     SFEM_PACK_SIZE=0    atomic    lin_it  7
#     SFEM_PACK_SIZE=256  2 packs   lin_it 14      <-- the defect
#     SFEM_PACK_SIZE=512  1 pack    lin_it  7
#     SFEM_PACK_SIZE=2048 1 pack    lin_it  7
#
# This is SERIAL. None of the four MPI fixes in this branch executes at one rank, and the
# trigger is pack count, not mesh size -- the earlier N=16 case was not special, it merely
# exceeded the 2048 default. Severity: SFEM_PACK_SIZE defaults to 2048, so any run with more
# than 2048 macro elements silently takes this path, and packed throughput is the standing
# performance gate.
#
# This job asks ONE question: which layer is wrong?
#
# The ordering is the standing rule for this codebase -- clear the bare Krylov method before
# the smoother, and the smoother before the coarse space. I went straight to the nodal
# gradient probe instead and learned nothing, so this starts at the bottom.
#
#   GMG=0             no hierarchy at all. A split here means the BASE OPERATOR APPLY is
#                     wrong and every result above it is downstream. This is the arm that
#                     matters; the rest only narrow it further.
#   GMG + jacobi      block-Jacobi inverts one 4x4 block per node, so it cannot itself be
#                     pack-dependent. A split here but not at GMG=0 implicates the hierarchy.
#   GMG + vanka + RC_EXACT_JAC=0   frozen Rhie-Chow drops the nodal reconstruction out of the
#                     Jacobian. If the split vanishes here, the fault is in that reconstruction.
#   GMG + vanka       the configuration the split was found in, as measured.
#
# READ, committed before the run:
#
#   split at GMG=0                  -> base operator apply; nothing above it is worth looking
#                                      at until it is fixed
#   no split at GMG=0, split above  -> assembly (Galerkin/Vanka) reads the packed operator in
#                                      a way the apply does not
#   split vanishes at RC_EXACT_JAC=0 -> the nodal gradient reconstruction, which is where the
#                                      two-pass packed drain lives
#   no split anywhere but vanka     -> Vanka patch assembly specifically
#
# A count at the iteration cap did not converge and is a floor, not a result; newton_converged
# is printed for every arm and must be read before any count is compared.

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
OUT="${OUT:-$SCRATCH/cvfem_packbisect/${SLURM_JOB_ID:-manual}}"
RAW="$OUT/raw"
mkdir -p "$RAW"
[ -x "$BIN" ] || { echo "error: no binary at $BIN" >&2; exit 1; }

# The serial arm of this case is ~2 s on a laptop; 300 s is far past generous, and an arm that
# is merely slow must never be recorded as a timeout.
ARM_TIMEOUT="${ARM_TIMEOUT:-300}"

COMMON="OMP_NUM_THREADS=72 OMP_PROC_BIND=true SFEM_FGMRES=1 SFEM_ENABLE_OUTPUT=0 \
        SFEM_PARALLEL_MF_LEGACY=1 SFEM_ALLOW_MPI=1 \
        SFEM_LSOLVE_MAX_IT=4000 SFEM_LSOLVE_RTOL=1e-4 SFEM_LSOLVE_ATOL=1e-14"
CASE="SFEM_CASE=cavity SFEM_N=8 SFEM_ELEMENT_REFINE_LEVEL=4"
ONESTEP="SFEM_NL_MAX_IT=1 SFEM_VERIFY_TOL=1e9"

run_arm() {
    local cfgname="$1" pk="$2"; shift 2
    local tag="${cfgname}_pk${pk}"
    timeout --signal=TERM --kill-after=20 "${ARM_TIMEOUT}" \
        srun -u --ntasks=1 --cpus-per-task=72 --kill-on-bad-exit=1 \
            --uenv="$UENV_IMAGE" --view="$UENV_VIEW" \
            bash -lc "$* SFEM_PACK_SIZE=${pk} stdbuf -oL ${BIN} ${OUT}/${tag}" \
        > "$RAW/${tag}.log" 2>&1
    local rc=$?
    local it cv
    it=$(grep -a lin_it_total "$RAW/${tag}.log" | head -1 | sed 's/.*lin_it_total: //')
    cv=$(grep -ao "newton_converged: [01]" "$RAW/${tag}.log" | head -1 | awk '{print $2}')
    printf "  %-26s pk=%-6s %-10s conv=%-3s rc=%s\n" \
        "$cfgname" "$pk" "lin_it=${it:-NONE}" "${cv:-?}" "$rc"
}

echo "cavity N=8 refine 4 -- 512 macro elements, SERIAL, 72 threads"
echo "pk=0 atomic / pk=256 TWO PACKS / pk=512 one pack"
echo

echo "### (1) NO HIERARCHY -- the arm that decides whether anything above matters"
for pk in 0 256 512; do
    run_arm "gmg0" "$pk" "${COMMON} ${CASE} ${ONESTEP} SFEM_GMG=0"
done
echo

echo "### (2) MG + block-Jacobi -- structurally cannot be pack-dependent itself"
for pk in 0 256 512; do
    run_arm "mg_jacobi" "$pk" "${COMMON} ${CASE} ${ONESTEP} \
        SFEM_GMG=1 SFEM_GMG_GALERKIN=2 SFEM_SMOOTHER=jacobi"
done
echo

echo "### (3) MG + Vanka, FROZEN Rhie-Chow -- drops the nodal reconstruction from the Jacobian"
for pk in 0 256 512; do
    run_arm "mg_vanka_rc0" "$pk" "${COMMON} ${CASE} ${ONESTEP} \
        SFEM_GMG=1 SFEM_GMG_GALERKIN=2 SFEM_SMOOTHER=vanka SFEM_RC_EXACT_JAC=0"
done
echo

echo "### (4) MG + Vanka as measured -- expect 7 / 14 / 7, reproducing the laptop"
for pk in 0 256 512; do
    run_arm "mg_vanka" "$pk" "${COMMON} ${CASE} ${ONESTEP} \
        SFEM_GMG=1 SFEM_GMG_GALERKIN=2 SFEM_SMOOTHER=vanka"
done
echo
echo "raw logs: $RAW"
