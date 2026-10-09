#!/usr/bin/env bash
#SBATCH --account=c40
#SBATCH --job-name=cvfem-scale288
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=288
#SBATCH --time=00:30:00
#SBATCH --exclusive
#SBATCH --hint=nomultithread
#SBATCH --uenv-passthrough=ignore
#SBATCH --output=scale288_%j.out
#SBATCH --error=scale288_%j.err

# DOES DECOMPOSITION INVARIANCE HOLD AT FULL NODE WIDTH?
#
# Everything so far has been measured to eight ranks, because the case in use has 64 macro
# elements and the distributed cube guard refuses a mesh with fewer elements than ranks. The
# objective is one rank per core on a GH200 node -- 288 -- so this runs a case with 512 macro
# elements, which admits every rank count below.
#
# FOUR DEFECTS ARE FIXED AND CONFIRMED ON CRAY-MPICH AT 1..8 RANKS:
#
#   (1) GlobalDenseLU gathered the coarse right-hand side by rank-order tiling while scattering
#       the answer through the local-to-global map. Two-level block-Jacobi went from
#       158 / 158 / 158 / 726 at 1 / 2 / 4 / 8 to 158 everywhere.
#   (2) The fine Vanka patch entries were folded rank-locally, so pairs between two non-owned
#       nodes were short. Completed with a gather of each node's 27-direction lattice row.
#   (3) The level operators and their smoothers at depth carried no ParallelOperator face, so
#       level_owned returned the LOCAL count and reduce_norm2 skipped its allreduce; the
#       smoothers were additionally built on the raw operator and never gathered their input.
#   (4) The coarse Vanka patch entries had the same rank-local fold as (2), completed the same
#       way but keyed on each level's own mesh. Four-level Vanka went from 3 / 3 / 5 / 5 / 6 / 14
#       to 3 at every rank count.
#
# The one anomaly that remains is NOT a defect: block-Jacobi at four levels reads 45 instead of
# 43 at a single rank count, and only at rtol = 1e-3. Swept against the binding tolerance it is
# 25/25 at 1e-2, 43/45 at 1e-3, 56/56 at 1e-4 and 114/114 at 1e-5 -- a knife-edge at the
# convergence threshold, where reassociation in the collective dots crosses the bound an
# iteration later. Not a decomposition error.
#
# READ, committed before the run:
#
#   every arm equal to its own serial value at 1..288 -> invariance holds at node width and the
#                                                        parallelisation goal is met for this case
#   a count drifting only at the largest rank counts  -> a defect that needs a subdomain smaller
#                                                        than anything tested so far; the face
#                                                        check and stencilsum in the same arm
#                                                        separate bookkeeping from assembly
#   a count off by one or two at a single rank count  -> check it against rtol before calling it
#                                                        a defect; see the knife-edge above
#
# Timings are reported with the dof count, since a second without a problem size is not a
# result. A TIMEOUT is unmeasured. A count at the iteration cap did NOT converge and is a floor.

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
OUT="${OUT:-$SCRATCH/cvfem_scale288/${SLURM_JOB_ID:-manual}}"
RAW="$OUT/raw"
mkdir -p "$RAW"
[ -x "$BIN" ] || { echo "error: no binary at $BIN" >&2; exit 1; }

ARM_TIMEOUT="${ARM_TIMEOUT:-300}"

run_arm() {
    local tag="$1" nranks="$2"; shift 2
    echo "### ${tag} ranks=${nranks} ###"
    timeout --signal=TERM --kill-after=30 "${ARM_TIMEOUT}" \
        srun -u --ntasks="${nranks}" --cpus-per-task=1 --kill-on-bad-exit=1 \
            --time="00:$(printf '%02d' $(( (ARM_TIMEOUT + 59) / 60 ))):00" \
            --uenv="$UENV_IMAGE" --view="$UENV_VIEW" \
            bash -lc "$* stdbuf -oL ${BIN} ${OUT}/${tag}_n${nranks}" 2>&1 \
        | tee "$RAW/${tag}_n${nranks}.log" \
        | stdbuf -oL grep -aE "^stencilsum|lin_it_total|newton_converged|nnodes:|gmg: [0-9]+ levels|seconds"
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
        SFEM_LSOLVE_MAX_IT=4000 SFEM_LSOLVE_ATOL=1e-14"
# refine 8, not 4: 512 macro elements either way, but this gives FOUR levels and so actually
# exercises the depth fixes, where refine 4 gives three. 274,625 nodes, 1,098,500 dof, 12.7 s
# serial for one Newton step -- comfortably inside an arm, and ~953 nodes per rank at 288.
#
# At 288 ranks over 512 elements most ranks own one or two macro-elements, so this is an
# INVARIANCE test under an extreme cut, not a throughput measurement. A scaling run needs far
# more elements and is separate work.
CASE="SFEM_CASE=cavity SFEM_N=8 SFEM_ELEMENT_REFINE_LEVEL=8"
MG="SFEM_GMG=1 SFEM_GMG_GALERKIN=2"
ONESTEP="SFEM_NL_MAX_IT=1 SFEM_VERIFY_TOL=1e9"

# rtol away from the knife-edge, so a one-iteration difference cannot be mistaken for a defect.
TOL="SFEM_LSOLVE_RTOL=1e-4"

# Vanka carries the depth test on the large case: four levels, 1,098,500 dof, 6 linear
# iterations and 12.7 s serial for one Newton step.
for n in 1 8 32 64 128 288; do
    run_arm "vanka" "$n" \
        "${COMMON} ${CASE} ${MG} ${ONESTEP} ${TOL} SFEM_SMOOTHER=vanka SFEM_VANKA_STENCIL_SUM=1"
done

# Block-Jacobi runs on the SMALLER case, deliberately.
#
# It is here as a control, not as a depth test, and on the large case it is far too slow to
# serve as one: it needs tens of iterations where Vanka needs six, and serial at 1,098,500 dof
# ran past 600 s locally. With a 300 s arm cap that arm would time out, and a timed-out arm is
# unmeasured -- which would leave the control with no serial reference at all, the one thing it
# exists to provide. Refine 4 gives three levels and 143,748 dof, seconds per arm.
CASE_SMALL="SFEM_CASE=cavity SFEM_N=8 SFEM_ELEMENT_REFINE_LEVEL=4"
for n in 1 8 32 64 128 288; do
    run_arm "bjacobi" "$n" "${COMMON} ${CASE_SMALL} ${MG} ${ONESTEP} ${TOL} SFEM_SMOOTHER=bjacobi"
done

echo "raw logs: $RAW"
