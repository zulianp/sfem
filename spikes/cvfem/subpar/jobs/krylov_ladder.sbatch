#!/usr/bin/env bash
#SBATCH --account=c40
#SBATCH --job-name=cvfem-krylov-ladder
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=288
#SBATCH --time=00:25:00
#SBATCH --exclusive
#SBATCH --hint=nomultithread
#SBATCH --uenv-passthrough=ignore
#SBATCH --output=krylov_ladder_%j.out
#SBATCH --error=krylov_ladder_%j.err

# THE LADDER: debug the Krylov method before the smoother, and the smoother before the
# coarse space. Every arm here is expected to be DECOMPOSITION INVARIANT -- the iteration
# count must not move with the rank count -- and any arm that is not marks the layer where
# the defect lives.
#
# RUNG 1: FGMRES with no multigrid at all, preconditioned by block-Jacobi.
#   Block-Jacobi inverts one 4x4 nodal block, so its owned output depends only on owned
#   input and it is partition-independent by construction. With no hierarchy, no transfers
#   and no coarse operator in the picture, this arm isolates the Krylov layer alone.
#   Already measured and ALREADY BROKEN (cavity L2, single-threaded, all converged):
#       1 rank 832   2 ranks 1776   4 ranks 2718      (2.1x / 3.3x)
#
# RUNG 2: multigrid with the same block-Jacobi smoother.
#   Trivially decomposition invariant for the same reason, so this arm should reproduce its
#   1-rank count at every rank count. Already measured (cavity L2):
#       1 rank 162   2 ranks 6104   4 ranks 10067     (38x / 62x)
#
# INSTRUMENTS (SFEM_FGMRES_DIAG=1 prints both, once per solve):
#   fgmres: distributed <0|1>  size <p>  n(owned) <n>  n_local <n>  alloc <n>
#       Every reduction routes through a cvfem::Domain populated only if the constructor's
#       dynamic_pointer_cast to ParallelOperator succeeded. If an operator reaches the solver
#       with that face erased the cast fails SILENTLY, distributed() is 0, and every dot and
#       norm is rank-local while the source still reads as collective. If this line says
#       "distributed 0" on a multi-rank arm, that is the defect and nothing else needs
#       examining.
#   fgmres: orth_worst <e>  its <n>
#       max_i |<v_i, v_{j+1}>| over the Arnoldi basis. Modified Gram-Schmidt has just
#       projected each new vector against every earlier one, so on a correct reduction this
#       is at rounding level. Measured serially on cavity L2: 2.98e-14 to 3.39e-13 across
#       every solve, with lin_it_total 832. A rank-local dot leaves it O(1) while the solve
#       still converges slowly -- the signature that reads as "a weaker preconditioner" and
#       sends the investigation upward instead of downward.
#
# NO OPENMP ANYWHERE: OMP_NUM_THREADS=1 on every arm. Pure MPI removes atomics and thread
# scheduling as explanations, so whatever degradation survives is genuinely about the
# decomposition. It also matters because Vanka's patch corrections combine with atomics and
# thread noise already faked a 38% effect once in this campaign (nozzle 4068/2521/4332 at 72
# threads, all exactly 3202 at one thread).
#
# THE 288-RANK ARM is the sharpest form of the test: one rank per core on the node, so the
# domain is cut as finely as the hardware allows. It runs on Poiseuille L4 (8281 nodes,
# 33,124 dof, ~29 nodes per rank) and NOT on the cavity, whose 729 nodes would give 2.5 nodes
# per rank -- ranks owning almost nothing, which stresses the collectives but measures
# nothing about the algorithm. An arm that fails purely for size reasons is not a finding.
#
# SFEM_LSOLVE_MAX_IT is raised from its 1000 default: the earlier Poiseuille no-multigrid
# arms reported 5000/29000/33000 with newton_converged 0, which is the cap times the Newton
# step count, not a convergence measurement. An iteration count from a run that did not
# converge is a floor and must never be compared against one that did.

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
OUT="${OUT:-$SCRATCH/cvfem_krylov_ladder/${SLURM_JOB_ID:-manual}}"
RAW="$OUT/raw"
mkdir -p "$RAW"
[ -x "$BIN" ] || { echo "error: no binary at $BIN" >&2; exit 1; }

run_arm() {
    local tag="$1" nranks="$2"; shift 2
    echo "### ${tag} ranks=${nranks} ###"
    srun -u --ntasks="${nranks}" --cpus-per-task=1 --uenv="$UENV_IMAGE" --view="$UENV_VIEW" \
        bash -lc "$* stdbuf -oL ${BIN} ${OUT}/${tag}_n${nranks}" 2>&1 \
        | tee "$RAW/${tag}_n${nranks}.log" \
        | stdbuf -oL grep -aE "^fgmres:|newton_converged|lin_it_total|ndof|nnodes"
    echo
}

COMMON="OMP_NUM_THREADS=1 SFEM_FGMRES=1 SFEM_FGMRES_DIAG=1 SFEM_ENABLE_OUTPUT=0 \
        SFEM_PARALLEL_MF_LEGACY=1 SFEM_ALLOW_MPI=1 SFEM_LSOLVE_MAX_IT=4000"

CAVITY="SFEM_CASE=cavity SFEM_N=4 SFEM_ELEMENT_REFINE_LEVEL=2"
POIS="SFEM_CASE=poiseuille SFEM_NX=12 SFEM_NY=3 SFEM_NZ=3 SFEM_ELEMENT_REFINE_LEVEL=4"

# ---- Rung 1: no multigrid. The Krylov layer alone.
for n in 1 2 4; do
    run_arm "r1_cavity_nomg" "$n" "${COMMON} ${CAVITY} SFEM_GMG=0 SFEM_SMOOTHER=bjacobi"
done

# ---- Rung 2: multigrid with the block-Jacobi smoother. Expected identical at every n.
for n in 1 2 4; do
    run_arm "r2_cavity_mgjac" "$n" \
        "${COMMON} ${CAVITY} SFEM_GMG=1 SFEM_GMG_GALERKIN=2 SFEM_SMOOTHER=bjacobi"
done

# ---- Pure MPI at full node width, on the case that is big enough for it.
for n in 1 288; do
    run_arm "r1_pois_nomg"  "$n" "${COMMON} ${POIS} SFEM_GMG=0 SFEM_SMOOTHER=bjacobi"
    run_arm "r2_pois_mgjac" "$n" \
        "${COMMON} ${POIS} SFEM_GMG=1 SFEM_GMG_GALERKIN=2 SFEM_SMOOTHER=bjacobi"
done

echo "raw logs: $RAW"
