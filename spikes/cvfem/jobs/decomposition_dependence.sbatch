#!/usr/bin/env bash
#SBATCH --account=c40
#SBATCH --job-name=cvfem-decomp-dep
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=18
#SBATCH --time=00:28:00
#SBATCH --exclusive
#SBATCH --hint=nomultithread
#SBATCH --uenv-passthrough=ignore
#SBATCH --output=decomp_dep_%j.out
#SBATCH --error=decomp_dep_%j.err

# WHERE does the multi-rank gap enter, and is the algorithm decomposition-dependent?
#
# Serial takes 61 linear iterations on the cavity; two ranks take 3088 and four take 9082.
# Everything that has been checked individually is decomposition-INDEPENDENT: the residual
# matches serial exactly, the matrix-free apply is correct, the restriction does
# gather/scatter_add/gather, the prolongation gathers its input, the FGMRES reductions are
# collective, and the assembled coarse matrix is bit-identical at one and two ranks
# (frob2 = 2.580688703308e+02 to 13 digits) with an exact scatter-back. So the gap is not in
# any of those, and guessing which remaining piece it is has already cost one wrong answer.
#
# This measures the degradation factor per PRECONDITIONER CONFIGURATION at 1, 2 and 4 ranks.
# Anything decomposition-independent gives the same iteration count at every rank count; the
# configuration whose count explodes is the one carrying the dependence.
#
#   gmg0_bjacobi   no hierarchy, point-block preconditioner   -- the control: reads only its
#                  own row, so it cannot be decomposition-dependent by construction
#   gmg_bjacobi    V-cycle, point-block smoother              -- adds transfers + coarse solve
#   gmg_vanka      V-cycle, Vanka smoother                    -- adds patch solves
#
# THE MISSING CONTROL this job exists to supply: the SERIAL block-Jacobi number was never
# measured. Its 9125 iterations at two ranks may be degradation, or may simply be what
# block-Jacobi costs on this problem. Without the one-rank value the two-rank value says
# nothing, and that omission is why the gap has been attributed to the smoother on no evidence.
#
# Two hierarchy depths, because one candidate is depth-gated. Coarse-level Vanka is built from
# the LOCAL assembled BSR (make_diagonal_vanka_from_bsr), whose ghost and aura rows carry only
# local-element contributions and are therefore incomplete; a patch anchored on an aura
# macro-element then solves with a wrong matrix. That can only happen when a coarse level uses
# Vanka at all, which needs THREE levels -- at two levels the coarse level is the coarsest and
# is solved by the dense LU instead. cavity L2 gives two levels, poiseuille L4 gives three.
# If the gap is present at two levels, that candidate is not the cause.
#
# SINGLE-THREADED, every arm. Vanka's patch corrections are combined with atomics, so
# lin_it_total is thread-dependent: on the nozzle, 72 threads produced 4068 / 2521 / 4332 for
# three settings that single-threaded all give exactly 3202. A thread-noise artefact already
# masqueraded as a 38% improvement in this campaign once; it will not do so twice.

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
OUT="${OUT:-$SCRATCH/cvfem_decomp/${SLURM_JOB_ID:-manual}}"
RAW="$OUT/raw"
mkdir -p "$RAW"
[ -x "$BIN" ] || { echo "error: no binary at $BIN" >&2; exit 1; }

run_arm() {
    local tag="$1" nranks="$2"; shift 2
    echo "### ${tag} ranks=${nranks} ###"
    srun -u --ntasks="${nranks}" --cpus-per-task=1 --uenv="$UENV_IMAGE" --view="$UENV_VIEW" \
        bash -lc "$* stdbuf -oL ${BIN} ${OUT}/${tag}_n${nranks}" 2>&1 \
        | tee "$RAW/${tag}_n${nranks}.log" \
        | stdbuf -oL grep -aE "ndof:|gmg: |newton_converged|lin_it_total|preconditioner"
    echo
}

# OMP_NUM_THREADS=1 everywhere: see the note above about thread noise.
BASE="OMP_NUM_THREADS=1 SFEM_FGMRES=1 SFEM_ENABLE_OUTPUT=0 \
      SFEM_PARALLEL_MF_LEGACY=1 SFEM_ALLOW_MPI=1"
CAVITY="SFEM_CASE=cavity SFEM_N=4 SFEM_ELEMENT_REFINE_LEVEL=2"
POIS="SFEM_CASE=poiseuille SFEM_NX=12 SFEM_NY=3 SFEM_NZ=3 SFEM_ELEMENT_REFINE_LEVEL=4"

for geom_tag in cavity_L2 pois_L4; do
    case "$geom_tag" in
        cavity_L2) GEOM="$CAVITY" ;;
        pois_L4)   GEOM="$POIS" ;;
    esac
    for n in 1 2 4; do
        # No hierarchy at all. Point-block reads only its own row, so any rank-dependence
        # here is in the Krylov layer or the operator, not in a smoother.
        run_arm "${geom_tag}_gmg0_bjacobi" "$n" \
            "${BASE} ${GEOM} SFEM_GMG=0 SFEM_PRECOND=bjacobi"

        # V-cycle with the immune smoother: isolates transfers + coarse solve.
        run_arm "${geom_tag}_gmg_bjacobi" "$n" \
            "${BASE} ${GEOM} SFEM_GMG=1 SFEM_GMG_GALERKIN=2 SFEM_SMOOTHER=bjacobi"

        # V-cycle with Vanka: adds the patch solves.
        run_arm "${geom_tag}_gmg_vanka" "$n" \
            "${BASE} ${GEOM} SFEM_GMG=1 SFEM_GMG_GALERKIN=2 SFEM_SMOOTHER=vanka"
    done
done

echo "raw logs: $RAW"
