#!/usr/bin/env bash
#SBATCH --account=c40
#SBATCH --job-name=cvfem-state-by-rank
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=18
#SBATCH --time=00:20:00
#SBATCH --exclusive
#SBATCH --hint=nomultithread
#SBATCH --uenv-passthrough=ignore
#SBATCH --output=state_by_rank_%j.out
#SBATCH --error=state_by_rank_%j.err

# DOES THE LINEARISATION STATE ITSELF DIFFER BETWEEN DECOMPOSITIONS?
#
# SFEM_GMG_CONST_STATE=1 makes the iteration counts identical at 1, 2 and 4 ranks
# (177/177/177 cgc0, 92/92/92 full, job 4714783), while the normal path degrades
# 61 -> 3394 -> 7255. That localises the decomposition dependence to the state the
# cycle is linearised about, and nowhere else. It does not say WHICH level's state,
# nor whether the state differs at all or merely produces different arithmetic.
#
# SFEM_GMG_STATE_SUM=1 prints, at every refresh_gmg, each level's state reduced over
# the OWNED range only and keyed by GLOBAL id:
#
#     gmgstate level <i>: owned_nodes <n>  sum <s>  gidsum <g>  free_abs <f>
#
# Owned-only and gid-keyed is what makes the number comparable across partitions --
# local indices are incomparable by construction, and the ghost and aura entries each
# rank also stores belong to their owners. gidsum is the discriminating column: a
# permutation of the same values moves it while leaving sum untouched. free_abs sums
# |v| over UNCONSTRAINED dofs only, which separates real interior content from
# boundary data.
#
# READ:
#   owned_nodes must equal the serial node count at every rank count. If it does not,
#   the ownership partition itself is wrong and nothing below it means anything.
#
#   The FIRST print of each run is the pre-Newton refresh and is uninformative: the
#   state there is boundary data alone (initial state checksum 81, p_exact 0 on the
#   cavity), every non-zero entry sits on a constrained dof, both transfers zero
#   constrained dofs on their output, and the hierarchy is derefined with homogeneous
#   Dirichlet data -- so level 1 is legitimately zero there. Serially measured:
#     call 1:  level 0 sum 81         gidsum 125644      free_abs 0
#              level 1 sum 0          gidsum 0           free_abs 0
#     call 2:  level 0 sum 81.000539  gidsum 135258.704  free_abs 90.655
#              level 1 sum -1.169408  gidsum -113.391    free_abs 9.0456
#
#   Compare call 2 across rank counts. If level 0 agrees and level 1 does not, the
#   restriction is where the state diverges. If level 0 already disagrees, the fine
#   state is divergent before any transfer runs and the coarse levels are innocent.
#   If both agree to rounding, the state does NOT differ and CONST_STATE collapses the
#   gap for some other reason -- which would falsify the reading that motivated this job.
#
# Newton paths may diverge between rank counts after the first step, so only the EARLY
# calls are comparable arm to arm; later ones drift for legitimate reasons.
#
# SINGLE-THREADED: Vanka's patch corrections combine with atomics, and thread noise
# already faked a 38% effect once in this campaign (nozzle 4068/2521/4332 at 72
# threads, all exactly 3202 at one thread).

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
OUT="${OUT:-$SCRATCH/cvfem_state_by_rank/${SLURM_JOB_ID:-manual}}"
RAW="$OUT/raw"
mkdir -p "$RAW"
[ -x "$BIN" ] || { echo "error: no binary at $BIN" >&2; exit 1; }

run_arm() {
    local tag="$1" nranks="$2"; shift 2
    echo "### ${tag} ranks=${nranks} ###"
    srun -u --ntasks="${nranks}" --cpus-per-task=1 --uenv="$UENV_IMAGE" --view="$UENV_VIEW" \
        bash -lc "$* stdbuf -oL ${BIN} ${OUT}/${tag}_n${nranks}" 2>&1 \
        | tee "$RAW/${tag}_n${nranks}.log" \
        | stdbuf -oL grep -aE "^gmgstate level|initial state checksum|newton_converged|lin_it_total"
    echo
}

BASE="OMP_NUM_THREADS=1 SFEM_FGMRES=1 SFEM_ENABLE_OUTPUT=0 \
      SFEM_PARALLEL_MF_LEGACY=1 SFEM_ALLOW_MPI=1 \
      SFEM_CASE=cavity SFEM_N=4 SFEM_ELEMENT_REFINE_LEVEL=2 \
      SFEM_GMG=1 SFEM_GMG_GALERKIN=2 SFEM_SMOOTHER=vanka \
      SFEM_GMG_STATE_SUM=1"

for n in 1 2 4; do
    run_arm "state" "$n" "${BASE}"
done

echo "raw logs: $RAW"
