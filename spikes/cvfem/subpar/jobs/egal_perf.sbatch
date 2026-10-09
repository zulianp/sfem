#!/usr/bin/env bash
#SBATCH --account=c40
#SBATCH --job-name=cvfem-egal-perf
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --time=00:25:00
#SBATCH --exclusive
#SBATCH --hint=nomultithread
#SBATCH --uenv-passthrough=ignore
#SBATCH --output=egal_perf_%j.out
#SBATCH --error=egal_perf_%j.err

# Where does a V-cycle actually spend its time?
#
# The workload is preconditioner applications and nothing else: one Newton step, with the
# Krylov solve capped so the run is a fixed number of V-cycles. Setup still happens, so
# the profile is read with that in mind -- the driver's own phase report is printed
# alongside to separate the two.
#
# Profiled at 108 macro-elements, L=8, 242,500 dofs on 72 Grace cores, which is the
# saturating configuration; a profile from an under-filled machine would mostly show
# whatever the idle threads do.
#
# The measured binary is profiled as built (-O3 -DNDEBUG). Adding -g or frame pointers
# for nicer call graphs would risk profiling something other than what actually runs.

set -u
# Locate the spike root -- the directory holding CMakeLists.txt, drivers/ and scripts/.
# Resolved rather than hardcoded, so the job does not carry one user's $SCRATCH layout.
#
# BASH_SOURCE is NOT usable on its own here. Slurm copies the batch script to
# /var/spool/slurmd/job<id>/slurm_script before running it, so under sbatch the script's
# own path says nothing about where the tree is -- it resolves to /var/spool/slurmd and
# every relative path after it fails. SLURM_SUBMIT_DIR, the directory sbatch was invoked
# from, is the reliable one; the rest are for running the script directly.
_cvfem_probe=scripts/alps_env.sh
SPIKE_ROOT=""
for _cand in "${SLURM_SUBMIT_DIR:-}" "${SLURM_SUBMIT_DIR:-}/.." "${CVFEM_DIR:-}" \
             "$(dirname "${BASH_SOURCE[0]}")/.."; do
    [ -n "${_cand}" ] && [ -f "${_cand}/${_cvfem_probe}" ] || continue
    SPIKE_ROOT="$(cd "${_cand}" && pwd)"; break
done
[ -n "${SPIKE_ROOT}" ] || {
    echo "error: could not locate spikes/cvfem; sbatch from there or set CVFEM_DIR" >&2; exit 1; }
cd "$SPIKE_ROOT"

UENV_IMAGE="prgenv-gnu/24.11:v2"
UENV_VIEW="default"
BIN="./build_alps/cvfem_hex8_ns_ssgmg"
OUT="$SCRATCH/egal_out_perf"
DATA="$SCRATCH/perf_egal.data"
mkdir -p "$OUT"

export OMP_PROC_BIND=true
export OMP_PLACES=cores

CYCLES="${CYCLES:-100}"

echo "### perf: ${CYCLES} V-cycles, 108 macro-elements, L=8, 242500 dofs, 72 threads ###"
srun -u --ntasks=1 --cpus-per-task=72 --uenv="$UENV_IMAGE" --view="$UENV_VIEW" \
    bash -lc "OMP_NUM_THREADS=72 OMP_PROC_BIND=true OMP_PLACES=cores \
              SFEM_CASE=poiseuille SFEM_NX=12 SFEM_NY=3 SFEM_NZ=3 \
              SFEM_ELEMENT_REFINE_LEVEL=8 SFEM_GMG=1 SFEM_GMG_GALERKIN=2 \
              SFEM_GMG_EGAL=1 SFEM_NL_MAX_IT=1 SFEM_LSOLVE_MAX_IT=${CYCLES} \
              SFEM_VERIFY_TOL=1e9 \
              perf record -F 499 --no-inherit -o ${DATA} -- ${BIN} ${OUT}" \
    2>&1 | grep -vE "^\[ perf record|Warning|bind: warning"

echo
echo "=== perf stat: is the cycle memory-bound? ==="
srun -u --ntasks=1 --cpus-per-task=72 --uenv="$UENV_IMAGE" --view="$UENV_VIEW" \
    bash -lc "OMP_NUM_THREADS=72 OMP_PROC_BIND=true OMP_PLACES=cores \
              SFEM_CASE=poiseuille SFEM_NX=12 SFEM_NY=3 SFEM_NZ=3 \
              SFEM_ELEMENT_REFINE_LEVEL=8 SFEM_GMG=1 SFEM_GMG_GALERKIN=2 \
              SFEM_GMG_EGAL=1 SFEM_NL_MAX_IT=1 SFEM_LSOLVE_MAX_IT=${CYCLES} \
              SFEM_VERIFY_TOL=1e9 \
              perf stat -e instructions,cycles,cache-references,cache-misses,stalled-cycles-backend \
                        stdbuf -oL ${BIN} ${OUT}" \
    2>&1 | grep -E "instructions|cycles|cache|stalled|elapsed|insn per" 

echo
echo "=== flat profile (self time) ==="
srun -u --ntasks=1 --cpus-per-task=8 --uenv="$UENV_IMAGE" --view="$UENV_VIEW" \
    bash -lc "perf report -i ${DATA} --stdio --no-children --percent-limit 0.25 2>/dev/null \
              | grep -vE '^#|^$' | head -34"
