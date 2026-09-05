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
cd "$SCRATCH/sfem/spikes/cvfem"

UENV_IMAGE="prgenv-gnu/24.11:v2"
UENV_VIEW="default"
BIN="./build_alps/cvfem_hex8_ns_ssgmg"
OUT="$SCRATCH/egal_out_perf"
DATA="$SCRATCH/perf_egal.data"
mkdir -p "$OUT"

export OMP_PROC_BIND=close
export OMP_PLACES=cores

CYCLES="${CYCLES:-100}"

echo "### perf: ${CYCLES} V-cycles, 108 macro-elements, L=8, 242500 dofs, 72 threads ###"
srun -u --ntasks=1 --cpus-per-task=72 --uenv="$UENV_IMAGE" --view="$UENV_VIEW" \
    bash -lc "OMP_NUM_THREADS=72 OMP_PROC_BIND=close OMP_PLACES=cores \
              SFEM_CASE=poiseuille SFEM_NX=12 SFEM_NY=3 SFEM_NZ=3 \
              SFEM_ELEMENT_REFINE_LEVEL=8 SFEM_GMG=1 SFEM_GMG_GALERKIN=2 \
              SFEM_GMG_EGAL=1 SFEM_NL_MAX_IT=1 SFEM_LSOLVE_MAX_IT=${CYCLES} \
              SFEM_VERIFY_TOL=1e9 \
              perf record -F 499 --no-inherit -o ${DATA} -- ${BIN} ${OUT}" \
    2>&1 | grep -vE "^\[ perf record|Warning|bind: warning"

echo
echo "=== perf stat: is the cycle memory-bound? ==="
srun -u --ntasks=1 --cpus-per-task=72 --uenv="$UENV_IMAGE" --view="$UENV_VIEW" \
    bash -lc "OMP_NUM_THREADS=72 OMP_PROC_BIND=close OMP_PLACES=cores \
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
