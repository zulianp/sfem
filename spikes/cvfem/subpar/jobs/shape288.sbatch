#!/usr/bin/env bash
#SBATCH --account=c40
#SBATCH --job-name=cvfem-shape
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=288
#SBATCH --time=00:29:00
#SBATCH --exclusive
#SBATCH --hint=nomultithread
#SBATCH --uenv-passthrough=ignore
#SBATCH --output=shape288_%j.out
#SBATCH --error=shape288_%j.err

# RANK x THREAD SHAPE AT A FIXED 288 CORES -- WHERE IS THE SWEET SPOT?
#
# The baseline for one GH200 node is 4 MPI ranks x 72 OpenMP threads: four sockets of 72
# Neoverse V2 cores, one rank per socket, each thread team inside one socket's memory. This
# sweeps around that baseline at a constant 288 cores.
#
# EVERY TIMING IN THIS WORKSTREAM SO FAR WAS TAKEN AT THE WRONG SHAPE. The numbers were either
# 1 rank x 72 threads (the serial reference) or 288 ranks x 1 thread (the decomposition
# invariance instrument, which removes atomics and thread scheduling as explanations for a
# moving iteration count). Neither is what the machine is meant to run, so none of them is a
# throughput result. This job produces the first ones that are.
#
# WHAT THE SHAPE IS EXPECTED TO TRADE.
#
#   fewer, fatter ranks  -- the coarsest level is a dense LU that EVERY rank holds in full and
#                           assembles through an n^2 allreduce. At 1,098,500 dof the coarse
#                           level is 19,652 dofs = 3.09 GB per rank, so 4 ranks need 12.4 GB
#                           and 288 ranks need ~890 GB, which exceeded the node and was
#                           OOM-killed. Fewer ranks also shrink the allreduce: measured at a
#                           smaller coarse level, its cost went 5.2 s at 128 ranks to 54.9 s at
#                           288. And getrf is threaded, so fat ranks factorise faster.
#   more, leaner ranks   -- more parallelism in the matrix-free sweep and the smoother, which
#                           is the part that actually scaled: t_solve went 12.57 s -> 1.35 s
#                           from 1 to 128 ranks on the 740,772-dof case.
#
# So the sweet spot is a genuine trade rather than a monotone curve, which is why it is swept.
#
# 288x1 IS EXCLUDED, not forgotten: it OOMs on this case for the reason above. 144x2 is left
# out too, since 144 x 3.09 GB = 445 GB is over half the node and would measure memory pressure
# rather than shape.
#
# READ, committed before the run:
#
#   lin_it identical at every shape  -> the shapes are solving the same problem and the wall
#                                       clocks are comparable
#   lin_it moving                    -> a threading or decomposition defect the pure-MPI sweeps
#                                       did not expose, and that outranks any timing here
#   a minimum away from both ends    -> the real sweet spot; report it with the phase split so
#                                       it is clear WHAT it is trading
#   monotone toward 4x72             -> the coarse solve dominates everything and the shape
#                                       question is really a coarse-solver question
#
# Timings carry the dof count. A count at the cap did not converge and is a floor.

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
OUT="${OUT:-$SCRATCH/cvfem_shape/${SLURM_JOB_ID:-manual}}"
RAW="$OUT/raw"
mkdir -p "$RAW"
[ -x "$BIN" ] || { echo "error: no binary at $BIN" >&2; exit 1; }

ARM_TIMEOUT="${ARM_TIMEOUT:-220}"

COMMON="SFEM_FGMRES=1 SFEM_ENABLE_OUTPUT=0 SFEM_PARALLEL_MF_LEGACY=1 SFEM_ALLOW_MPI=1 \
        SFEM_LSOLVE_MAX_IT=4000 SFEM_LSOLVE_RTOL=1e-4 SFEM_LSOLVE_ATOL=1e-14"
CASE="SFEM_CASE=cavity SFEM_N=16 SFEM_ELEMENT_REFINE_LEVEL=4"
ONESTEP="SFEM_NL_MAX_IT=1 SFEM_VERIFY_TOL=1e9"

echo "cavity N=16 refine 4 -- 4096 macro elements, 274,625 nodes, 1,098,500 dof, 3 levels."
echo "MG + Vanka, GALERKIN=2. Fixed 288 cores; OMP_PROC_BIND=true throughout."
echo "Baseline is the first row: 4 ranks x 72 threads, one rank per socket."
echo
printf "  %-10s %-8s %-9s %-11s %-12s %-12s %s\n" \
       "shape" "ranks" "threads" "lin_it" "total (s)" "coarse_fac" "t_solve"
for shape in "4 72" "8 36" "12 24" "18 16" "36 8" "72 4"; do
    set -- $shape
    n=$1; t=$2
    tag="s${n}x${t}"
    timeout --signal=TERM --kill-after=30 "${ARM_TIMEOUT}" \
        srun -u --ntasks="${n}" --cpus-per-task="${t}" --kill-on-bad-exit=1 \
            --uenv="$UENV_IMAGE" --view="$UENV_VIEW" \
            bash -lc "OMP_NUM_THREADS=${t} OMP_PROC_BIND=true ${COMMON} ${CASE} ${ONESTEP} \
                      SFEM_GMG=1 SFEM_GMG_GALERKIN=2 SFEM_SMOOTHER=vanka \
                      stdbuf -oL ${BIN} ${OUT}/${tag}" \
        > "$RAW/${tag}.log" 2>&1
    rc=$?
    it=$(grep -a lin_it_total "$RAW/${tag}.log" | head -1 | sed 's/.*lin_it_total: //')
    sec=$(grep -a "ns_ssgmg: .* seconds" "$RAW/${tag}.log" | head -1 | sed 's/.*ssgmg: \([0-9.]*\).*/\1/')
    cf=$(grep -am1 "coarse_factor" "$RAW/${tag}.log" | awk '{print $2}')
    ts=$(grep -am1 "t_solve" "$RAW/${tag}.log" | sed 's/.*t_solve: \([0-9.]*\).*/\1/')
    [ "$rc" = "124" ] && it="TIMEOUT"
    [ "$it" = "4000" ] && it="4000(cap)"
    grep -aq "oom_kill" "$RAW/${tag}.log" && it="OOM-KILLED"
    printf "  %-10s %-8s %-9s %-11s %-12s %-12s %s\n" \
        "${n}x${t}" "$n" "$t" "${it:-NONE}" "${sec:-n/a}" "${cf:-n/a}" "${ts:-n/a}"
done
echo
echo "288x1 excluded: 288 x 3.09 GB of replicated dense coarse factor exceeds the node."
grep -ah "^nnodes:" "$RAW"/s4x72.log | head -1
grep -ah "^coarse LU:" "$RAW"/s4x72.log | head -1
echo "raw logs: $RAW"
