#!/usr/bin/env bash
#SBATCH --account=c40
#SBATCH --job-name=cvfem-shapeleg
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=288
#SBATCH --time=00:29:00
#SBATCH --exclusive
#SBATCH --hint=nomultithread
#SBATCH --uenv-passthrough=ignore
#SBATCH --output=shapeleg_%j.out
#SBATCH --error=shapeleg_%j.err

# THE COMMUNICATION-OVERLAP COST AT EACH SHAPE -- THE HALF I HAVE NEVER MEASURED.
#
# SFEM_PARALLEL_MF_LEGACY=1 selects a BLOCKING gather followed by a full ALL apply, with no
# ElementScope overlap. The library default is 0: gather_begin, then an OpenMP-parallel region
# that computes the owned-not-shared elements while the halo is in flight.
#
# Every measurement in this workstream -- correctness AND both shape sweeps -- was taken with
# LEGACY=1. It came from the plan's first correctness target and was then carried into every
# job unexamined. It is the wrong path for a throughput statement, and it interacts with shape
# precisely because the overlap path spends a thread on communication:
#
#     4x72  -> one thread of 72 is 1.4%
#     72x4  -> one thread of 4 is 25%
#
# So the overlap could sharpen the preference for fat ranks, or move the optimum. Either way
# the shape verdict from job 4720450 (4x72 fastest, monotone) describes the LEGACY path only.
#
# WHERE TO LOOK. In that sweep coarse_factor was 57-92% of the run and is untouched by this
# flag, so `total` will barely move. The discriminator is t_solve, which is where the matvec
# and therefore the overlap actually lives: it ran 1.290 / 1.326 / 1.530 / 1.727 s at
# 4x72 / 8x36 / 18x16 / 72x4 under LEGACY=1.
#
# READ, committed before the run:
#
#   overlap faster, and more so at lean shapes -> overlap pays and partly offsets the shape
#                                                 penalty; the optimum may flatten
#   overlap faster only at fat shapes          -> the comm thread costs more than it hides once
#                                                 threads are few; fat ranks win twice over
#   overlap SLOWER everywhere                  -> the overlap path is not paying on this
#                                                 operator, and LEGACY=1 is the right default
#                                                 for this spike -- worth knowing either way
#   lin_it differs between the two paths       -> they are not computing the same thing; that is
#                                                 a correctness defect and outranks every timing
#
# Same case and same coarse solver as job 4720450, so the LEGACY=1 rows are directly comparable
# with it. Timings carry the dof count; nnodes: lines in raw logs are PER-RANK locals.

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

UENV_IMAGE="prgenv-gnu/24.11:v2"; UENV_VIEW="default"
BIN="${CVFEM_BIN:-./build/cvfem_hex8_ns_ssgmg}"
OUT="${OUT:-$SCRATCH/cvfem_shapeleg/${SLURM_JOB_ID:-manual}}"; RAW="$OUT/raw"; mkdir -p "$RAW"
[ -x "$BIN" ] || { echo "error: no binary at $BIN" >&2; exit 1; }
ARM_TIMEOUT="${ARM_TIMEOUT:-220}"

COMMON="SFEM_FGMRES=1 SFEM_ENABLE_OUTPUT=0 SFEM_ALLOW_MPI=1 \
        SFEM_LSOLVE_MAX_IT=4000 SFEM_LSOLVE_RTOL=1e-4 SFEM_LSOLVE_ATOL=1e-14"
CASE="SFEM_CASE=cavity SFEM_N=16 SFEM_ELEMENT_REFINE_LEVEL=4"
ONESTEP="SFEM_NL_MAX_IT=1 SFEM_VERIFY_TOL=1e9"

echo "cavity N=16 refine 4 -- 4096 macro elements, 274,625 nodes, 1,098,500 dof, 3 levels."
echo "MG + Vanka, GALERKIN=2, dense LU coarse (as job 4720450). Fixed 288 cores."
echo "legacy=1 is blocking gather + full apply; legacy=0 is the overlap path (library default)."
echo
printf "  %-9s %-8s %-10s %-12s %-12s %s\n" "shape" "legacy" "lin_it" "total (s)" "t_solve" "coarse_fac"
for shape in "4 72" "8 36" "18 16" "72 4"; do
    set -- $shape; n=$1; t=$2
    for lg in 1 0; do
        tag="s${n}x${t}_L${lg}"
        timeout --signal=TERM --kill-after=30 "${ARM_TIMEOUT}" \
            srun -u --ntasks="${n}" --cpus-per-task="${t}" --kill-on-bad-exit=1 \
                --uenv="$UENV_IMAGE" --view="$UENV_VIEW" \
                bash -lc "OMP_NUM_THREADS=${t} OMP_PROC_BIND=true SFEM_PARALLEL_MF_LEGACY=${lg} \
                          ${COMMON} ${CASE} ${ONESTEP} SFEM_GMG=1 SFEM_GMG_GALERKIN=2 \
                          SFEM_SMOOTHER=vanka stdbuf -oL ${BIN} ${OUT}/${tag}" \
            > "$RAW/${tag}.log" 2>&1
        rc=$?
        it=$(grep -a lin_it_total "$RAW/${tag}.log" | head -1 | sed 's/.*lin_it_total: //')
        sec=$(grep -a "ns_ssgmg: .* seconds" "$RAW/${tag}.log" | head -1 | sed 's/.*ssgmg: \([0-9.]*\).*/\1/')
        ts=$(grep -am1 "t_solve" "$RAW/${tag}.log" | sed 's/.*t_solve: \([0-9.]*\).*/\1/')
        cf=$(grep -am1 "coarse_factor" "$RAW/${tag}.log" | awk '{print $2}')
        [ "$rc" = "124" ] && it="TIMEOUT"; [ "$it" = "4000" ] && it="4000(cap)"
        grep -aq "oom_kill" "$RAW/${tag}.log" && it="OOM-KILLED"
        printf "  %-9s %-8s %-10s %-12s %-12s %s\n" "${n}x${t}" "$lg" "${it:-NONE}" \
            "${sec:-n/a}" "${ts:-n/a}" "${cf:-n/a}"
    done
done
echo
echo "lin_it must match between legacy=1 and legacy=0 at every shape; a difference is a defect."
echo "global problem: 1,098,500 dof."
echo "raw logs: $RAW"
