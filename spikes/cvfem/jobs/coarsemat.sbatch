#!/usr/bin/env bash
#SBATCH --account=c40
#SBATCH --job-name=cvfem-coarsemat
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=288
#SBATCH --time=00:20:00
#SBATCH --exclusive
#SBATCH --hint=nomultithread
#SBATCH --uenv-passthrough=ignore
#SBATCH --output=coarsemat_%j.out
#SBATCH --error=coarsemat_%j.err

# IS THE MATRIX HANDED TO THE COARSE FACTORISATION COMPLETE?
#
# Where this stands. On one Newton step off the pre-Newton state the first linear solve takes
#
#     n=1  5     n=2  5     n=3  5     n=4  5
#     n=5 54     n=6 210    n=7 50     n=8 18
#
# Flat through four ranks, then erratic -- not a threshold and not a trend, with the worst
# case at six ranks rather than eight. With 64 macro elements in a 4x4x4 lattice a contiguous
# partition into one to four ranks lands on whole 16-element layers, and from five ranks on
# the cut falls inside a layer and the subdomains stop being slabs. The magnitudes then track
# how awkward each particular shape is, which is the signature of a partition-geometry defect
# rather than a rank-count one.
#
# Everything else in the two-level correction measures INVARIANT at every one of those rank
# counts, including five, six and seven where the cycle is up to forty times worse:
#
#   coarse block diagonal   258.95110245728023 at n = 1..8, every printed figure
#   coarse operator action  abs 0.98625725178447954 +/- 1 ulp at n = 1..8
#   transfer adjoint        <Rx,y> = <x,Py> = -1.633495e+02, ratio 1.000000, at 1, 4 and 8
#   coarse LU null space    1 of 500 dropped, component p, at every rank count
#   no coarse correction    13 linear iterations at 1, 2, 4 and 8 -- flat
#
# The one object that does move is the coarse SOLVE. Its owned-range checksum agrees to
# thirteen figures at one, two and four ranks and moves in the FOURTH at eight:
#
#     n=1 -8650463.8508590795    n=2 -8650463.8508593775
#     n=4 -8650463.8508594036    n=8 -8644689.2816008534
#
# Reassociation does not explain that, because two and four ranks reduce over different
# orderings as well and stay at thirteen figures.
#
# THE HYPOTHESIS THIS TESTS. The operator's ACTION and the matrix the factorisation is FED are
# two different objects. The action comes from element matrices applied through the local
# reduce table over aura elements, which is complete. The dense LU is assembled from
# g.Amat[level], a BSR whose sparsity comes from the LOCAL coarse graph, taking owned rows
# only and trusting each to be complete. A row missing a column it has globally is silently
# zero in the dense matrix, every rank agrees on the same wrong matrix, and the result is
# consistent across ranks and wrong against serial -- which only a 1-versus-N comparison
# catches, and which is exactly the pattern above.
#
# READ, committed before the run. coarsemat is keyed by GLOBAL row and column and counts
# blocks as well as summing them, because a row short of a column is the failure being hunted
# and it moves the count while barely moving the sum:
#
#   blocks and sums flat 1..8   -> the assembled matrix is complete and the fault is inside
#                                  the factorisation or its gather, not in what it is fed
#   blocks drop from n=5        -> owned coarse rows are losing columns once the cut stops
#                                  falling on layer boundaries; the assembly owes the missing
#                                  entries, and the fix is a cross-rank completion of the
#                                  coarse sparsity rather than anything in the solver
#
# A TIMEOUT is unmeasured, not a result. One Newton step is about a second.

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
OUT="${OUT:-$SCRATCH/cvfem_coarsemat/${SLURM_JOB_ID:-manual}}"
RAW="$OUT/raw"
mkdir -p "$RAW"
[ -x "$BIN" ] || { echo "error: no binary at $BIN" >&2; exit 1; }

ARM_TIMEOUT="${ARM_TIMEOUT:-120}"

run_arm() {
    local tag="$1" nranks="$2"; shift 2
    echo "### ${tag} ranks=${nranks} ###"
    timeout --signal=TERM --kill-after=30 "${ARM_TIMEOUT}" \
        srun -u --ntasks="${nranks}" --cpus-per-task=1 --kill-on-bad-exit=1 \
            --time="00:$(printf '%02d' $(( (ARM_TIMEOUT + 59) / 60 ))):00" \
            --uenv="$UENV_IMAGE" --view="$UENV_VIEW" \
            bash -lc "$* stdbuf -oL ${BIN} ${OUT}/${tag}_n${nranks}" 2>&1 \
        | tee "$RAW/${tag}_n${nranks}.log" \
        | stdbuf -oL grep -aE "^coarsemat|^solvesum |^applysum |^diagsum |lin_it_total|newton_converged"
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
MG="SFEM_GMG=1 SFEM_GMG_GALERKIN=2 SFEM_SMOOTHER=bjacobi"
SUMS="SFEM_GMG_COARSE_MAT_SUM=1 SFEM_GMG_SOLVE_OWNED_SUM=1 \
      SFEM_GMG_APPLY_OWNED_SUM=1 SFEM_GMG_DIAG_OWNED_SUM=1"
ONESTEP="SFEM_NL_MAX_IT=1 SFEM_VERIFY_TOL=1e9"

# Every rank count across the transition, since the break begins at five and peaks at six.
for n in 1 2 3 4 5 6 7 8; do
    run_arm "cmat" "$n" "${COMMON} ${CAVITY} ${MG} ${SUMS} ${ONESTEP}"
done

echo "raw logs: $RAW"
