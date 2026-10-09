#!/usr/bin/env bash
#SBATCH --account=c40
#SBATCH --job-name=cvfem-tol-ab
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=288
#SBATCH --time=00:25:00
#SBATCH --exclusive
#SBATCH --hint=nomultithread
#SBATCH --uenv-passthrough=ignore
#SBATCH --output=tol_ab_%j.out
#SBATCH --error=tol_ab_%j.err

# IS THE "DECOMPOSITION DEPENDENCE" A TOLERANCE ARTIFACT?
#
# The ladder (job 4714993) established, on cavity L2 single-threaded:
#
#   * FGMRES is decomposition invariant. The diagnostic reports distributed 1 with the
#     correct owned and allocated sizes at every rank count, and the Arnoldi basis stays
#     orthogonal to rounding: orth_worst 1.2e-14 .. 1.05e-13 at two ranks against
#     1.96e-14 .. 2.99e-13 at one. All four reductions are collective.
#   * Newton is decomposition invariant. Same step count per stage, and the residual
#     sequences agree to three digits (stage 4/5: 3.672459e-03, 1.360979e-04, 4.663373e-07,
#     4.473636e-10, 4.330198e-13 at one rank against 3.672459e-03, 1.361101e-04,
#     4.692717e-07, 4.212043e-10, 1.914288e-13 at two).
#   * The early linear solves are invariant too: 30 35 80 33 41 58 42 36 45 59 at one rank
#     against 30 36 82 33 41 58 42 37 44 59 at two.
#
# What is NOT invariant is a handful of late solves that run to the iteration cap. EVERY
# capped solve measured sits immediately after a Newton residual that is already tiny, and
# produces an update of no consequence:
#
#     ||R|| before the solve      lin_it    |dx|_inf      arm
#     4.212043e-10                  4000    8.552800e-09  no multigrid, 2 ranks
#     2.203986e-07                  4000    7.829093e-05  mg + jacobi, 2 ranks
#     9.005484e-10                  4000    1.506874e-06  mg + jacobi, 2 ranks
#     3.874130e-09                  4000    6.793245e-07  mg + jacobi, 2 ranks
#     4.391497e-10                  4000    3.429548e-08  mg + jacobi, 2 ranks
#     6.053453e-08                  4000    8.664659e-07  mg + jacobi, 2 ranks
#     1.645272e-10                  4000    7.403026e-09  mg + jacobi, 2 ranks
#
# Not one has a healthy right-hand side; the largest is 2.2e-07 and most are below 1e-09.
# With SFEM_LSOLVE_RTOL=1e-3 against a bnorm that small, the relative branch of the test at
# cvfem_fgmres.hpp:291 (resid / bnorm < rtol_ || resid < atol_) asks for a residual below
# the absolute floor atol_=1e-14, so neither branch can fire and the solve burns its whole
# cap on a right-hand side that is numerical noise.
#
# The productive solves, by contrast, agree closely across decompositions -- 30/30, 33/33,
# 41/41, 58/58, 42/42, 59/59, 56/56, 53/53 at one rank against two -- so what differs
# between decompositions is only how many degenerate tail solves each one happens to request.
#
# THE TEST: raise SFEM_LSOLVE_ATOL so a noise-level right-hand side is recognised as solved.
#
#   if the capped solves collapse and the counts become rank-invariant
#        -> the degradation is a tolerance policy artifact, not a parallel defect
#   if the caps survive
#        -> there is a genuine operator-level decomposition dependence and the next layer
#           down (the matvec itself) is where to look
#
# LOOSENING A TOLERANCE CAN MASK A REAL FAILURE by declaring victory sooner, so iteration
# counts alone cannot settle this. Every arm therefore also reports newton_converged and the
# full ||R|| sequence: the atol arms must reach the SAME final residuals in the SAME number
# of Newton steps as the baseline. An arm that converges in fewer Newton steps, or to a
# worse residual, has been broken by the knob rather than fixed by it.
#
# THE PURE-MPI ARM runs on cavity N=8 (512 macro elements), not Poiseuille L4. At 288 ranks
# the previous attempt aborted with MPI_Abort on every rank because Poiseuille L4 has only
# 108 macro elements -- about 180 ranks would own none. That abort was a degenerate
# partition being refused, not an algorithmic result, and reporting it as one would be
# wrong. 512 elements over 288 ranks is a bit under two elements per rank: still thin, but
# every rank owns work.
#
# SINGLE-THREADED throughout: pure MPI removes atomics and thread scheduling as
# explanations, which is the point of the 288-rank arm.

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
OUT="${OUT:-$SCRATCH/cvfem_tol_ab/${SLURM_JOB_ID:-manual}}"
RAW="$OUT/raw"
mkdir -p "$RAW"
[ -x "$BIN" ] || { echo "error: no binary at $BIN" >&2; exit 1; }

run_arm() {
    local tag="$1" nranks="$2"; shift 2
    echo "### ${tag} ranks=${nranks} ###"
    srun -u --ntasks="${nranks}" --cpus-per-task=1 --uenv="$UENV_IMAGE" --view="$UENV_VIEW" \
        bash -lc "$* stdbuf -oL ${BIN} ${OUT}/${tag}_n${nranks}" 2>&1 \
        | tee "$RAW/${tag}_n${nranks}.log" \
        | stdbuf -oL grep -aE "^fgmres:|^newton |newton_converged|lin_it_total|lin_it:|nelements"
    echo
}

COMMON="OMP_NUM_THREADS=1 SFEM_FGMRES=1 SFEM_FGMRES_DIAG=1 SFEM_ENABLE_OUTPUT=0 \
        SFEM_PARALLEL_MF_LEGACY=1 SFEM_ALLOW_MPI=1 SFEM_LSOLVE_MAX_IT=4000"

# The same, on the matvec path the library actually uses by default.
#
# SFEM_PARALLEL_MF_LEGACY defaults to 0, and every multi-rank measurement in this campaign
# has forced it to 1 -- so the phased path has never been measured here at all. Legacy is a
# blocking gather followed by one full ALL apply; the default splits the apply into
# OWNED_NOT_SHARED and SHARED_AND_AURA around a non-blocking gather, with thread 0 as the
# communication thread. At OMP_NUM_THREADS=1 that thread split cannot engage, which is
# precisely why this arm belongs in a single-threaded job: it isolates the SCOPED apply from
# the threaded overlap, and any difference it shows is about element scoping alone.
COMMON_PHASED="OMP_NUM_THREADS=1 SFEM_FGMRES=1 SFEM_FGMRES_DIAG=1 SFEM_ENABLE_OUTPUT=0 \
        SFEM_PARALLEL_MF_LEGACY=0 SFEM_ALLOW_MPI=1 SFEM_LSOLVE_MAX_IT=4000"
CAVITY="SFEM_CASE=cavity SFEM_N=4 SFEM_ELEMENT_REFINE_LEVEL=2"
MG="SFEM_GMG=1 SFEM_GMG_GALERKIN=2 SFEM_SMOOTHER=bjacobi"

# ---- Baseline vs raised atol, no multigrid.
for n in 1 2 4; do
    run_arm "base_nomg" "$n" "${COMMON} ${CAVITY} SFEM_GMG=0 SFEM_SMOOTHER=bjacobi"
    run_arm "atol_nomg" "$n" \
        "${COMMON} ${CAVITY} SFEM_GMG=0 SFEM_SMOOTHER=bjacobi SFEM_LSOLVE_ATOL=1e-12"
done

# ---- Same pair with multigrid + Jacobi, where the caps were most frequent.
for n in 1 2 4; do
    run_arm "base_mgjac" "$n" "${COMMON} ${CAVITY} ${MG}"
    run_arm "atol_mgjac" "$n" "${COMMON} ${CAVITY} ${MG} SFEM_LSOLVE_ATOL=1e-12"
done

# ---- The confound nobody has closed: the default matvec path, legacy off.
for n in 1 2 4; do
    run_arm "phased_nomg"  "$n" "${COMMON_PHASED} ${CAVITY} SFEM_GMG=0 SFEM_SMOOTHER=bjacobi"
    run_arm "phased_mgjac" "$n" "${COMMON_PHASED} ${CAVITY} ${MG}"
done

# ---- Pure MPI at full node width, on a case with enough elements to partition.
BIGCAV="SFEM_CASE=cavity SFEM_N=8 SFEM_ELEMENT_REFINE_LEVEL=2"
for n in 1 288; do
    run_arm "wide_mgjac" "$n" "${COMMON} ${BIGCAV} ${MG} SFEM_LSOLVE_ATOL=1e-12"
done

echo "raw logs: $RAW"
