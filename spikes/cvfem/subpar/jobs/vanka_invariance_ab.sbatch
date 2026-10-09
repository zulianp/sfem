#!/usr/bin/env bash
#SBATCH --account=c40
#SBATCH --job-name=cvfem-vanka-ab
#SBATCH --partition=debug
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=72
#SBATCH --time=00:29:00
#SBATCH --exclusive --hint=nomultithread
#SBATCH --uenv=prgenv-gnu/24.11:v2
#SBATCH --view=default
#SBATCH --output=vankaab_%j.out
#SBATCH --error=vankaab_%j.err
#
# IS THE ARENA CONVERSION INVARIANT ON GRACE?
#
# The one commit this session that touched the Vanka smoother and the Galerkin coarse operator is
# e4e530a90, which moved thirteen per-thread std::vector scratch buffers into arena slots 9-13. A
# refactor that only changes where a buffer comes from must leave every number the solver computes
# identical, so this runs the same cases with both binaries and compares the iteration counts, the
# Vanka-against-BSR gate and the residual history. NOT the timings: one pass cannot separate them,
# and this job is about invariance, not throughput -- that was settled by jobs/ab_refactor.sbatch.
#
#   new  the tree under test
#   ref  fb86be26a, the commit before the arena conversion
#
# Two cases, deliberately:
#
#   nozzle  for the gate arm only, because SFEM_VANKA_CHECK -- the Vanka stencil application
#           against a BSR built for the same state -- is what jobs/vanka_default_P28.sbatch runs
#           and the one assertion in this area that makes a claim.
#   cavity  for the cycle arms, because the nozzle with multigrid is a NINE-STAGE Reynolds
#           continuation at roughly 35 s a stage: the production gate's 300 s with_mg arm times
#           out at stage 8 and can never pass, and there is no env var to cut the staging. The
#           cavity reaches a converged answer in seconds through the same Vanka smoother and the
#           same Galerkin coarse operator, which is what invariance needs.
set -u
NEW=/ritom/scratch/cscs/zulianp/cvfem-refactor/build-newsfem/cvfem_hex8_ns_ssgmg
REF=/ritom/scratch/cscs/zulianp/cvfem-ref/build-newsfem/cvfem_hex8_ns_ssgmg
OUT=$SCRATCH/cvfem_vanka_ab/$SLURM_JOB_ID; mkdir -p "$OUT"
export OMP_NUM_THREADS=72 OMP_PROC_BIND=true OMP_PLACES=cores
NOZ=(SFEM_CASE=nozzle SFEM_NOZZLE_RE=500 SFEM_GEOM=affine SFEM_NOZZLE_NCORE=2
     SFEM_NOZZLE_NBORE=1 SFEM_NOZZLE_NOUTER=1 "SFEM_NOZZLE_NAXIAL=4 3 8 12")
CAV=(SFEM_CASE=cavity SFEM_NX=8 SFEM_NY=8 SFEM_NZ=8 SFEM_GEOM=affine)

# The case environment is an ARRAY, not a string. SFEM_NOZZLE_NAXIAL's value is "4 3 8 12" --
# four words -- and a value with spaces does not survive an unquoted expansion: `env` took the
# second word as the command it was asked to run and every arm exited 127.
run() {   # tag binary -- case-env... -- extra-env...
    local tag=$1 bin=$2; shift 2
    local -a e=()
    while [ "$#" -gt 0 ] && [ "$1" != "--" ]; do e+=("$1"); shift; done
    shift || true
    mkdir -p "$OUT/$tag" && cd "$OUT/$tag"
    stdbuf -oL -eL timeout 420 env "${e[@]}" SFEM_ELEMENT_REFINE_LEVEL=4 SFEM_FGMRES=1 \
        "$@" "$bin" out > out.log 2>&1
    echo "--- $tag exit $?"
}

for side in ref new; do
    b=$REF; [ "$side" = new ] && b=$NEW
    run "${side}_gate" "$b" "${NOZ[@]}" -- SFEM_GMG=1 SFEM_NL_MAX_IT=3 SFEM_VANKA_CHECK=1
    run "${side}_mg"   "$b" "${CAV[@]}" -- SFEM_GMG=1 SFEM_NL_MAX_IT=8
    run "${side}_nomg" "$b" "${CAV[@]}" -- SFEM_GMG=0 SFEM_NL_MAX_IT=8
done

echo
echo "=== the Vanka-against-BSR gate"
for side in ref new; do
    printf '  %-4s ' "$side"
    grep -h "vanka stencil vs bsr" "$OUT/${side}_gate/out.log" 2>/dev/null || echo "(absent)"
done

echo
echo "=== what the solver computed, which must be IDENTICAL"
for arm in gate mg nomg; do
    echo "  -- $arm"
    for side in ref new; do
        printf '     %-4s ' "$side"
        grep -hoE "ndof: [0-9]+|newton_converged: [0-9]+|newton_total: [0-9]+|lin_it_total: [0-9]+" \
             "$OUT/${side}_${arm}/out.log" 2>/dev/null | tr '\n' ' '
        grep -hE "^newton " "$OUT/${side}_${arm}/out.log" 2>/dev/null | tail -1
    done
done

echo
echo "=== every reported scalar, diffed; timings excluded because one pass cannot separate them"
for arm in gate mg nomg; do
    for side in ref new; do
        grep -ohE "[a-z_]+: *[-0-9.e+]+" "$OUT/${side}_${arm}/out.log" 2>/dev/null \
            | grep -vE "^(t_|us_per|_ns_|t_solve|t_precond|t_operator)" > "$OUT/$side.$arm.scalars"
    done
    n=$(wc -l < "$OUT/ref.$arm.scalars")
    m=$(wc -l < "$OUT/new.$arm.scalars")
    if [ "$n" -lt 10 ] || [ "$m" -lt 10 ]; then
        # Two empty files compare equal. Saying so would be claiming agreement about nothing,
        # which is what this job printed the first time it ran, with every arm at exit 127.
        echo "  $arm: NO RESULT -- ref produced $n scalars, new $m; the arms did not run"
        fail=1
    elif diff -q "$OUT/ref.$arm.scalars" "$OUT/new.$arm.scalars" >/dev/null 2>&1; then
        echo "  $arm: IDENTICAL ($n scalars)"
    else
        echo "  $arm: DIFFERS"
        diff "$OUT/ref.$arm.scalars" "$OUT/new.$arm.scalars" | head -20
        fail=1
    fi
done
if [ "${fail:-0}" = 1 ]; then
    echo
    echo "FAILED: the arena conversion is not invariant, or the arms did not produce a result."
else
    echo
    echo "PASSED: every scalar the solver reports is identical across the arena conversion."
fi
echo "=== DONE $(date)"
