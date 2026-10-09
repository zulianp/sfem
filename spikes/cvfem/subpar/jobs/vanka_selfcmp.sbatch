#!/usr/bin/env bash
#SBATCH --account=c40
#SBATCH --job-name=cvfem-vanka-selfcmp
#SBATCH --partition=debug
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=72
#SBATCH --time=00:20:00
#SBATCH --exclusive --hint=nomultithread
#SBATCH --uenv=prgenv-gnu/24.11:v2
#SBATCH --view=default
#SBATCH --output=vankaself_%j.out
#SBATCH --error=vankaself_%j.err
#
# IS THE SOLVER BITWISE REPRODUCIBLE AGAINST ITSELF?
#
# Job 4986218 compared the arena conversion against the commit before it and found every
# iteration count, every convergence flag and every final residual identical -- and ONE derived
# scalar different: `newton 4 rel: 9.436903e-12` against `9.436901e-12`, at the end of
# continuation stage 4 of 5, where ||R|| is 9.2e-14 and printed the same on both sides. Only the
# ratio moved, so the denominator differed below print precision.
#
# A difference that small can be the allocation (the arena hands out different alignment than a
# fresh std::vector, which changes how a reduction groups) or it can be the solver not being
# bitwise reproducible against ITSELF, in which case it has nothing to do with the arena at all.
# A baseline comparison cannot tell those apart -- it can be satisfied by luck. Running one
# binary twice can.
#
# THE ANSWER, job 4986232: the same binary disagrees with itself at that same line, ref against
# ref reading 9.436902e-12 against 9.436901e-12 and new against new 9.436901e-12 against
# 9.436903e-12. Line 74 is the only line that ever differs, in all four comparisons, out of 94
# scalars. So that scalar is not reproducible run to run whatever binary produces it, the arena
# conversion is invariant, and this solver is NOT bitwise reproducible at one derived ratio --
# which is a fact about the driver's reductions, not about the packed layouts, whose
# atomics-free determinism is gated separately.
set -u
NEW=/ritom/scratch/cscs/zulianp/cvfem-refactor/build-newsfem/cvfem_hex8_ns_ssgmg
REF=/ritom/scratch/cscs/zulianp/cvfem-ref/build-newsfem/cvfem_hex8_ns_ssgmg
OUT=$SCRATCH/cvfem_vanka_self/$SLURM_JOB_ID; mkdir -p "$OUT"
export OMP_NUM_THREADS=72 OMP_PROC_BIND=true OMP_PLACES=cores
CAV=(SFEM_CASE=cavity SFEM_NX=8 SFEM_NY=8 SFEM_NZ=8 SFEM_GEOM=affine)

run() {   # tag binary
    mkdir -p "$OUT/$1" && cd "$OUT/$1"
    stdbuf -oL -eL timeout 420 env "${CAV[@]}" SFEM_ELEMENT_REFINE_LEVEL=4 SFEM_FGMRES=1 \
        SFEM_GMG=1 SFEM_NL_MAX_IT=8 "$2" out > out.log 2>&1
    echo "--- $1 exit $?"
}

run ref_a "$REF"; run ref_b "$REF"
run new_a "$NEW"; run new_b "$NEW"

scal() { grep -ohE "[a-z_]+: *[-0-9.e+]+" "$OUT/$1/out.log" | grep -vE "^(t_|us_per|_ns_)"; }
for t in ref_a ref_b new_a new_b; do scal "$t" > "$OUT/$t.s"; done

echo
echo "=== scalar counts (a comparison of nothing is not agreement)"
for t in ref_a ref_b new_a new_b; do printf '  %-6s %s\n' "$t" "$(wc -l < "$OUT/$t.s")"; done

echo
cmp2() {   # label fileA fileB
    if diff -q "$2" "$3" >/dev/null 2>&1; then echo "  $1: IDENTICAL"
    else echo "  $1: DIFFERS"; diff "$2" "$3" | head -8; fi
}
echo "=== the same binary against ITSELF -- this is the question"
cmp2 "ref vs ref" "$OUT/ref_a.s" "$OUT/ref_b.s"
cmp2 "new vs new" "$OUT/new_a.s" "$OUT/new_b.s"
echo
echo "=== and across the change, for reference"
cmp2 "ref_a vs new_a" "$OUT/ref_a.s" "$OUT/new_a.s"
cmp2 "ref_b vs new_b" "$OUT/ref_b.s" "$OUT/new_b.s"
echo "=== DONE $(date)"
