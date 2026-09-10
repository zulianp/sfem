#!/usr/bin/env bash
#SBATCH --account=c40
#SBATCH --job-name=cvfem-tp-ab
#SBATCH --partition=debug
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=72
#SBATCH --time=00:28:00
#SBATCH --exclusive --hint=nomultithread --uenv-passthrough=ignore
#SBATCH --output=tpab_%j.out
#SBATCH --error=tpab_%j.err
# The throughput sweep with both binaries in one allocation, on one node.
#
# Running the sweep twice on different nodes is not a comparison. Measured that way the
# semi-structured configurations -- which no change in this range touches, and which are
# therefore the control -- came out about 8% slower across every size, which is a
# node-level offset and not a result. Both binaries run here back to back on the same
# node so the control can do its job.
#
#   ssgmg_base  the tree before the Rhie-Chow coefficient was hoisted and before the
#               boundary closure was restricted to the boundary elements
#   ssgmg       both changes in
#
# The scope throughput reported is calls * ndof / seconds on the operator's own element
# sweep, from the tracer, which does not depend on how many Krylov iterations a run took.
set -u
B=$SCRATCH/cvfem_blk/after/build_blk
OUT=${OUT:-$SCRATCH/cvfem_tpab/$SLURM_JOB_ID}; mkdir -p "$OUT"
export OMP_NUM_THREADS=72 OMP_PROC_BIND=true OMP_PLACES=cores
U=$(grep '^uenv start' ~/init.sh | sed 's/^uenv start //')
echo "host $(hostname)  threads $OMP_NUM_THREADS  $(date)"
printf "%-22s %-10s %10s %10s %8s   %s\n" run dof base_MDOFs new_MDOFs ratio "scope"

one() {  # one <label> <n> <level> <scope-regex>
  local label=$1 n=$2 lvl=$3 rx=$4
  local base_v="" new_v="" ndof=""
  for v in base new; do
    local bin=$([ $v = base ] && echo ssgmg_base || echo cvfem_hex8_ns_ssgmg)
    local d="$OUT/${label}_$v"; mkdir -p "$d"
    ( cd "$d"
      export SFEM_CASE=poiseuille SFEM_N=$n SFEM_ELEMENT_REFINE_LEVEL=$lvl
      export SFEM_GMG=0 SFEM_PRECOND=bjacobi SFEM_NL_MAX_IT=3 SFEM_LSOLVE_MAX_IT=150
      export SFEM_MU=0.01 SFEM_ENABLE_OUTPUT=0
      uenv run $U -- "$B/$bin" "$d/out" > run.log 2>&1 )
    ndof=$(grep -m1 -o 'ndof: [0-9]*' "$d/run.log" | awk '{print $2}')
    local m=""
    [ -f "$d/smesh.trace.csv" ] && m=$(awk -F, -v rx="$rx" -v nd="${ndof:-0}" '
        $1 ~ rx { printf "%.1f", (nd*$2)/$3/1e6; exit }' "$d/smesh.trace.csv")
    [ $v = base ] && base_v=$m || new_v=$m
  done
  local ratio="-"
  [ -n "$base_v" ] && [ -n "$new_v" ] && ratio=$(awk -v a="$base_v" -v b="$new_v" 'BEGIN{printf "%.2fx", b/a}')
  printf "%-22s %-10s %10s %10s %8s\n" "$label" "${ndof:-?}" "${base_v:--}" "${new_v:--}" "$ratio"
}

echo "--- flat HEX8 (changed) ---"
for n in 16 24 32 48 64; do one "flat_N$n" $n 1 "apply_jacobian_action_packed"; done
echo "--- semi-structured L2 (control: unchanged code) ---"
for n in 16 24 32 48 64; do one "ss_L2_N$((n/2))" $((n/2)) 2 "apply_macro_local_hoisted"; done
echo "--- semi-structured L4 (control) ---"
for n in 16 32 64; do one "ss_L4_N$((n/4))" $((n/4)) 4 "apply_macro_local_hoisted"; done

echo
echo "done $(date)"
