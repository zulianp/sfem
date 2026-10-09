#!/usr/bin/env bash
#SBATCH --account=c40
#SBATCH --job-name=cvfem-sfc-drv
#SBATCH --partition=debug
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=72
#SBATCH --time=00:28:00
#SBATCH --exclusive --hint=nomultithread --uenv-passthrough=ignore
#SBATCH --output=sfcd_%j.out
#SBATCH --error=sfcd_%j.err
# What space-filling the driver's mesh is worth.
#
# The benchmark said 1.25x for the ordering alone on a cube (2735 nodes per 2048-element
# pack space-filled against 4373 lexicographic, 1483 against 1183 MDOF/s). This runs the
# same binary with SFEM_SFC=1 and 0 back to back on one node, so the only difference is
# the order, and reads the element sweep from the tracer rather than from wall clock --
# the linear-iteration count moves slightly with the summation order and would otherwise
# contaminate a solve-time comparison.
#
# The semi-structured rows are here because the reorder is applied to the macro mesh
# before to_semistructured, so it reaches that path too and nobody has measured whether
# it helps or hurts there.
set -u
B=$SCRATCH/cvfem_blk/after/build_blk
OUT=${OUT:-$SCRATCH/cvfem_sfcd/$SLURM_JOB_ID}; mkdir -p "$OUT"
export OMP_NUM_THREADS=72 OMP_PROC_BIND=true OMP_PLACES=cores
U=$(grep '^uenv start' ~/init.sh | sed 's/^uenv start //')
echo "host $(hostname)  threads $OMP_NUM_THREADS  $(date)"
printf "%-16s %-10s %10s %10s %8s  %s\n" run dof off_MDOFs on_MDOFs ratio "u_linf on/off"

one() {  # one <label> <n> <level> <scope-regex>
  local label=$1 n=$2 lvl=$3 rx=$4
  local off="" on="" ndof="" e_off="" e_on=""
  for sfc in 0 1; do
    local d="$OUT/${label}_sfc$sfc"; mkdir -p "$d"
    ( cd "$d"
      export SFEM_SFC=$sfc SFEM_CASE=poiseuille SFEM_N=$n SFEM_ELEMENT_REFINE_LEVEL=$lvl
      export SFEM_GMG=0 SFEM_PRECOND=bjacobi SFEM_NL_MAX_IT=3 SFEM_LSOLVE_MAX_IT=150
      export SFEM_MU=0.01 SFEM_ENABLE_OUTPUT=0
      uenv run $U -- "$B/cvfem_hex8_ns_ssgmg" "$d/out" > run.log 2>&1 )
    ndof=$(grep -m1 -o 'ndof: [0-9]*' "$d/run.log" | awk '{print $2}')
    local e; e=$(grep -m1 -o 'u_linf: [0-9.e+-]*' "$d/run.log" | awk '{print $2}')
    local m=""
    [ -f "$d/smesh.trace.csv" ] && m=$(awk -F, -v rx="$rx" -v nd="${ndof:-0}" '
        $1 ~ rx { printf "%.1f", (nd*$2)/$3/1e6; exit }' "$d/smesh.trace.csv")
    if [ $sfc = 0 ]; then off=$m; e_off=$e; else on=$m; e_on=$e; fi
  done
  local ratio="-"
  [ -n "$off" ] && [ -n "$on" ] && ratio=$(awk -v a="$off" -v b="$on" 'BEGIN{printf "%.2fx", b/a}')
  printf "%-16s %-10s %10s %10s %8s  %s / %s\n" "$label" "${ndof:-?}" "${off:--}" "${on:--}" "$ratio" "${e_on:--}" "${e_off:--}"
}

echo "--- flat HEX8 ---"
for n in 32 48 64; do one "flat_N$n" $n 1 "apply_jacobian_action_packed"; done
echo "--- semi-structured L2 ---"
for n in 16 24 32; do one "ss_L2_N$n" $n 2 "apply_macro_local_hoisted"; done

echo
echo "done $(date)"
