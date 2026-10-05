#!/usr/bin/env bash
#SBATCH --account=c40
#SBATCH --job-name=cvfem-kmix
#SBATCH --partition=debug
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=72
#SBATCH --time=00:25:00
#SBATCH --exclusive --hint=nomultithread --uenv-passthrough=ignore
#SBATCH --output=kmix_%j.out
#SBATCH --error=kmix_%j.err

# WHERE THE KERNEL'S TIME GOES, PACKED AGAINST STANDARD. (paper Table 8, tab:kmix)
#
# This job existed once, produced kmix_4951437.out, and was then lost -- it was written ad hoc
# and never committed, so when the operator changed there was no way to re-measure and Table 8
# silently described an older binary. It lives in jobs/ now for that reason.
#
# The output format is deliberately dumb: one `kmix <layout> <key> <value>` line per quantity,
# so python/perf_figures.py's parse_kmix does not have to guess at perf's output, which differs
# between counter groups and between versions. Anything the parser needs must appear as such a
# line; everything else in this file is commentary.
#
# Counters, and what each separates:
#   INST_SPEC                     instructions speculatively executed, the denominator
#   ASE_SPEC / VFP_SPEC / DP_SPEC advanced SIMD, scalar FP, integer -- is the kernel vectorised
#   LD_SPEC / ST_SPEC             loads and stores, the staging and accumulator traffic
#   cycles / instructions         IPC, which is what separates the two layouts
#   STALL_BACKEND / _MEM          whether the stalls are memory waits or issue limits
#
# Two groups rather than one run: Neoverse V2 has six programmable counters, and asking for more
# multiplexes them, which makes the ratios depend on the sampling rather than on the kernel.
#
# Two convective arms, not one. The table's question is whether the layouts' instruction mixes
# explain the throughput gap, and an answer taken only on first-order upwinding leaves open
# whether it survives the arithmetic a higher-order scheme adds -- which is exactly where the
# gap narrows (paper Figure 8). The higher-order arm is Darwish--Moukalled, the same limiter
# Figures 7 and 8 single out, and it carries the Rhie--Chow coupling like the first-order arm so
# the two differ in the correction alone. Its key is `<layout>_ho`; the first-order arm keeps the
# bare `<layout>` key so the data files written before this arm existed still parse.
#
# Note the order of the two sub() calls below. perf -x, puts the event in field 3 as
# `armv8_pmuv3_0/INST_SPEC/`, and stripping the prefix with a greedy `.*/` first consumes the
# TRAILING slash too and leaves an empty string -- which silently dropped every PMU event while
# keeping `cycles` and `instructions`, because those carry no slash. The trailing slash goes
# first.

set -uo pipefail
cd "$SLURM_SUBMIT_DIR" || exit 1
source scripts/alps_env.sh || exit 1

B=$PWD/build/cvfem_hex8_ns_upwind_bench
[ -x "$B" ] || { echo "no bench at $B" >&2; exit 1; }
N=${N:-128}
REPEAT=${REPEAT:-10}
PACK=${PACK:-1024}

echo "### kernel instruction mix, n=$N, 72 threads"
echo "### host $(hostname)  $(date '+%F %T')"

run_group() {   # layout, perf event list, scheme flags
    local lay=$1 events=$2 extra=$3
    local layflag="--layout $lay"
    [ "$lay" = "packed" ] && layflag="$layflag --pack-size $PACK"
    srun -u --ntasks=1 --cpus-per-task=72 --uenv="$CVFEM_UENV" --view="$CVFEM_VIEW" \
      bash -lc "cd $PWD && OMP_NUM_THREADS=72 OMP_PROC_BIND=true OMP_PLACES=cores \
                perf stat -x, -e $events -- \
                $B --n $N --repeat $REPEAT --warmup 3 $layflag --rhie-chow $extra" 2>&1
}

for arm in "|" "_ho|--conv-ho 3"; do
  suffix=${arm%%|*}
  extra=${arm#*|}
  for lay in packed atomic; do
    key="${lay}${suffix}"
    # The throughput, and the kernel-only figure the table reports beside it.
    out=$(srun -u --ntasks=1 --cpus-per-task=72 --uenv="$CVFEM_UENV" --view="$CVFEM_VIEW" \
            bash -lc "cd $PWD && OMP_NUM_THREADS=72 OMP_PROC_BIND=true OMP_PLACES=cores \
                      $B --n $N --repeat $REPEAT --warmup 3 --layout $lay \
                      $([ "$lay" = packed ] && echo "--pack-size $PACK") --rhie-chow $extra" 2>&1)
    mdof=$(printf '%s\n' "$out" | sed -n 's/^  MDOF\/s: \(.*\)$/\1/p' | head -1)
    [ -n "$mdof" ] && echo "kmix $key mdof $mdof"

    # The same sweep with --kernel-only, which has the element kernel write to a dense stack
    # buffer and skip the scatter entirely. The table reports mdof/mdof_kernel_only as the share
    # of the sweep the scatter and its staging cost, so this must be a SEPARATE run rather than
    # a phase timing: deriving it from ms_element_kernel does not work, because the standard
    # layout does not report those phase fields at all and the ratio would be silently wrong for
    # one of the two columns.
    outk=$(srun -u --ntasks=1 --cpus-per-task=72 --uenv="$CVFEM_UENV" --view="$CVFEM_VIEW" \
             bash -lc "cd $PWD && OMP_NUM_THREADS=72 OMP_PROC_BIND=true OMP_PLACES=cores \
                       $B --n $N --repeat $REPEAT --warmup 3 --layout $lay \
                       $([ "$lay" = packed ] && echo "--pack-size $PACK") --rhie-chow $extra --kernel-only" 2>&1)
    mdofk=$(printf '%s\n' "$outk" | sed -n 's/^  MDOF\/s: \(.*\)$/\1/p' | head -1)
    [ -n "$mdofk" ] && echo "kmix $key mdof_kernel_only $mdofk"

    # Group 1: the instruction mix.
    printf '%s\n' "$(run_group "$lay" 'armv8_pmuv3_0/INST_SPEC/,armv8_pmuv3_0/ASE_SPEC/,armv8_pmuv3_0/VFP_SPEC/,armv8_pmuv3_0/DP_SPEC/,armv8_pmuv3_0/LD_SPEC/,armv8_pmuv3_0/ST_SPEC/' "$extra")" \
      | awk -F, -v l="$key" '$1 ~ /^[0-9]/ && $3 != "" {
            k=$3; sub(/\/$/,"",k); sub(/^.*\//,"",k); if (k != "") printf "kmix %s %s %s\n", l, k, $1 }'

    # Group 2: cycles, instructions and the stall split.
    printf '%s\n' "$(run_group "$lay" 'cycles,instructions,armv8_pmuv3_0/STALL_BACKEND/,armv8_pmuv3_0/STALL_BACKEND_MEM/' "$extra")" \
      | awk -F, -v l="$key" '$1 ~ /^[0-9]/ && $3 != "" {
            k=$3; sub(/\/$/,"",k); sub(/^.*\//,"",k); if (k != "") printf "kmix %s %s %s\n", l, k, $1 }'
  done
done
echo "### done"
