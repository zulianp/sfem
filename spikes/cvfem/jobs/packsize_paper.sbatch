#!/usr/bin/env bash
#SBATCH --account=c40
#SBATCH --job-name=cvfem-packsz-paper
#SBATCH --partition=debug
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=72
#SBATCH --time=00:29:00
#SBATCH --exclusive --hint=nomultithread --uenv-passthrough=ignore
#SBATCH --output=packsz_paper_%j.out
#SBATCH --error=packsz_paper_%j.err

# PACK SIZE, THE INDEX-TYPE CEILING, AND SPATIAL ORDERING. (paper M5 and M7, figure F7)
#
# Two questions the paper needs, measured together because they are the same sweep.
#
# 1. WHAT SELECTS THE DEFAULT PACK SIZE, AND IS IT THE RIGHT CONSTRAINT?
#
#    A pack may not address more distinct nodes than its index type admits -- pack_idx_t is
#    uint16_t, so 65536 -- and the packer sizes packs against that ceiling using the pessimistic
#    bound n_elements * nodes_per_element, which credits no sharing at all. For HEX8 that yields
#    about 8192 elements per pack. But a pack is ONE OpenMP iteration, so the pack count IS the
#    available parallelism, and the ceiling knows nothing about how many cores the machine has.
#
#    TWO PROBLEM SIZES, because the failure only appears at one of them and a sweep at the large
#    size alone would miss it entirely:
#      n = 128: 2,097,152 elements, so even a 8192-element pack leaves 256 packs for 72 threads;
#      n =  64:   262,144 elements, so a 8192-element pack leaves THIRTY-TWO -- fewer than the
#                 machine has cores, and less than half of them can be busy.
#    This is the point the paper makes analytically; the job is what turns it into evidence.
#
# 2. IS SPATIAL ORDERING A PRECONDITION OR A TUNING KNOB?
#
#    Packs are contiguous ranges of ELEMENT INDICES, so the node set of a pack is decided by the
#    element order. --no-sfc disables the space-filling pass and leaves the lexicographic order.
#    Reported beside the ordered run at the same pack size: if the node set per pack grows and
#    the throughput falls, ordering is a precondition of the format rather than an optimisation
#    applied to it.
#
# mean_nodes_per_pack is printed for every row because it is the mechanism behind both answers:
# it is what a larger pack and an unordered mesh each inflate.

set -uo pipefail
cd "$SLURM_SUBMIT_DIR" || exit 1
source scripts/alps_env.sh || exit 1

B=$PWD/build/cvfem_hex8_ns_upwind_bench
[ -x "$B" ] || { echo "no bench at $B" >&2; exit 1; }
OUT=${OUT:-$SCRATCH/cvfem_packsz_paper/${SLURM_JOB_ID:-manual}}
mkdir -p "$OUT"
CSV=$OUT/packsize.csv

SIZES=${SIZES:-"64 128"}
# 128 and 512 are in the grid because the optimum at n=128 sits at 512: a sweep whose grid does
# not contain the peak reports the nearest point it happens to hold and every arm pinned from it
# is then off the peak. They were dropped from this list once and the next run duly nominated
# 1024.
PACKS=${PACKS:-"128 256 512 1024 2048 4096 8192"}
REPEAT=${REPEAT:-10}
WARMUP=${WARMUP:-3}

echo "### host $(hostname)  threads 72  repeat=$REPEAT warmup=$WARMUP  $(date '+%F %T')"
echo "### 72 cores; a pack is one OpenMP iteration, so n_packs is the available parallelism"
echo "### operators: residual and the EXACT Rhie-Chow Jacobian action, both with the boundary"
echo "###            closure. The exact action is the two-sweep form -- it reconstructs the"
echo "###            direction gradient every matvec -- not the lagged one the matrix holds."
echo

# BEST OF THREE PASSES, NOT ONE.
#
# A single pass cannot separate this band. Measured by this job's own first version, which took
# one pass per configuration: the identical configuration (n=64, pack 2048, space-filling)
# appeared twice in one allocation and read 1333.6 and 842.0 MDOF/s for the residual -- 58%
# apart -- while the Jacobian action read 976.3 and 978.8, which is 0.3%. So the residual
# carries a large run-to-run variance that the Jacobian action does not, and any residual
# conclusion drawn from one pass is unsafe.
#
# jobs/pack_size_sweep.sbatch had already established best-of-three for exactly this
# measurement and recorded why; this job was written without carrying that over.
#
# Best rather than median: background load can only make a run slower.
PASSES=${PASSES:-3}

run_one() {   # label n pack_size op extra
    local lbl=$1 n=$2 ps=$3 op=$4 extra=${5:-}
    local flags="--layout packed --rhie-chow --boundary --pack-size $ps $extra"
    [ "$op" = jac ] && flags="$flags --jac-action"
    LAST_MDOF=""; LAST_PACKS=""; LAST_MEAN=""; LAST_EPP=""
    local pass log v
    for pass in $(seq 1 "$PASSES"); do
        log=$OUT/${lbl}_p${pass}.log
        timeout "${RUN_TIMEOUT:-300}" \
          srun -u --ntasks=1 --cpus-per-task=72 --uenv="$CVFEM_UENV" --view="$CVFEM_VIEW" \
          bash -lc "cd $PWD && OMP_NUM_THREADS=72 OMP_PROC_BIND=true OMP_PLACES=cores $B --n $n --repeat $REPEAT --warmup $WARMUP $flags --csv $CSV --tag ${lbl}_p${pass}" \
          > "$log" 2>&1
        # Anchored on the exact key: a loose MDOF/s pattern also matches MDOF/s_element_visits.
        val() { sed -n "s|^  $1: \\(.*\\)$|\\1|p" "$log" | head -1; }
        v=$(val 'MDOF/s')
        [ -n "$v" ] || continue
        LAST_MDOF=$(awk -v a="${LAST_MDOF:-0}" -v b="$v" 'BEGIN{print (b+0>a+0)?b:a}')
        LAST_PACKS=$(val 'n_packs'); LAST_MEAN=$(val 'mean_nodes_per_pack')
        LAST_EPP=$(val 'n_elements_per_pack')
    done
    [ -n "$LAST_MDOF" ] || LAST_MDOF="-" 
}

echo "### pack-size sweep, space-filling order"
printf '%-5s %9s %9s %8s %9s %10s %10s %11s\n' \
       n pack_req elems/pack n_packs packs/thr mean_nodes res_MDOFs jac_MDOFs
for n in $SIZES; do
  for ps in $PACKS; do
    run_one "sfc_n${n}_p${ps}_res" "$n" "$ps" res
    res=$LAST_MDOF; packs=$LAST_PACKS; mean=$LAST_MEAN; epp=$LAST_EPP
    run_one "sfc_n${n}_p${ps}_jac" "$n" "$ps" jac
    jac=$LAST_MDOF
    # Packs per thread, the quantity that decides whether the socket can be filled at all.
    ppt=$(awk -v p="${packs:-0}" 'BEGIN{ if (p+0>0) printf "%.1f", p/72.0; else print "-" }')
    printf '%-5s %9s %9s %8s %9s %10s %10s %11s\n' \
           "$n" "$ps" "${epp:--}" "${packs:--}" "$ppt" "${mean:--}" "${res:--}" "${jac:--}"
  done
done

echo
echo "### spatial ordering: same pack size, space-filling against lexicographic"
printf '%-5s %9s %-14s %10s %10s %11s\n' n pack order mean_nodes res_MDOFs jac_MDOFs
for n in $SIZES; do
  for ps in 2048; do
    for ord in sfc nosfc; do
      extra=""; [ "$ord" = nosfc ] && extra="--no-sfc"
      run_one "ord_${ord}_n${n}_res" "$n" "$ps" res "$extra"; res=$LAST_MDOF; mean=$LAST_MEAN
      run_one "ord_${ord}_n${n}_jac" "$n" "$ps" jac "$extra"; jac=$LAST_MDOF
      printf '%-5s %9s %-14s %10s %10s %11s\n' \
             "$n" "$ps" "$ord" "${mean:--}" "${res:--}" "${jac:--}"
    done
  done
done

echo
echo "csv: $CSV"
echo "logs in $OUT"
