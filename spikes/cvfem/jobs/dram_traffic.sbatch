#!/usr/bin/env bash
#SBATCH --account=c40
#SBATCH --job-name=cvfem-dram
#SBATCH --partition=debug
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=72
#SBATCH --time=00:25:00
#SBATCH --exclusive --hint=nomultithread --uenv-passthrough=ignore
#SBATCH --output=dram_%j.out
#SBATCH --error=dram_%j.err

# MEASURED DRAM TRAFFIC, from the Grace SCF counters. (paper: the roofline's bytes per dof)
#
# Like jobs/kernel_mix.sbatch, this reproduces a job that was written ad hoc, produced
# dram_4816010_4816011.out, and was then lost -- after which the roofline's arithmetic intensity
# could not be re-measured. It lives in jobs/ now.
#
# THE METHOD, which is the whole point. The SCF counters cover a whole process, so a single
# capture includes mesh setup, pack construction and the warmup applies, and cannot simply be
# divided by the apply count. Two captures at different repeat counts are differenced instead, so
# everything that does not scale with the apply count cancels exactly:
#
#     bytes per apply = (bytes_hi - bytes_lo) / (repeat_hi - repeat_lo)
#
# Both captures must otherwise be identical -- same n, same warmup, same binding -- or the
# difference picks up whatever else changed.
#
# UNITS. cmem_rd_access is exactly half cmem_rd_data in every row, which fixes cmem_rd_data as a
# count of 32-byte flits (a 64-byte line being two flits and one access); cmem_wr_total_bytes is
# already in bytes. Hence
#
#     bytes = cmem_rd_data * 32 + cmem_wr_total_bytes
#
# The emitted table keeps BOTH captures rather than the difference, so the parser can see the
# method and a later reader can check the cancellation rather than trust it.

set -uo pipefail
cd "$SLURM_SUBMIT_DIR" || exit 1
source scripts/alps_env.sh || exit 1

B=$PWD/build/cvfem_hex8_ns_upwind_bench
[ -x "$B" ] || { echo "no bench at $B" >&2; exit 1; }
N=${N:-128}
LO=${LO:-10}
HI=${HI:-40}
PACK=${PACK:-1024}
EV='nvidia_scf_pmu_0/cmem_rd_data/,nvidia_scf_pmu_0/cmem_wr_total_bytes/,nvidia_scf_pmu_0/cmem_rd_access/,nvidia_scf_pmu_0/cmem_wr_access/'

capture() {  # op-flags layout repeat -> "bytes dofs"
    local extra=$1 lay=$2 rep=$3
    local layflag="--layout $lay"
    [ "$lay" = "packed" ] && layflag="$layflag --pack-size $PACK"
    local out
    out=$(srun -u --ntasks=1 --cpus-per-task=72 --uenv="$CVFEM_UENV" --view="$CVFEM_VIEW" \
            bash -lc "cd $PWD && OMP_NUM_THREADS=72 OMP_PROC_BIND=true OMP_PLACES=cores \
                      perf stat -x, -e $EV -- \
                      $B --n $N --repeat $rep --warmup 3 $layflag --kernel sumfact $extra" 2>&1)
    local rd wr dofs
    rd=$(printf '%s\n' "$out"  | awk -F, '$3 ~ /cmem_rd_data/        {print $1; exit}')
    wr=$(printf '%s\n' "$out"  | awk -F, '$3 ~ /cmem_wr_total_bytes/ {print $1; exit}')
    dofs=$(printf '%s\n' "$out"| sed -n 's/^  dofs: \(.*\)$/\1/p' | head -1)
    [ -n "$rd" ] && [ -n "$wr" ] && [ -n "$dofs" ] || return 1
    awk -v r="$rd" -v w="$wr" -v d="$dofs" 'BEGIN{printf "%.0f %s", r*32 + w, d}'
}

echo "# MEASURED DRAM TRAFFIC, Grace SCF counters, job ${SLURM_JOB_ID:-manual} (repeat=$LO and repeat=$HI)."
echo "#"
echo "# Two captures differenced across repeat count so that mesh setup, pack construction and the"
echo "# three warmup applies cancel exactly. Both runs used --warmup 3, n=$N, 72 threads,"
echo "# OMP_PROC_BIND=true. See jobs/dram_traffic.sbatch for the units."
echo "#"
echo "# bytes = cmem_rd_data * 32 + cmem_wr_total_bytes"
echo "#"
echo "# operation  layout  repeat_lo  repeat_hi  bytes_lo        bytes_hi         dofs"
# BOTH the bare and the Rhie-Chow arms, under distinct labels. The file this replaces carried
# rows labelled only "residual" and "jac_action" with no record of which flags produced them, and
# when it was re-measured the numbers came out 1.5x and 3.7x higher -- not because anything
# regressed, but because the new run carried Rhie-Chow and the old one evidently did not. Nobody
# could tell from the file. Labelling the arm is the fix.
for arm in "residual|" "residual_rc|--rhie-chow" \
           "jac_action|--jac-action" "jac_action_rc|--jac-action --rhie-chow"; do
    IFS='|' read -r op extra <<<"$arm"
    for lay in packed atomic; do
        lo=$(capture "$extra" "$lay" "$LO") || { echo "# MISSING $op $lay lo"; continue; }
        hi=$(capture "$extra" "$lay" "$HI") || { echo "# MISSING $op $lay hi"; continue; }
        printf '%-12s %-7s %-3s %-3s %-13s %-14s %s\n' \
               "$op" "$lay" "$LO" "$HI" "${lo% *}" "${hi% *}" "${lo#* }"
    done
done
