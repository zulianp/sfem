#!/usr/bin/env bash
#SBATCH --account=c40
#SBATCH --job-name=cvfem-live
#SBATCH --partition=debug
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=72
#SBATCH --time=00:25:00
#SBATCH --exclusive --hint=nomultithread --uenv-passthrough=ignore
#SBATCH --output=lv_%j.out
#SBATCH --error=lv_%j.err
# Does the Krylov working set account for the gap between the bench and the solver?
#
# At 4.12M dof the bench's packed Jacobian action reads 1458 MDOF/s and the solver's own
# scope, on 4.34M dof and 1.05M elements against the bench's 1.00M, reads 851. Size is
# not the difference -- the bench is flat from 4.1M to 11.2M -- and the pack geometry is
# the same 2048 elements per pack. What is left is that the bench replays one apply on
# one direction with a small warm working set, while the solver evaluates the same kernel
# with its Krylov basis live and a different direction every iteration.
#
# --live-vectors N is that difference and nothing else: N vectors of the solution size,
# churned between applies and used in rotation as the direction, with only the apply on
# the clock. BiCGStab carries about seven. Sweeping N says whether the gap closes, and
# where.
set -u
B=$SCRATCH/cvfem_blk/after/build_blk
OUT=${OUT:-$SCRATCH/cvfem_lv/$SLURM_JOB_ID}; mkdir -p "$OUT"
export OMP_NUM_THREADS=72 OMP_PROC_BIND=true OMP_PLACES=cores
U=$(grep '^uenv start' ~/init.sh | sed 's/^uenv start //')
echo "host $(hostname)  threads $OMP_NUM_THREADS  $(date)"
echo "the solver's own scope at 4,343,300 dof reads 851 MDOF/s"
echo

for n in 100; do
  for lv in 0 1 3 5 7 10 14; do
    best=""; dofs=""; mib=""
    arg=""; [ "$lv" -gt 0 ] && arg="--live-vectors $lv"
    for r in 1 2 3; do
      log="$OUT/n${n}_lv${lv}_r$r.log"
      uenv run $U -- $B/cvfem_hex8_ns_upwind_bench --n $n --layout packed --jac-action \
        --rhie-chow 1.0 --repeat 20 --warmup 5 $arg > "$log" 2>&1 || { echo "  lv=$lv FAILED"; tail -2 "$log"; break; }
      dofs=$(grep -m1 -o 'dofs: [0-9]*' "$log" | awk '{print $2}')
      mib=$(grep -m1 -o '([0-9.]* MiB total)' "$log" | tr -d '()')
      k=$(grep -m1 'MDOF/s_jac_action_kernel_only:' "$log" | awk '{print $2}')
      best=$(awk -v a="$best" -v b="$k" 'BEGIN{print (a=="" || b+0>a+0)?b:a}')
    done
    printf "  live_vectors=%-3s ndof=%-9s %-18s kernel_only=%-9s vs_solver=%s\n" \
      "$lv" "${dofs:-?}" "${mib:-–}" "${best:--}" \
      "$(awk -v k="$best" 'BEGIN{ if (k=="") print "-"; else printf "%.2fx", k/851.0 }')"
  done
done

echo
echo "done $(date)"
