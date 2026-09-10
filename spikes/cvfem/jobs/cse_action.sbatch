#!/usr/bin/env bash
#SBATCH --account=c40
#SBATCH --job-name=cvfem-cse-action
#SBATCH --partition=debug
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=72
#SBATCH --time=00:28:00
#SBATCH --exclusive --hint=nomultithread --uenv-passthrough=ignore
#SBATCH --output=cse_%j.out
#SBATCH --error=cse_%j.err
# The CSE arrangement question, asked of the Jacobian action for the first time.
#
# HEX8 had no generated action until now, so the arrangement axis has only ever been
# measured on assembly -- where sympy_face lost decisively (24.0 against 54.3 MDOF/s on
# CPU atomic, issuing 2016 atomic adds against 768) and sympy_row lost a 0.2% tie inside a
# 4% noise band. Neither result says anything about the action.
#
# Three arrangements over one expression tree, differing only in the scope handed to one
# sp.cse call, against the hand-written scalar action as the reference:
#
#   sumfact             the hand-written kernel (this is what --layout atomic runs today)
#   sympy_action        all 32 outputs in one scope   1295 temporaries
#   sympy_action_node   the 4 dofs of one node, x8    1597 temporaries
#   sympy_action_comp   one component across nodes,x4 1411 temporaries
#   sympy_action_face   one sub-control surface,     x12
#
# Face-wise is here because the assembly verdict against it does not carry over: it lost
# there by issuing 2016 atomic adds against flat's 768, and the action accumulates into a
# local r[] where those are register or stack traffic instead. It is also the finest cut,
# and cutting the scope is what helped in the first round.
#
# Reuse against live ranges. Two sizes, because an arrangement that wins only below
# saturation has won nothing -- and because writing one off on a single sample is how the
# assembly verdict came to rest on a 0.2% margin.
set -u
B=$SCRATCH/cvfem_blk/after/build_blk
OUT=${OUT:-$SCRATCH/cvfem_cse/$SLURM_JOB_ID}; mkdir -p "$OUT"
export OMP_NUM_THREADS=72 OMP_PROC_BIND=true OMP_PLACES=cores
U=$(grep '^uenv start' ~/init.sh | sed 's/^uenv start //')
echo "host $(hostname)  threads $OMP_NUM_THREADS  $(date)"
printf "%-10s %-22s %-12s %-12s %s\n" size kernel "MDOF/s" "vs handwritten" checksum

for n in 64 100; do
  ref=""
  for k in sumfact sympy_action sympy_action_node sympy_action_comp sympy_action_face; do
    best=""; cs=""; dofs=""
    for r in 1 2 3; do
      log="$OUT/n${n}_${k}_$r.log"
      uenv run $U -- $B/cvfem_hex8_ns_upwind_bench --n $n --layout atomic --jac-action \
        --kernel $k --repeat 20 --warmup 5 > "$log" 2>&1 || { echo "  $k FAILED"; break; }
      m=$(grep -m1 '^  MDOF/s:' "$log" | awk '{print $2}')
      best=$(awk -v a="$best" -v b="$m" 'BEGIN{print (a=="" || b+0>a+0)?b:a}')
      cs=$(grep -m1 'checksum:' "$log" | awk '{print $2}')
      dofs=$(grep -m1 -o 'dofs: [0-9]*' "$log" | awk '{print $2}')
    done
    [ "$k" = sumfact ] && ref=$best
    rel=$(awk -v a="$ref" -v b="$best" 'BEGIN{ if (a=="" || b=="") print "-"; else printf "%.2fx", b/a }')
    printf "%-10s %-22s %-12s %-12s %s\n" "${dofs:-?}" "$k" "${best:--}" "$rel" "$cs"
  done
  echo
done
echo "done $(date)"
