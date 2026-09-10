#!/usr/bin/env bash
#SBATCH --account=c40
#SBATCH --job-name=cvfem-kernels
#SBATCH --partition=debug
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=72
#SBATCH --time=00:30:00
#SBATCH --exclusive --hint=nomultithread --uenv-passthrough=ignore
#SBATCH --output=kernels_%j.out
#SBATCH --error=kernels_%j.err
#
# The measurement behind docs/CVFEM_Kernels.md.
#
#   sbatch jobs/kernels.sbatch
#   python3 python/cvfem_kernel_report.py <csv> -o docs/CVFEM_Kernels.md --html
#
# One allocation, one CSV. Everything in it is measured on the same idle socket, which is
# what makes the rows comparable at all: the same binary reads 5-11% apart on two different
# Grace nodes, so a report assembled from separate jobs would be reporting the machine.
#
# --exclusive is not optional for the same reason, and OMP_PROC_BIND=true rather than
# close or spread is this repository's convention.
#
# Two sizes. 4,121,204 dof is past saturation -- the packed residual reads 1798/1458/1476
# MDOF/s at 1.1M/4.1M/11.2M dof, so the ordering is settled well before this -- and
# 1,098,500 dof is there so that no ordering in the report rests on a single point.
#
# The order matters if the allocation runs out: the cascades come first, because they are
# the part of the report that cannot be reconstructed from anything else. The
# cross-product of layouts and kernels comes last.
set -u

ROOT=${SLURM_SUBMIT_DIR:-$PWD}
BIN=${BIN:-$ROOT/build_alps/cvfem_hex8_ns_upwind_bench}
OUT=${OUT:-$SCRATCH/cvfem_kernels/${SLURM_JOB_ID:-manual}}
CSV=$OUT/kernels.csv
mkdir -p "$OUT"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-72} OMP_PROC_BIND=true OMP_PLACES=cores
U=$(grep '^uenv start' ~/init.sh | sed 's/^uenv start //')

REPEAT=${REPEAT:-5}
WARMUP=${WARMUP:-2}
BIG=${BIG:-100}      # 4,121,204 dof
SMALL=${SMALL:-64}   # 1,098,500 dof

echo "host $(hostname)  threads $OMP_NUM_THREADS  $(date)"
echo "csv  $CSV"

run() {  # run <tag> <n> <flags...>
    local tag=$1 n=$2; shift 2
    printf '%-38s n=%-4s ' "$tag" "$n"
    if uenv run $U -- "$BIN" --n "$n" --repeat "$REPEAT" --warmup "$WARMUP" \
            --csv "$CSV" --tag "$tag" "$@" > "$OUT/$tag.n$n.log" 2>&1; then
        grep -m1 '  MDOF/s:' "$OUT/$tag.n$n.log" | awk '{printf "%10.1f MDOF/s\n", $2}'
    else
        echo "FAILED (see $OUT/$tag.n$n.log)"
    fi
}

# ---------------------------------------------------------------- the two cascades
#
# The element kernel alone, then the same kernel carrying each thing the solver needs. The
# two cascades are different lengths and the report says so rather than padding either: the
# residual has no direction-gradient stage, because there is no direction.
echo
echo "== residual cascade, packed =="
run res_bare            $BIG --layout packed
run res_rc              $BIG --layout packed --rhie-chow
run res_rc_perapply     $BIG --layout packed --rhie-chow --pgrad-per-apply
run res_rc_bnd          $BIG --layout packed --rhie-chow --boundary
run res_rc_bnd_dt       $BIG --layout packed --rhie-chow --boundary --transient 0.01
run res_solver          $BIG --layout packed --rhie-chow --pgrad-per-apply --boundary --transient 0.01

echo
echo "== Jacobian-action cascade, packed =="
# --rhie-chow on the action is the EXACT form: it differentiates through the nodal gradient
# reconstruction, and rebuilds the direction's gradient inside every apply because that one
# cannot be hoisted out of a Krylov solve. The frozen form appears in the assembled matrix
# below, deliberately, which is what the report's `frozen by design` cell means.
run jac_bare            $BIG --layout packed --jac-action
run jac_rc              $BIG --layout packed --jac-action --rhie-chow
run jac_rc_bnd          $BIG --layout packed --jac-action --rhie-chow --boundary
run jac_rc_bnd_dt       $BIG --layout packed --jac-action --rhie-chow --boundary --transient 0.01
run jac_live7           $BIG --layout packed --jac-action --rhie-chow --boundary --live-vectors 7
run spmv                $BIG --layout packed --bsr-apply

# ------------------------------------------------------- the same cascades, smaller
echo
echo "== both cascades at the smaller size =="
run res_bare            $SMALL --layout packed
run res_rc_bnd          $SMALL --layout packed --rhie-chow --boundary
run jac_bare            $SMALL --layout packed --jac-action
run jac_rc_bnd          $SMALL --layout packed --jac-action --rhie-chow --boundary

# ------------------------------------------------------------- the variant sweep
echo
echo "== residual, layout x kernel =="
for lay in packed atomic colored store; do
    for k in sumfact sympy current; do run "res_${lay}_${k}" $BIG --layout $lay --kernel $k; done
done
run res_packed_isoparam  $BIG --layout packed --geom isoparam --kernel current
run res_atomic_isoparam  $BIG --layout atomic --geom isoparam --kernel current
run res_atomic_iso_sympy $BIG --layout atomic --geom isoparam --kernel sympy

echo
echo "== Jacobian action, layout x geometry =="
for lay in packed atomic colored store; do run "jac_${lay}" $BIG --layout $lay --jac-action; done
run jac_atomic_isoparam  $BIG --layout atomic --jac-action --geom isoparam
run jac_packed_isoparam  $BIG --layout packed --jac-action --geom isoparam
# The generated arrangements. Scalar kernels, so the atomic layout is the only one that can
# call them -- see subpar/README.md for what the arrangement is worth.
for k in sympy_action sympy_action_node sympy_action_comp sympy_action_face \
         sympy_action_geom sympy_action_geomface; do
    run "jac_$k" $BIG --layout atomic --jac-action --kernel $k
done

echo
echo "== assembly, layout x kernel =="
for lay in atomic packed colored store; do
    for k in sumfact sympy sympy_block; do run "asm_${lay}_${k}" $BIG --layout $lay --assemble --kernel $k; done
done
run asm_atomic_fd        $BIG --layout atomic --assemble --kernel fd
run asm_atomic_split     $BIG --layout atomic --assemble --kernel split
run asm_atomic_isoparam  $BIG --layout atomic --assemble --geom isoparam --kernel current
run asm_atomic_rc        $BIG --layout atomic --assemble --rhie-chow
run asm_atomic_rc_bnd    $BIG --layout atomic --assemble --rhie-chow --boundary
run asm_packed_rc_bnd    $BIG --layout packed --assemble --rhie-chow --boundary

echo
echo "== block diagonal =="
run diag_bare            $BIG --layout atomic --assemble-diag
run diag_rc              $BIG --layout atomic --assemble-diag --rhie-chow
run diag_rc_bnd          $BIG --layout atomic --assemble-diag --rhie-chow --boundary
run diag_solver          $BIG --layout atomic --assemble-diag --rhie-chow --boundary --transient 0.01
run diag_isoparam        $BIG --layout atomic --assemble-diag --geom isoparam --rhie-chow --boundary

echo
echo "rows: $(( $(wc -l < "$CSV") - 1 ))"
echo "done $(date)"
echo "report: python3 python/cvfem_kernel_report.py $CSV -o docs/CVFEM_Kernels.md --html"
