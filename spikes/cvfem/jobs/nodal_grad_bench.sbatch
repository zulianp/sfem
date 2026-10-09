#!/usr/bin/env bash
#SBATCH --account=c40
#SBATCH --job-name=cvfem-ngrad
#SBATCH --partition=debug
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=72
#SBATCH --time=00:29:00
#SBATCH --exclusive --hint=nomultithread --uenv-passthrough=ignore
#SBATCH --output=ngrad_%j.out
#SBATCH --error=ngrad_%j.err

# THE NODAL GRADIENT RECONSTRUCTION, ON ITS OWN.
#
# It has only ever been timed from inside the operators that call it, as a fraction of their
# matvec, and it is now the whole of the exact higher-order Jacobian action's 4.3x. This is the
# baseline every optimisation of it is measured against.
#
# nf = 1 is the pressure gradient every Rhie-Chow residual builds; nf = 3 is the velocity gradient
# the exact higher-order action rebuilds on every matvec. Both layouts, because the gradient sweep
# now follows the layout under test and a one-layout number is not a result here.

set -uo pipefail
cd "$SLURM_SUBMIT_DIR" || exit 1
source scripts/alps_env.sh || exit 1

B=$PWD/build/cvfem_hex8_ns_upwind_bench
[ -x "$B" ] || { echo "no bench at $B" >&2; exit 1; }
OUT=${OUT:-$SCRATCH/cvfem_ngrad/${SLURM_JOB_ID:-manual}}
mkdir -p "$OUT"
CSV=$OUT/ngrad.csv
N=${N:-128}
REPEAT=${REPEAT:-20}
PASSES=${PASSES:-3}
# The measured optimum for this n, pinned rather than left to the library default. The default
# derives from the index-type ceiling and the thread count, so it moves with the thread count and
# with n, and an arm that takes it is not reproducible against the rest of the paper even when it
# happens to land on the peak -- which at n=128 it does. Read it from the pack-size sweep
# (\packLargeBest); PACK= (empty) restores the library default deliberately.
PACK=${PACK:-1024}

echo "### host $(hostname)  n=$N repeat=$REPEAT passes=$PASSES  $(date '+%F %T')"
echo

printf '%-10s %-8s %10s %12s %10s\n' fields layout MDOF_s MCOMP_s dofs
for nf in 1 3; do
for lay in packed atomic; do
    best=""; bestc=""; dofs=""
    PACKFLAG=""; [ -n "$PACK" ] && [ "$lay" = packed ] && PACKFLAG="--pack-size $PACK"
    for pass in $(seq 1 "$PASSES"); do
        log=$OUT/nf${nf}_${lay}_p${pass}.log
        timeout "${RUN_TIMEOUT:-420}" \
          srun -u --ntasks=1 --cpus-per-task=72 --uenv="$CVFEM_UENV" --view="$CVFEM_VIEW" \
          bash -lc "cd $PWD && OMP_NUM_THREADS=72 OMP_PROC_BIND=true OMP_PLACES=cores $B --n $N --repeat $REPEAT --warmup 3 --layout $lay $PACKFLAG --nodal-grad $nf --csv $CSV --tag ngrad_nf${nf}_${lay}_p${pass}" \
          > "$log" 2>&1
        val() { sed -n "s|^  $1: \(.*\)$|\1|p" "$log" | head -1; }
        v=$(val 'MDOF/s_nodal_grad'); [ -n "$v" ] || continue
        best=$(awk -v a="${best:-0}" -v b="$v" 'BEGIN{print (b+0>a+0)?b:a}')
        c=$(val 'MCOMP/s_nodal_grad')
        bestc=$(awk -v a="${bestc:-0}" -v b="$c" 'BEGIN{print (b+0>a+0)?b:a}')
        dofs=$(val 'dofs')
    done
    printf '%-10s %-8s %10s %12s %10s\n' "$nf" "$lay" "${best:--}" "${bestc:--}" "${dofs:--}"
done
done
echo
echo "csv: $CSV"
