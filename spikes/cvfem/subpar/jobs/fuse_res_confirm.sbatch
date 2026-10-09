#!/usr/bin/env bash
#SBATCH --account=c40
#SBATCH --job-name=cvfem-fusecnf
#SBATCH --partition=debug
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=72
#SBATCH --time=00:29:00
#SBATCH --exclusive --hint=nomultithread --uenv-passthrough=ignore
#SBATCH --output=fusecnf_%j.out
#SBATCH --error=fusecnf_%j.err

# CONFIRMING THE ONE ARM THAT WON: THE FIRST-ORDER RESIDUAL WITH RHIE-CHOW.
#
# Job 4953412 put shape 2 -- twelve faces in one lane loop, accumulating into thirty-two
# lane-private scalars stored once -- ahead of the per-face loop on this arm by 5.4% packed and
# 4.5% standard, while the higher-order residual was flat (0.994-0.996) and the jac_exact control
# read 0.997/1.003. This is the packed matrix-free residual, the standing performance gate, so the
# claim has to survive a proper measurement before it is believed: 5.4% sits only just above the
# 4.3% run-to-run band established for this benchmark, and two passes is not enough to separate
# them. Five passes, best of each, with the same control.

set -uo pipefail
cd "$SLURM_SUBMIT_DIR" || exit 1
source scripts/alps_env.sh || exit 1

BIN_DIR=${BIN_DIR:-$SCRATCH/cvfem_fuse_res}
OUT=${OUT:-$BIN_DIR/${SLURM_JOB_ID:-manual}}
mkdir -p "$OUT"
CSV=$OUT/fuseres.csv
N=${N:-128}
REPEAT=${REPEAT:-10}
PASSES=${PASSES:-5}
PACK=${PACK:-1024}

for s in s0 s1 s2; do
    [ -x "$BIN_DIR/bench_$s" ] || { echo "no binary $BIN_DIR/bench_$s" >&2; exit 1; }
done

run_arm() {   # lbl layout shapes extra...
    local lbl=$1 lay=$2 shapes=$3 extra=$4
    local layflag="--layout $lay"
    [ -n "$PACK" ] && [ "$lay" = "packed" ] && layflag="$layflag --pack-size $PACK"
    local -A best=(); local dofs=""
    for shape in $shapes; do
        local B=$BIN_DIR/bench_$shape b=""
        for pass in $(seq 1 "$PASSES"); do
            local log=$OUT/${lbl}_${lay}_${shape}_p${pass}.log
            timeout "${RUN_TIMEOUT:-360}" \
              srun -u --ntasks=1 --cpus-per-task=72 --uenv="$CVFEM_UENV" --view="$CVFEM_VIEW" \
              bash -lc "cd $PWD && OMP_NUM_THREADS=72 OMP_PROC_BIND=true OMP_PLACES=cores $B --n $N --repeat $REPEAT --warmup 3 $layflag $extra --csv $CSV --tag ${lbl}_${lay}_${shape}_p${pass}" \
              > "$log" 2>&1
            local v; v=$(sed -n 's|^  MDOF/s: \(.*\)$|\1|p' "$log" | head -1)
            [ -n "$v" ] || continue
            b=$(awk -v a="${b:-0}" -v x="$v" 'BEGIN{print (x+0>a+0)?x:a}')
            [ -n "$dofs" ] || dofs=$(sed -n 's|^  dofs: \(.*\)$|\1|p' "$log" | head -1)
        done
        best[$shape]=$b
    done
    local u=${best[s0]:-} f1=${best[s1]:-} f2=${best[s2]:-} r1="-" r2="-"
    [ -n "$u" ] && [ -n "$f1" ] && r1=$(awk -v a="$u" -v b="$f1" 'BEGIN{printf "%.3f", b/a}')
    [ -n "$u" ] && [ -n "$f2" ] && r2=$(awk -v a="$u" -v b="$f2" 'BEGIN{printf "%.3f", b/a}')
    printf '%-18s %-8s %11s %11s %7s %11s %7s %11s\n' "$lbl" "$lay" "${u:--}" "${f1:--}" "$r1" "${f2:--}" "$r2" "${dofs:--}"
}

echo "### host $(hostname)  n=$N repeat=$REPEAT passes=$PASSES  $(date '+%F %T')  affine geometry"
echo
printf '%-18s %-8s %11s %11s %7s %11s %7s %11s\n' arm layout s0_perface s1_fused s1/s0 s2_locals s2/s0 dofs

run_arm residual_fo   packed "s0 s1 s2" "--rhie-chow"
run_arm residual_fo   atomic "s0 s1 s2" "--rhie-chow"
run_arm jac_exact     packed "s0 s1 s2" "--jac-action --rhie-chow"

echo
echo "jac_exact is the control: its face loop was not touched here, so it should read ~1.00."
echo "residual_ho_gen runs the generated whole-element kernels, which have no face loop to fuse;"
echo "it is reported on s0 only, as the incumbent the sumfact HO residual should be read against."
echo "csv: $CSV"
