#!/usr/bin/env bash
#SBATCH --account=c40
#SBATCH --job-name=cvfem-fuseres
#SBATCH --partition=debug
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=72
#SBATCH --time=00:29:00
#SBATCH --exclusive --hint=nomultithread --uenv-passthrough=ignore
#SBATCH --output=fuseres_%j.out
#SBATCH --error=fuseres_%j.err

# THE SAME FUSION QUESTION, ASKED OF THE RESIDUAL -- AND OF THE HIGHER-ORDER RESIDUAL.
#
# The Jacobian's twelve convective faces were measured fused and per-face, and per-face won by 9
# to 22%. That says nothing about the residual: it is a different kernel with a smaller live set
# (no direction field, no correction derivative), so the register pressure that sank the fused
# Jacobian may not bite here. A result on one kernel is not a result on the other.
#
# The higher-order residual is the arm that prompted this, and it matters which path it takes.
# At --kernel sumfact it goes through cvfem_hex8_conv_all_simd, the twelve per-face loops, and is
# therefore fusion-affected. At --kernel sympy it goes to the generated per-limiter kernels
# instead, which are whole-element bodies with no face loop to fuse -- so those arms are
# fusion-independent and serve as a second control alongside the Jacobian.

set -uo pipefail
cd "$SLURM_SUBMIT_DIR" || exit 1
source scripts/alps_env.sh || exit 1

BIN_DIR=${BIN_DIR:-$SCRATCH/cvfem_fuse_res}
OUT=${OUT:-$BIN_DIR/${SLURM_JOB_ID:-manual}}
mkdir -p "$OUT"
CSV=$OUT/fuseres.csv
N=${N:-128}
REPEAT=${REPEAT:-10}
PASSES=${PASSES:-2}
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

run_arm residual_ho   packed "s0 s1 s2" "--kernel sumfact --rhie-chow --conv-ho 0"
run_arm residual_ho   atomic "s0 s1 s2" "--kernel sumfact --rhie-chow --conv-ho 0"
run_arm residual_fo   packed "s0 s1 s2" "--kernel sumfact --rhie-chow"
run_arm residual_fo   atomic "s0 s1 s2" "--kernel sumfact --rhie-chow"
run_arm jac_exact     packed "s0 s1 s2" "--kernel sumfact --jac-action --rhie-chow"
run_arm residual_ho_gen packed "s0" "--kernel sympy --rhie-chow --conv-ho 0"
run_arm residual_ho_gen atomic "s0" "--kernel sympy --rhie-chow --conv-ho 0"

echo
echo "jac_exact is the control: its face loop was not touched here, so it should read ~1.00."
echo "residual_ho_gen runs the generated whole-element kernels, which have no face loop to fuse;"
echo "it is reported on s0 only, as the incumbent the sumfact HO residual should be read against."
echo "csv: $CSV"
