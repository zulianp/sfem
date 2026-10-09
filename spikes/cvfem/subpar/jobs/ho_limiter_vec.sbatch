#!/usr/bin/env bash
#SBATCH --account=c40
#SBATCH --job-name=cvfem-holimvec
#SBATCH --partition=debug
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=72
#SBATCH --time=00:25:00
#SBATCH --exclusive --hint=nomultithread --uenv-passthrough=ignore
#SBATCH --output=holimvec_%j.out
#SBATCH --error=holimvec_%j.err

# WHAT THE UNVECTORISED VENKATAKRISHNAN KERNEL COSTS.
#
# GCC's vectoriser report on the driver says the four generated higher-order residual kernels
# split three to one: defcor_lim0, lim1 and lim3 each report "loop vectorized using 16 byte
# vectors" with no missed message, while lim2 -- Venkatakrishnan -- reports none and instead
# "not vectorized: relevant stmt not supported: iftmp = _c ? iftmp : iftmp", then "couldn't
# vectorize loop". Both lim2 variants are hit, with and without Rhie-Chow, and nothing else in
# the generated file is.
#
# The cause is in the emitted expression, not the loop: SymPy writes a ternary whose CONDITION
# is itself a ternary -- ((c) ? (...) : (expr != 0)) ? A : B -- so GCC if-converts the inner one
# into a PHI and the outer select over PHI operands is not a statement it can vectorise.
#
# This job measures the consequence. If the report is right, lim2 should sit well below its
# neighbours on both layouts, and the gap should not be explained by its flop count alone --
# which is why GFLOP/s_model is reported beside MDOF/s.

set -uo pipefail
cd "$SLURM_SUBMIT_DIR" || exit 1
source scripts/alps_env.sh || exit 1

B=$PWD/build/cvfem_hex8_ns_upwind_bench
[ -x "$B" ] || { echo "no bench at $B" >&2; exit 1; }
OUT=${OUT:-$SCRATCH/cvfem_holimvec/${SLURM_JOB_ID:-manual}}
mkdir -p "$OUT"
CSV=$OUT/holimvec.csv
N=${N:-128}
REPEAT=${REPEAT:-10}
PASSES=${PASSES:-3}
PACK=${PACK:-1024}

echo "### host $(hostname)  n=$N repeat=$REPEAT passes=$PASSES  $(date '+%F %T')  affine"
echo "### limiter 0 unlimited, 1 bounded-face clip, 2 Venkatakrishnan, 3 Darwish-Moukalled"
echo
printf '%-8s %-10s %12s %14s %12s\n' limiter layout MDOF_s GFLOP_s_model dofs

for lay in packed atomic; do
    for L in 0 1 2 3; do
        layflag="--layout $lay"
        [ "$lay" = "packed" ] && layflag="$layflag --pack-size $PACK"
        best=""; gf=""; dofs=""
        for pass in $(seq 1 "$PASSES"); do
            log=$OUT/lim${L}_${lay}_p${pass}.log
            timeout "${RUN_TIMEOUT:-360}" \
              srun -u --ntasks=1 --cpus-per-task=72 --uenv="$CVFEM_UENV" --view="$CVFEM_VIEW" \
              bash -lc "cd $PWD && OMP_NUM_THREADS=72 OMP_PROC_BIND=true OMP_PLACES=cores $B --n $N --repeat $REPEAT --warmup 3 $layflag --rhie-chow --conv-ho $L --csv $CSV --tag ho_lim${L}_${lay}_p${pass}" \
              > "$log" 2>&1
            v=$(sed -n 's|^  MDOF/s: \(.*\)$|\1|p' "$log" | head -1)
            [ -n "$v" ] || continue
            newbest=$(awk -v a="${best:-0}" -v x="$v" 'BEGIN{print (x+0>a+0)?x:a}')
            if [ "$newbest" != "${best:-}" ]; then
                gf=$(sed -n 's|^  GFLOP/s_model: \(.*\)$|\1|p' "$log" | head -1)
            fi
            best=$newbest
            [ -n "$dofs" ] || dofs=$(sed -n 's|^  dofs: \(.*\)$|\1|p' "$log" | head -1)
        done
        printf '%-8s %-10s %12s %14s %12s\n' "$L" "$lay" "${best:--}" "${gf:--}" "${dofs:--}"
    done
done
echo
echo "csv: $CSV"
