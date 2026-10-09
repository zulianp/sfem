#!/usr/bin/env bash
#SBATCH --account=c40
#SBATCH --job-name=cvfem-gradprec
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --time=00:25:00
#SBATCH --exclusive
#SBATCH --hint=nomultithread
#SBATCH --uenv-passthrough=ignore
#SBATCH --output=gradprec_%j.out
#SBATCH --error=gradprec_%j.err
# THE GRADIENT FIELDS' STORAGE PRECISION, double against single, interleaved in one allocation.
# Same binary, two settings of --grad-precision, order alternating every pass, medians of 5, at
# the paper's n=128. Every arm reads the nodal velocity gradient; the Jacobian arms read two.
set -uo pipefail
cd "$SLURM_SUBMIT_DIR" || exit 1
source scripts/alps_env.sh || exit 1
B=$PWD/build/cvfem_hex8_ns_upwind_bench
[ -x "$B" ] || { echo "no bench at $B" >&2; exit 1; }
export OMP_NUM_THREADS=72 OMP_PROC_BIND=true OMP_PLACES=cores
N=${N:-128}; REPEAT=${REPEAT:-10}; PASSES=${PASSES:-5}
echo "### host $(hostname)  n=$N repeat=$REPEAT passes=$PASSES  $(date '+%F %T')"
declare -A V
ARMS=("res_ho_unlim|packed|--rhie-chow --conv-ho 0" "res_ho_venk|packed|--rhie-chow --conv-ho 2" "res_ho_dm|packed|--rhie-chow --conv-ho 3" \
      "jac_ho_unlim|packed|--jac-action --rhie-chow --conv-ho 0" "jac_ho_venk|packed|--jac-action --rhie-chow --conv-ho 2" "jac_ho_dm|packed|--jac-action --rhie-chow --conv-ho 3" \
      "res_ho_venk|atomic|--rhie-chow --conv-ho 2" "jac_ho_venk|atomic|--jac-action --rhie-chow --conv-ho 2")
for pass in $(seq 1 $PASSES); do
  for arm in "${ARMS[@]}"; do
    IFS='|' read -r key lay extra <<<"$arm"
    if [ $((pass % 2)) -eq 1 ]; then order="double single"; else order="single double"; fi
    for gp in $order; do
      v=$(cvfem_uenv $B --n $N --repeat $REPEAT --warmup 3 --layout $lay $extra --grad-precision $gp 2>&1 | sed -n 's/^  MDOF\/s: //p' | head -1)
      echo "pass$pass $key $lay $gp $v"
      V["$key|$lay|$gp"]+="$v "
    done
  done
done
echo
printf '%-14s %-7s %9s %9s %8s\n' arm layout f64 f32 f32/f64
median() { printf '%s\n' "$@" | sort -g | awk '{a[NR]=$1} END {print (NR%2)?a[(NR+1)/2]:(a[NR/2]+a[NR/2+1])/2}'; }
for arm in "${ARMS[@]}"; do
  IFS='|' read -r key lay extra <<<"$arm"
  m64=$(median ${V["$key|$lay|double"]}); m32=$(median ${V["$key|$lay|single"]})
  printf '%-14s %-7s %9.1f %9.1f %8.3f\n' "$key" "$lay" "$m64" "$m32" "$(awk -v a=$m64 -v b=$m32 'BEGIN{print b/a}')"
done
