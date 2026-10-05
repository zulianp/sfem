#!/usr/bin/env bash
#SBATCH --account=c40
#SBATCH --job-name=cvfem-defcor-arms
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --time=00:28:00
#SBATCH --exclusive
#SBATCH --hint=nomultithread
#SBATCH --uenv-passthrough=ignore
#SBATCH --output=cvfem_defcor_arms_%j.out
#SBATCH --error=cvfem_defcor_arms_%j.err
#
# WHICH DEFERRED-CORRECTION MICRO-KERNEL SURVIVES.
#
# DESIGN.md's correction says one micro-kernel per kernel, the one Grace measured fastest. For
# the packed higher-order residual there are three arms and the campaign settles none of them:
#
#   --conv-ho L              the GENERATED lane-blocked kernel (default where it applies)
#   --conv-ho L --ho-simd    the HAND-WRITTEN lane-blocked kernel
#   --conv-ho L --ho-scalar  the hand-written scalar kernel
#
# The only number on record is "1.39x the scalar one it replaced", which compares the generated
# kernel against the SCALAR arm and says nothing about the hand-written SIMD arm beside it.
# Retiring either of the two SIMD arms on that number would be retiring an unmeasured kernel.
#
# All four limiter arms are covered because the generated kernels are emitted per limiter -- two
# variants each, with and without Rhie-Chow, eight in all -- so a win on the unlimited arm does
# not carry to the limited ones: the limiter is where the generated expression grows.
#
# Interleaved in one allocation and best-of-three, for the reasons conv_ho_bench.sbatch records:
# the node-to-node band is 5-11% and a single pass of this residual has been measured varying by
# far more than the gap this job has to resolve.
#
# SLURM_SUBMIT_DIR, not BASH_SOURCE: Slurm copies the batch script to
# /var/spool/slurmd/job<id>/slurm_script.
set -u
: "${CVFEM_SRC:=${SLURM_SUBMIT_DIR:-$PWD}}"
: "${CVFEM_SFEM_INSTALL:=$SCRATCH/installations/sfem-omp}"
export CVFEM_SRC CVFEM_SFEM_INSTALL
source "$CVFEM_SRC/scripts/alps_env.sh"
cd "$CVFEM_SRC"

B="$CVFEM_SRC/build/cvfem_hex8_ns_upwind_bench"
[ -x "$B" ] || { echo "bench not built: $B" >&2; exit 2; }
OUT="$CVFEM_SRC/defcor_out"; mkdir -p "$OUT"
CSV="$OUT/defcor.csv"
rm -f "$OUT/table.txt"

N=${N:-128}
REPEAT=${REPEAT:-10}
PASSES=${PASSES:-3}
echo "### host $(hostname)  n=$N repeat=$REPEAT passes=$PASSES  $(date '+%F %T')"
echo "### gen = generated lane-blocked defcor; hw = hand-written lane-blocked; sc = hand-written scalar"
echo
printf '%-26s %10s %10s %14s\n' arm MDOF_s dofs checksum

for arm in "lim0_gen|--conv-ho 0" \
           "lim0_hw|--conv-ho 0 --ho-simd" \
           "lim0_sc|--conv-ho 0 --ho-scalar" \
           "lim1_gen|--conv-ho 1" \
           "lim1_hw|--conv-ho 1 --ho-simd" \
           "lim2_gen|--conv-ho 2" \
           "lim2_hw|--conv-ho 2 --ho-simd" \
           "lim3_gen|--conv-ho 3" \
           "lim3_hw|--conv-ho 3 --ho-simd" \
           "rc_lim0_gen|--rhie-chow --conv-ho 0" \
           "rc_lim0_hw|--rhie-chow --conv-ho 0 --ho-simd" \
           "rc_lim0_sc|--rhie-chow --conv-ho 0 --ho-scalar" \
           "rc_lim2_gen|--rhie-chow --conv-ho 2" \
           "rc_lim2_hw|--rhie-chow --conv-ho 2 --ho-simd" \
           "rc_lim3_gen|--rhie-chow --conv-ho 3" \
           "rc_lim3_hw|--rhie-chow --conv-ho 3 --ho-simd"; do
    IFS='|' read -r lbl extra <<<"$arm"
    best=""; dofs=""; ck=""
    for pass in $(seq 1 "$PASSES"); do
        log=$OUT/${lbl}_p${pass}.log
        # A PER-ARM TIMEOUT, so one wedged arm cannot take the whole wall clock with it.
        timeout "${RUN_TIMEOUT:-360}" \
          srun -u --ntasks=1 --cpus-per-task=72 --uenv="$CVFEM_UENV" --view="$CVFEM_VIEW" \
          bash -lc "cd $PWD && OMP_NUM_THREADS=72 OMP_PROC_BIND=true OMP_PLACES=cores $B --n $N --repeat $REPEAT --warmup 3 --layout packed --kernel sumfact $extra --csv $CSV --tag ${lbl}_p${pass}" \
          > "$log" 2>&1
        val() { sed -n "s|^  $1: \(.*\)$|\1|p" "$log" | head -1; }
        v=$(val 'MDOF/s'); [ -n "$v" ] || continue
        best=$(awk -v a="${best:-0}" -v b="$v" 'BEGIN{print (b+0>a+0)?b:a}')
        dofs=$(val 'dofs'); ck=$(val 'checksum')
    done
    # The rate is printed as it arrives, not at the end: these runs are long and a filter that
    # holds the table back until the last arm hides a wedged one until the job is over.
    printf '%-26s %10s %10s %14s\n' "$lbl" "${best:-FAIL}" "${dofs:-?}" "${ck:-?}" | tee -a "$OUT/table.txt"
done
echo
echo "### gen-vs-hw ratios (>1 means the GENERATED kernel wins):"
awk '/^(rc_)?lim[0-9]_(gen|hw)/ {r[$1]=$2} END {
  for (k in r) if (k ~ /_gen$/) { h=k; sub(/_gen$/,"_hw",h);
    if (h in r && r[h]+0 > 0) printf "  %-14s gen=%-9s hw=%-9s ratio=%.3f\n", k, r[k], r[h], r[k]/r[h] }
}' "$OUT/table.txt"
echo "### done $(date '+%F %T')"
