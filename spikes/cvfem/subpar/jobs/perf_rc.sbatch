#!/usr/bin/env bash
#SBATCH --account=c40
#SBATCH --job-name=cvfem-perf-rc
#SBATCH --partition=debug
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=72
#SBATCH --time=00:29:00
#SBATCH --exclusive --hint=nomultithread --uenv-passthrough=ignore
#SBATCH --output=perf_rc_%j.out
#SBATCH --error=perf_rc_%j.err

# WHERE THE RHIE-CHOW RESIDUAL'S TIME GOES. (the 17% the stale baseline flagged)
#
# The finding this investigates: residual_packed_rc reads 1404.7 MDOF/s on HEAD against 1681.4
# in perf/baseline_grace.csv (recorded 2026-09-10, commit 735643efa), and the same number came
# back on two different nodes on two different days -- 1388.4 and 1404.7, 1.2% apart -- so it is
# not node variation. An interleaved A/B cleared this session's uncommitted work at 0.6%, which
# places the change in the COMMITTED history between 735643efa and a3253433f.
#
# The shape of it is the interesting part and what this job is pointed at. The two
# hoisted-gradient RC residual configurations lost 17%, while residual_packed_rc_perapply gained
# 64% and both Jacobian-action RC configurations gained 42-47%. RC-heavy paths slower,
# gradient-heavy path much faster, is what one would see if the Rhie-Chow term itself became
# more expensive while the nodal gradient sweep became cheaper.
#
# The standing hypothesis is that this is the velocity sensitivity added to the Rhie-Chow
# coefficient, which is a correctness fix and worth paying for. A profile can support or refute
# that directly: if it is right, the RC coefficient's own symbols carry the added time, and the
# gradient sweep is a smaller share than it used to be.
#
# THIS JOB CANNOT ATTRIBUTE ON ITS OWN. It profiles one binary, so it says where the time is
# now, not what changed. Confirming the cause needs the same profile from 735643efa beside it.
# Said here so the output is not over-read.
#
# perf_event_paranoid is 2 on these nodes, so user-space sampling works and kernel symbols do
# not resolve. Everything of interest is in the element loops.

set -uo pipefail
cd "$SLURM_SUBMIT_DIR" || exit 1
source scripts/alps_env.sh || exit 1

B=$PWD/build/cvfem_hex8_ns_upwind_bench
[ -x "$B" ] || { echo "no bench at $B" >&2; exit 1; }
OUT=${OUT:-$SCRATCH/cvfem_perf_rc/${SLURM_JOB_ID:-manual}}
mkdir -p "$OUT"

N=${N:-128}
REPEAT=${REPEAT:-30}

echo "### host $(hostname)  n=$N  repeat=$REPEAT  $(date '+%F %T')"
echo

# Three arms, because the contrast is the evidence, not any single profile:
#   rc          the configuration that lost 17%   (hoisted gradient, RC on)
#   rc_perapply the one that gained 64%           (gradient rebuilt every apply)
#   bare        the control, unchanged at -1.3%   (no RC, no gradient)
for arm in "bare|" \
           "rc|--rhie-chow" \
           "rc_perapply|--rhie-chow --pgrad-per-apply"; do
    IFS='|' read -r lbl extra <<<"$arm"
    echo "############ $lbl  ${extra:-(no extra flags)}"
    timeout 600 \
      srun -u --ntasks=1 --cpus-per-task=72 --uenv="$CVFEM_UENV" --view="$CVFEM_VIEW" \
      bash -lc "cd $PWD && OMP_NUM_THREADS=72 OMP_PROC_BIND=true OMP_PLACES=cores \
                perf record -F 499 --call-graph=dwarf,4096 -o $OUT/$lbl.data -- \
                $B --n $N --repeat $REPEAT --warmup 3 --layout packed $extra" \
      > "$OUT/$lbl.log" 2>&1
    if [ ! -s "$OUT/$lbl.data" ]; then
        echo "  NO PROFILE RECORDED -- tail of $lbl.log:"; tail -6 "$OUT/$lbl.log" | sed 's/^/    /'
        continue
    fi
    echo "--- MDOF/s: $(sed -n 's|^  MDOF/s: \(.*\)$|\1|p' "$OUT/$lbl.log" | head -1)"
    echo "--- flat profile, self time, cvfem symbols only"
    srun -u --ntasks=1 --cpus-per-task=72 --uenv="$CVFEM_UENV" --view="$CVFEM_VIEW" \
      bash -lc "perf report -i $OUT/$lbl.data --stdio --no-children --percent-limit 0.8 2>/dev/null" \
      2>/dev/null | grep -aE "^ +[0-9]" | head -14
    echo
done

echo "profiles in $OUT"
