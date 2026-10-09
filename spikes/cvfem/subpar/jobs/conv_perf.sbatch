#!/usr/bin/env bash
#SBATCH --account=c40
#SBATCH --job-name=cvfem-convperf
#SBATCH --partition=debug
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=72
#SBATCH --time=00:29:00
#SBATCH --exclusive --hint=nomultithread --uenv-passthrough=ignore
#SBATCH --output=convperf_%j.out
#SBATCH --error=convperf_%j.err

# WHERE the deferred correction's 3.16x goes, as opposed to how big it is.
#
# The cost is measured: first-order residual 514.1 MDOF/s against 144.2 for the bounded
# higher-order arm at 1,098,500 dof, and only 8-15% of that gap is the limiter itself. The
# rest is the correction, and nothing so far says which part of it -- the extra nodal
# velocity-gradient sweep, the per-sub-control-surface reconstruction, or the atomic scatter
# the path is forced onto. Guessing between those would pick the wrong thing to optimise.
#
# perf_event_paranoid is 2 here, so user-space sampling works and kernel symbols do not
# resolve. That is enough: everything of interest is in the element loops.
#
# One arm per record, because a profile that mixes them cannot attribute anything.

set -uo pipefail
cd "$SLURM_SUBMIT_DIR" || exit 1
source scripts/alps_env.sh || exit 1

D=$PWD/build_solver/cvfem_hex8_ns_ssgmg
OUT=${OUT:-$SCRATCH/cvfem_convperf/${SLURM_JOB_ID:-manual}}
mkdir -p "$OUT"
BASE="SFEM_CASE=cavity SFEM_FGMRES=1 SFEM_GMG=1 SFEM_ENABLE_OUTPUT=0 SFEM_N=16 SFEM_ELEMENT_REFINE_LEVEL=4 SFEM_MU=0.01 SFEM_NL_MAX_IT=8"

for a in "lo|SFEM_CONV_HO=0|" "ho_venkat|SFEM_CONV_HO=1|SFEM_CONV_LIMITER=2"; do
    IFS='|' read -r lbl e1 e2 <<<"$a"
    echo "############ $lbl"
    timeout 600 srun -u --ntasks=1 --cpus-per-task=72 --uenv="$CVFEM_UENV" --view="$CVFEM_VIEW" \
      bash -lc "cd $PWD && OMP_NUM_THREADS=72 OMP_PROC_BIND=true OMP_PLACES=cores \
                env $BASE $e1 ${e2:-SFEM_DUMMY=0} SFEM_CONV_FREEZE=0 \
                perf record -F 499 --call-graph=dwarf,4096 -o $OUT/$lbl.data -- $D $OUT/out_$lbl" \
      > "$OUT/$lbl.log" 2>&1
    echo "--- flat profile, self time"
    srun -u --ntasks=1 --cpus-per-task=72 --uenv="$CVFEM_UENV" --view="$CVFEM_VIEW" \
      bash -lc "perf report -i $OUT/$lbl.data --stdio --no-children --percent-limit 1.2 2>/dev/null | grep -aE '^ +[0-9]' | head -18" \
      2>/dev/null || echo "(report failed)"
    echo
done
echo "profiles in $OUT"
