#!/bin/bash
#SBATCH --job-name=prony-la-bench
#SBATCH --account=c40
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --output=/ritom/scratch/cscs/zulianp/sfem-prony-md/runs/prony_la_bench_%j.out
#SBATCH --uenv=prgenv-gnu/24.11:v2
#SBATCH --view=default
# The history formats' bytes, step cost and error (drivers/bench/bench_prony_history: its synthetic
# stress), without and with the remainder form (horizon 0.1655 s, LDP24A's strokes), on Grace (72
# threads, bound) and one H100, with the series of LDP24A's case file (LDP24A/sim2.yaml: the series
# fitted to 70 EPDM 281, three stored terms at 0.1 ms, one of them a remainder). The first pass runs
# every format, quant_block at its default tolerance; then quant_block alone at each tolerance of
# TOLERANCES (q5, q6, q7) with nearest and stochastic rounding (the rows' last two columns). Shapes:
# LDP24A's TET4 (742,862 one-point elements), eight points (HEX8), eleven (TET10) and 64 (HEX27).
# The output is this job's log. The machine: ../../machine.sh.
#   [CASE_YAML=LDP24A/sim2.yaml] [TOLERANCES="1e-5 1e-6 1e-7"]
SFEM_ROOT=${SFEM_ROOT:-/ritom/scratch/cscs/zulianp/sfem-prony-md}
. "$SFEM_ROOT/workflows/ldp_containment/machine.sh"
B=${SFEM_BUILD:-$SFEM_ROOT/build-cuda}/bench_prony_history
Y=${CASE_YAML:-$SFEM_ROOT/workflows/ldp_containment/transient/LDP24A/sim2.yaml}
# The case's series: material.relaxation.iso.prony, weights and times as two lists.
read -r SFEM_PRONY_G SFEM_PRONY_TAU < <($SFEM_PY - "$Y" <<'PYEOF'
import sys, yaml
terms = yaml.safe_load(open(sys.argv[1]))["material"]["relaxation"]["iso"]["prony"]
print(",".join("%.17g" % t["g"] for t in terms), ",".join("%.17g" % t["tau"] for t in terms))
PYEOF
)
[ -n "$SFEM_PRONY_G" ] || { echo "prony_la_bench: no series in $Y"; exit 1; }
export SFEM_PRONY_G SFEM_PRONY_TAU
echo "## series from $Y: g $SFEM_PRONY_G tau $SFEM_PRONY_TAU"
for space in host device; do
  for shape in "742862 1 30 16" "250000 8 30 0" "250000 11 30 0" "40000 64 30 0"; do
    for horizon in 0 0.1655; do
      echo "## $space $shape $horizon"
      SFEM_EXECUTION_SPACE=$space $SFEM_LAUNCH_GPU $B $shape $horizon 2>&1 | grep -E "^format|,host,|,device,|rror"
      for tol in ${TOLERANCES:-1e-5 1e-6 1e-7}; do
        for rounding in nearest stochastic; do
          echo "## $space $shape $horizon quant_block $tol $rounding"
          SFEM_EXECUTION_SPACE=$space SFEM_HISTORY_TOLERANCE=$tol SFEM_HISTORY_ROUNDING=$rounding \
            $SFEM_LAUNCH_GPU $B $shape $horizon 2>&1 | grep -E "^quant_block,|rror"
        done
      done
    done
  done
done
