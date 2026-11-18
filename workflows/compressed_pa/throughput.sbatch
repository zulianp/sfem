#!/bin/bash
#SBATCH --job-name=cpa-throughput
#SBATCH --account=c40
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --output=/ritom/scratch/cscs/zulianp/sfem-prony-md/runs/cpa_throughput_%j.out
#SBATCH --uenv=prgenv-gnu/24.11:v2
#SBATCH --view=default
# The compressed partial assembly paper's operator throughput (drivers/bench/bench_inexact): per element,
# the paper's formats (BSR f64 and f32, the quantized matrix, the element matrix, the point store), set-up and apply, MDOF/s, bytes
# per dof and the distance to the exact apply, on the host (one Grace chip per run, four elements at
# once) or the device (one GPU per run). The paper's sizes in dof (HEX8 n=150 and PROTEUS_HEX27 n=75:
# 10,328,853 dof; TET4 n=119 and TET10 n=60, about 10.2M and 10.5M: the library's cube has 12 tetrahedra
# and a centre node per cell, the paper's benchmark cube 6, its n=150 and 75), or with SWEEP=1 its sweep
# of sizes (the tetrahedral n as the paper's, so about twice its dof). Packs as the paper's runs:
# max(16, min(8192, elements/1440)) rounded down to 16, two-pass scatter. Each run's output is
# OUT/<element>_n<n>.out, OUT a new folder (the job stops if it exists). The machine: ../ldp_containment/machine.sh.
#   [SPACE=host|device] [ELEMENTS="TET4 TET10 HEX8 PROTEUS_HEX27"] [SWEEP=1] [REPEATS=10] [OUT=runs/cpa_throughput_<space>]
set -u
SFEM_ROOT=${SFEM_ROOT:-/ritom/scratch/cscs/zulianp/sfem-prony-md}
. "$SFEM_ROOT/workflows/ldp_containment/machine.sh"
SPACE=${SPACE:-host}
B=${SFEM_BUILD:-$SFEM_ROOT/build-cuda}/bench_inexact
OUT=${OUT:-$SFEM_RUNS/cpa_throughput_$SPACE}
[ -x "$B" ] || { echo "throughput: no $B"; exit 1; }
[ -e "$OUT" ] && { echo "throughput: $OUT exists; give a new OUT"; exit 1; }
mkdir -p "$OUT"
export SFEM_PACKED_TWO_PASS=1
# The paper's formats (and the double-precision point store of its correctness check); ROWS overrides
# them (bench_inexact's row names, comma-separated).
export SFEM_BENCH_ROWS=${ROWS:-BSR f64,BSR f32,SELL f16,SELL i8,element_matrix f16,element_matrix i8pair,points f64,points f16,points i8md}
sizes() {  # <element>: the paper's size, or its sweep
  case $1 in
    HEX8) [ "${SWEEP:-0}" = 1 ] && echo 16 24 36 50 70 100 150 200 || echo 150 ;;
    TET4) [ "${SWEEP:-0}" = 1 ] && echo 16 24 36 50 70 100 150 200 || echo 119 ;;
    TET10) [ "${SWEEP:-0}" = 1 ] && echo 8 12 18 25 35 50 75 100 || echo 60 ;;
    *) [ "${SWEEP:-0}" = 1 ] && echo 8 12 18 25 35 50 75 100 || echo 75 ;;
  esac
}
go() {  # <element> <n>
  local e=$1 n=$2 ne p
  case $e in TET4 | TET10) ne=$((12 * n * n * n)) ;; *) ne=$((n * n * n)) ;; esac
  p=$((ne / 1440 / 16 * 16)); [ $p -gt 8192 ] && p=8192; [ $p -lt 16 ] && p=16
  {
    echo "node $(hostname) $e n=$n repeats=${REPEATS:-10} space=$SPACE packs of $p"
    if [ "$SPACE" = device ]; then
      SFEM_EXECUTION_SPACE=device SMESH_ELEMENTS_PER_PACK=$p $SFEM_LAUNCH_GPU "$B" "$e" "$n" "${REPEATS:-10}"
    else
      SMESH_ELEMENTS_PER_PACK=$p $SFEM_LAUNCH_HOST "$B" "$e" "$n" "${REPEATS:-10}"
    fi
    echo "rc=$?"
  } > "$OUT/${e}_n$n.out" 2>&1
  echo "$e n=$n $(tail -1 "$OUT/${e}_n$n.out")"
}
for e in ${ELEMENTS:-TET4 TET10 HEX8 PROTEUS_HEX27}; do
  for n in $(sizes "$e"); do
    sfem_throttle
    go "$e" "$n" &
  done
done
wait
