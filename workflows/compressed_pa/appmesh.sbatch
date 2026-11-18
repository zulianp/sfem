#!/bin/bash
#SBATCH --job-name=cpa-appmesh
#SBATCH --account=c40
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=00:30:00
#SBATCH --output=/ritom/scratch/cscs/zulianp/sfem-prony-md/runs/cpa_appmesh_%j.out
#SBATCH --uenv=prgenv-gnu/24.11:v2
#SBATCH --view=default
# The compressed partial assembly paper's application benchmark (drivers/bench/bench_inexact MESH) on
# the meshes of two pumps' valves: each mesh folder of MESHES (TET4 as given and refined once, <name>_<level>;
# TET10 promoted, <name>_<level>_tet10), on the host (one Grace chip per run) or the device (one GPU
# per run), SFEM_RUNS_PER_NODE at a time. The meshes must be in a locality order (smesh's sfc), as the
# paper's benchmark ordered them (Morton, nodes renumbered in element order): in the order a mesher
# writes them the scatter runs 2-6 times slower. Packs as the paper's runs; repeats 100 on the meshes as
# given, 20 refined. Each run's output is OUT/<cpu|gpu>_<mesh>.out, OUT a new folder.
#   MESHES=<folder of mesh folders> [SPACE=host|device] [OUT=runs/cpa_appmesh_<space>]
# The machine: ../ldp_containment/machine.sh.
set -u
SFEM_ROOT=${SFEM_ROOT:-/ritom/scratch/cscs/zulianp/sfem-prony-md}
. "$SFEM_ROOT/workflows/ldp_containment/machine.sh"
SPACE=${SPACE:-host}
B=${SFEM_BUILD:-$SFEM_ROOT/build-cuda}/bench_inexact
OUT=${OUT:-$SFEM_RUNS/cpa_appmesh_$SPACE}
[ -x "$B" ] || { echo "appmesh: no $B"; exit 1; }
[ -d "${MESHES:-}" ] || { echo "appmesh: MESHES is not a folder: '${MESHES:-}'"; exit 1; }
[ -e "$OUT" ] && { echo "appmesh: $OUT exists; give a new OUT"; exit 1; }
mkdir -p "$OUT"
export SFEM_PACKED_TWO_PASS=1
# The paper's formats (and the double-precision point store of its correctness check); ROWS overrides
# them (bench_inexact's row names, comma-separated).
export SFEM_BENCH_ROWS=${ROWS:-BSR f64,BSR f32,SELL f16,SELL i8,element_matrix f16,element_matrix i8pair,points f64,points f16,points i8md}
tag=$([ "$SPACE" = device ] && echo gpu || echo cpu)
go() {  # <mesh folder>
  local m=$1 name ne p rep
  name=$(basename "$m")
  ne=$(awk '/n_elements:/{print $2}' "$m/meta.yaml")
  p=$((ne / 1440 / 16 * 16)); [ $p -gt 8192 ] && p=8192; [ $p -lt 16 ] && p=16
  case $name in *_0 | *_0_tet10) rep=100 ;; *) rep=20 ;; esac
  {
    echo "node $(hostname) $name repeats=$rep space=$SPACE packs of $p"
    if [ "$SPACE" = device ]; then
      SFEM_EXECUTION_SPACE=device SMESH_ELEMENTS_PER_PACK=$p $SFEM_LAUNCH_GPU "$B" MESH "$m" "$rep"
    else
      SMESH_ELEMENTS_PER_PACK=$p $SFEM_LAUNCH_HOST "$B" MESH "$m" "$rep"
    fi
    echo "rc=$?"
  } > "$OUT/${tag}_$name.out" 2>&1
  echo "$name $(tail -1 "$OUT/${tag}_$name.out")"
}
for m in "$MESHES"/*/; do
  [ -f "$m/meta.yaml" ] || continue
  sfem_throttle
  go "${m%/}" &
done
wait
