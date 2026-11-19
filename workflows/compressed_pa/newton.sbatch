#!/bin/bash
#SBATCH --job-name=cpa-newton
#SBATCH --account=c40
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=02:00:00
#SBATCH --output=/ritom/scratch/cscs/zulianp/sfem-prony-md/runs/cpa_newton_%j.out
#SBATCH --uenv=prgenv-gnu/24.11:v2
#SBATCH --view=default
# The compressed partial assembly paper's Newton study with the library's playground: the cases of
# spikes/inexact_apply_compare/qp/cases at the paper's sizes and steps, each operator format a run on one
# Grace chip (host) or one GPU (device), SFEM_RUNS_PER_NODE at a time. The tetrahedral meshes are smesh's
# cube (12 tetrahedra and a centre node per cell, the paper's), made once into SFEM_RUNS/cpa_meshes.
#   CASES     stretch_1M stretch_10M twist (default all); sweep: the stretch case at the paper's sweep of
#             sizes (stretch_s0 ... stretch_s4, the 1M case's step), a log per size
#   ELEMENTS  TET4 TET10 HEX8 PROTEUS_HEX27 (default all)
#   FORMATS   the paper's formats, by the paper's names: bsr64 bsr32 (BSR in double and single precision),
#             bsro16 bsroi8 (the quantized matrix, SELL-C-sigma on both machines), ead16 eadi8p (the
#             element matrix, fp16 difference form and int8 with a scale per node pair), f16 i8md (the
#             point store, fp16 and int8 mean and deviation); default all eight
#   SPACE     host | device
#   (each run's section: its steps, the error probe's lines if on, rc, its peak memory, its trace totals)
#   SET       further playground settings, key=value separated by spaces, for every run (e.g.
#             "material.poisson_ratio=0.45 solver.linear.preconditioner=jacobi solver.linear.error_probe=true")
#   DT        one step for every case instead of the paper's (e.g. the baseline's, for the robustness runs)
#   OUT       a new folder (default SFEM_RUNS/cpa_newton_<space>); each case's log OUT/<case>_<element>.out
#             has a section "== <format> <options>" per run with its steps and its trace totals, the run
#             itself in OUT/<case>_<element>/<format>
# The machine: ../ldp_containment/machine.sh.
set -u
SFEM_ROOT=${SFEM_ROOT:-/ritom/scratch/cscs/zulianp/sfem-prony-md}
. "$SFEM_ROOT/workflows/ldp_containment/machine.sh"
SPACE=${SPACE:-host}
B=${SFEM_BUILD:-$SFEM_ROOT/build-cuda}
OUT=${OUT:-$SFEM_RUNS/cpa_newton_$SPACE}
C=$SFEM_ROOT/spikes/inexact_apply_compare/qp/cases
for x in "$B/playground" "$B/external/smesh/cube"; do [ -x "$x" ] || { echo "newton: no $x"; exit 1; }; done
[ -e "$OUT" ] && { echo "newton: $OUT exists; give a new OUT"; exit 1; }
mkdir -p "$OUT"
export SFEM_PACKED_TWO_PASS=1

# case element -> the mesh option and the step (the paper's logs, data/measured/newton_* and twist_*).
setup() {  # <case> <element>: prints "<mesh option> <dt> <yaml>"
  local k=$1 e=$2 n dt
  if [[ $k == stretch_s* ]]; then
    local sizes
    case $e in
      HEX8) sizes="16 24 36 50 70" dt=0.25 ;;
      PROTEUS_HEX27) sizes="8 12 18 25 35" dt=0.125 ;;
      TET4) sizes="t16 t24 t36 t56" dt=0.25 ;;
      TET10) sizes="t8 t12 t18 t28" dt=0.0625 ;;
      *) return 1 ;;
    esac
    n=$(echo $sizes | cut -d' ' -f$((${k#stretch_s} + 1)))
    [ -n "$n" ] || return 1
  else
  case $k:$e in
    stretch_1M:TET4) n=t56 dt=0.25 ;;       stretch_10M:TET4) n=t120 dt=0.125 ;;
    stretch_1M:TET10) n=t28 dt=0.0625 ;;    stretch_10M:TET10) n=t60 dt=0.03125 ;;
    stretch_1M:HEX8) n=70 dt=0.25 ;;        stretch_10M:HEX8) n=150 dt=0.25 ;;
    stretch_1M:PROTEUS_HEX27) n=35 dt=0.125 ;; stretch_10M:PROTEUS_HEX27) n=75 dt=0.03125 ;;
    twist:TET4) n=t56 dt=0.0078125 ;;       twist:TET10) n=t28 dt=0.00390625 ;;
    twist:HEX8) n=70 dt=0.0625 ;;           twist:PROTEUS_HEX27) n=35 dt=0.03125 ;;
    *) return 1 ;;
  esac
  fi
  local y=$C/$(echo "$e" | tr A-Z a-z | sed s/proteus_//)_${k%%_*}
  [ "${k%%_*}" = stretch ] && y=${y}_shear
  case $n in
    t*) local d; d=$(mesh "${n#t}") || return 1; echo "mesh.path=$d $dt $y.yaml" ;;
    *) echo "mesh.box.n=[$n,$n,$n] $dt $y.yaml" ;;
  esac
}
mesh() {  # <n>: smesh's tetrahedral cube of n^3 cells, made once (jobs at once: the lock's holder makes it)
  local d=$SFEM_RUNS/cpa_meshes/tet4_$1
  mkdir -p "$SFEM_RUNS/cpa_meshes"
  if [ ! -f "$d/meta.yaml" ]; then
    if mkdir "$d.lock" 2> /dev/null; then
      $SFEM_LAUNCH_HOST "$B/external/smesh/cube" TET4 "$1" "$1" "$1" 0 0 0 1 1 1 "$d.part" > "$d.log" 2>&1 \
        && mv "$d.part" "$d"
      rmdir "$d.lock"
    else
      while [ -d "$d.lock" ]; do sleep 10; done
    fi
    [ -f "$d/meta.yaml" ] || { echo "newton: cube $1 failed ($d.log)" >&2; return 1; }
  fi
  echo "$d"
}
options() {  # <format>: the playground options
  case $1 in
    bsr64) echo "--set solver.linear.format=BSR" ;;
    bsr32) echo "--set solver.linear.format=BSR --set solver.linear.storage_bytes=4" ;;
    f16 | i8md) echo "--set solver.linear.format=INEXACT --set solver.linear.inexact.store=points --set solver.linear.inexact.encoding=$1" ;;
    ead16) echo "--set solver.linear.format=INEXACT --set solver.linear.inexact.store=element_matrix --set solver.linear.inexact.encoding=f16" ;;
    eadi8p) echo "--set solver.linear.format=INEXACT --set solver.linear.inexact.store=element_matrix --set solver.linear.inexact.encoding=i8pair" ;;
    bsro16) echo "--set solver.linear.format=SELL --set solver.linear.sell.encoding=f16" ;;
    bsroi8) echo "--set solver.linear.format=SELL --set solver.linear.sell.encoding=i8" ;;
    *) return 1 ;;
  esac
}
pack() {  # <mesh option>: the paper's pack size, max(16, min(8192, elements/1440)) rounded down to 16
  local ne n p
  case $1 in
    mesh.box.n=*) n=${1#*[}; n=${n%%,*}; ne=$((n * n * n)) ;;
    *) ne=$(awk '/n_elements:/{print $2}' "${1#mesh.path=}/meta.yaml") ;;
  esac
  p=$((ne / 1440 / 16 * 16)); [ $p -gt 8192 ] && p=8192; [ $p -lt 16 ] && p=16
  echo $p
}
FORMATS=${FORMATS:-bsr64 bsr32 bsro16 bsroi8 ead16 eadi8p f16 i8md}
CASES=${CASES:-stretch_1M stretch_10M twist}
CASES=${CASES//sweep/stretch_s0 stretch_s1 stretch_s2 stretch_s3 stretch_s4}
# Every run records its memory at each step's end (the playground's memory probe; host_peak_rss is the
# process's high-water mark, device_pool_peak the stream-ordered pools'), its section ends with the peaks.
EXTRA=()
for kv in ${SET:-}; do EXTRA+=(--set "$kv"); done

go() {  # <case> <element> <format>: one run into its folder, its section into a part log
  local k=$1 e=$2 f=$3 s m dt y o d p
  s=$(setup "$k" "$e") || return 0
  read -r m dt y <<< "$s"
  dt=${DT:-$dt}
  o=$(options "$f") || { echo "newton: no format $f" >&2; return 0; }
  d=$OUT/${k}_$e/$f
  mkdir -p "$d"
  p=$(pack "$m")
  # The playground resolves output paths against the case file's folder: the run's own, absolute.
  local mem=(--set "output.history_csv=$d/history.csv" --set "output.probes=[{type: memory}]")
  {
    echo "== $f $o ${EXTRA[*]}"
    # shellcheck disable=SC2086
    if [ "$SPACE" = device ]; then
      (cd "$d" && SMESH_ELEMENTS_PER_PACK=$p $SFEM_LAUNCH_GPU "$B/playground" "$y" --set "$m" --set time.dt=$dt --set execution_space=device $o "${mem[@]}" "${EXTRA[@]}" 2>&1)
    else
      (cd "$d" && SMESH_ELEMENTS_PER_PACK=$p $SFEM_LAUNCH_HOST "$B/playground" "$y" --set "$m" --set time.dt=$dt --set execution_space=host $o "${mem[@]}" "${EXTRA[@]}" 2>&1)
    fi | grep -E "^Playground|^step [0-9]|^error_probe|Abort|rror"
    echo "rc=${PIPESTATUS[0]}"
    [ -f "$d/history.csv" ] && awk -F, 'NR == 1 { for (i = 1; i <= NF; i++) c[$i] = i; next }
      { for (k in c) if (k ~ /peak|_now$|^bytes_tangent/ && $c[k] + 0 > m[k]) m[k] = $c[k] + 0 }
      END { printf "memory"; for (k in m) printf " %s %.0f", k, m[k]; printf "\n" }' "$d/history.csv"
    [ -f "$d/smesh.trace.csv" ] && grep -E "^(NewtonSolver::(solve|step)|Function::inexact_(update|apply)|ConjugateGradient::apply|BSRTangent::assemble|SELLTangent::assemble)," "$d/smesh.trace.csv" | sed 's/^/  trace /'
  } > "$d.part" 2>&1
}
for k in $CASES; do
  for e in ${ELEMENTS:-TET4 TET10 HEX8 PROTEUS_HEX27}; do
    setup "$k" "$e" > /dev/null || continue
    for f in $FORMATS; do
      [ "$e" = TET4 ] && [ "$f" = i8md ] && continue  # not in the paper
      sfem_throttle
      go "$k" "$e" "$f" &
    done
  done
done
wait
# One log per case and element, its header first (as the paper's newton.sbatch logs).
for k in $CASES; do
  for e in ${ELEMENTS:-TET4 TET10 HEX8 PROTEUS_HEX27}; do
    s=$(setup "$k" "$e") || continue
    read -r m dt y <<< "$s"
    dt=${DT:-$dt}
    {
      echo "node $(hostname) $e $y $m dt=$dt space=$SPACE ${SET:-}"
      for f in $FORMATS; do [ -f "$OUT/${k}_$e/$f.part" ] && cat "$OUT/${k}_$e/$f.part"; done
    } > "$OUT/${k}_$e.out"
    echo "${k}_$e: $(grep -c '^rc=0' "$OUT/${k}_$e.out") runs ok of $(grep -c '^== ' "$OUT/${k}_$e.out")"
  done
done
