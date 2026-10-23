#!/usr/bin/env bash
set -euo pipefail
binary=${1:?pass the CVFEM benchmark executable}
logs=$(mktemp -d "${TMPDIR:-/tmp}/cvfem-darwish-cache.XXXXXX")
trap 'rm -rf "$logs"' EXIT
export OMP_DYNAMIC=FALSE OPENBLAS_NUM_THREADS=1
for threads in ${CACHE_TEST_THREADS:-1 2}; do
  export OMP_NUM_THREADS=$threads
  for fixture in 0 1 2 3; do
    extra=()
    case $fixture in
      0) n=8; perturb=0.009 ;;
      1) n=9; perturb=0.017 ;;
      2) n=9; perturb=-0.031; extra=(--no-sfc) ;;
      3) n=8; perturb=0.017; extra=(--live-vectors 3) ;;
    esac
    for mode in experiment control; do
      check=()
      if [[ $fixture != 3 ]]; then check=(--darwish-cache-rebuild-check); fi
      log="$logs/t$threads-f$fixture-$mode.log"
      if ! "$binary" --n "$n" --pack-size 64 --layout packed --geom affine \
          --jac-action --conv-ho 3 --rhie-chow --perturb "$perturb" \
          --warmup 1 --repeat 3 --darwish-cache-"$mode" \
          ${extra[@]+"${extra[@]}"} ${check[@]+"${check[@]}"} >"$log" 2>&1; then
        cat "$log"; exit 1
      fi
      grep -qx 'darwish_cache_checks: 12' "$log"
      expected=read
      if [[ $mode == control ]]; then expected=control; fi
      grep -qx "darwish_cache_mode: $expected" "$log"
      if [[ $fixture != 3 ]]; then
        grep -qx 'darwish_rebuild_checks: 36' "$log"
        grep -qx 'darwish_stale_refusals: 3' "$log"
      fi
    done
  done
done
export OMP_NUM_THREADS=1
for scope in atomic limiter rebuild lagged alignment geometry matrix; do
  case $scope in
    atomic) extra=(--layout atomic) ;;
    limiter) extra=(--conv-ho 0) ;;
    rebuild) extra=(--rebuild-state-gradients) ;;
    lagged) extra=(--lagged-rc) ;;
    alignment) extra=(--pack-size 65) ;;
    geometry) extra=(--geom isoparam) ;;
    matrix) extra=(--bsr-apply) ;;
  esac
  log="$logs/refuse-$scope.log"
  if "$binary" --n 8 --pack-size 64 --layout packed --geom affine \
      --jac-action --conv-ho 3 --rhie-chow --perturb 0.017 --warmup 0 --repeat 1 \
      --darwish-cache-experiment "${extra[@]}" >"$log" 2>&1; then
    cat "$log"; echo "unsupported cache scope accepted: $scope"; exit 1
  fi
  grep -q 'Darwish cache' "$log"
done
echo 'Darwish cache: perturbed directions, state rebuilds and seven scope refusals PASS'
