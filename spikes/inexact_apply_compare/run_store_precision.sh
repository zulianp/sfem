#!/usr/bin/env bash
# Where the store's precision stops mattering, and where it starts again.
#
#   run_store_precision.sh [element] [amplitudes...]
#
# Sweeps the deformation severity at a fixed mesh and reports the three error
# columns side by side.  Two things make the sweep the right shape:
#
#   * At amplitude zero the deformation gradient is constant, so the projection
#     is exact on any element and the whole remaining difference is the store's.
#     That row is the control -- without it there is no way to tell a store
#     error from a projection error, because both show up in the same number.
#
#   * The projection error is first order in the deformation while the store's
#     contribution is not, so sweeping separates them by construction.
#
# The default increment is white noise, because the smooth one makes the ratio
# flat for reasons that have nothing to do with either error; see "The TET10
# non-convergence was the metric" in RESULTS.md.
set -euo pipefail

ELEMENT="${1:-TET10}"; shift || true
AMPLITUDES=("$@")
[ ${#AMPLITUDES[@]} -eq 0 ] && AMPLITUDES=(0 0.0025 0.005 0.01 0.02 0.04 0.08)

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
case "$ELEMENT" in
    TET10) ROW='^    384000' ;;
    HEX8)  ROW='^     64000' ;;
    TET4)  ROW='^    384000' ;;
    *) echo "unknown element $ELEMENT" >&2; exit 1 ;;
esac

printf 'store precision sweep: %s, white-noise increment, finest mesh\n\n' "$ELEMENT"
printf '%9s | %9s %9s %9s\n' "amplitude" "f64 diff" "f32 diff" "f16 diff"
printf '%9s | %9s %9s %9s\n' "" "(proj)" "" ""
for A in "${AMPLITUDES[@]}"; do
    row=$(MIXED_EXTRA_FLAGS="-DRANDOM_INCREMENT -DSTATE_AMPLITUDE=$A" \
          "$HERE/run_mixed.sh" "$ELEMENT" 1 2>&1 | grep "$ROW" || true)
    printf '%9s | %s\n' "$A" "$(echo "$row" | awk '{print $(NF-2), $(NF-1), $NF}')"
done
