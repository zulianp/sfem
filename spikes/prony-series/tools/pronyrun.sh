#!/usr/bin/env bash
# Run a prony-series driver so its diagnostics are ALWAYS observable while it runs.
#
# A hand-rolled `driver | grep ...` gets this wrong twice over: the filter tends to select
# end-of-run summary lines, so a long run prints nothing until it finishes, and the pipe
# block-buffers on top of that. A working run then looks exactly like a hung one, and killing
# it destroys its history because nothing reached disk. Use this instead.
#
#   tools/pronyrun.sh <tag> <case.yaml> [env assignments...]
#
# Guarantees: a complete unfiltered log at $PRONYRUN_DIR/<tag>.log written from the first
# line, a line-buffered filtered view on stdout that includes the per-step line (step number,
# time, Newton and linear iteration counts, residual, angle, torque), and START/END timestamps
# so a slow run is distinguishable from a stuck one.
set -u
: "${PRONYRUN_DIR:=/tmp/pronylogs}"
mkdir -p "$PRONYRUN_DIR"

if [[ $# -lt 2 ]]; then
    echo "usage: $0 <tag> <case.yaml> [env assignments...]" >&2
    exit 1
fi

tag=$1; shift
case_file=$1; shift

EXE="${PRONY_EXE:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." >/dev/null 2>&1 && pwd -P)/build/prony_visco_torsion}"
if [[ ! -x "$EXE" ]]; then
    echo "missing executable: $EXE (set PRONY_EXE)" >&2
    exit 1
fi

# The per-step line starts with the step number; the ndof line, the release announcement and
# the totals all matter too. Anything not matched still lands in the raw log.
: "${PRONYRUN_FILTER:=^[0-9]+ |^nnodes:|^Solving |^--- torsion released|^control point|^Total linear|^Final |^\[prony\]|^prony::Case|^ +}"

echo ">>> START $tag  $(date +%H:%M:%S)  raw: $PRONYRUN_DIR/$tag.log"
# stdbuf on the BINARY as well as on the filter. Line-buffering only the filter is not enough:
# the program's own stdout is block-buffered when it is a pipe, so its output would reach the
# raw log up to 4 KB late and a running job would still look silent.
stdbuf -oL -eL env "$@" "$EXE" "$case_file" 2>&1 \
  | tee "$PRONYRUN_DIR/$tag.log" \
  | stdbuf -oL grep -E "$PRONYRUN_FILTER"
# PIPESTATUS[0] is the driver itself; tee and grep succeed regardless of it.
status=${PIPESTATUS[0]}
echo "<<< END   $tag  $(date +%H:%M:%S)  status=$status  raw: $PRONYRUN_DIR/$tag.log"
exit "$status"
