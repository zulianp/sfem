#!/bin/zsh
# Run a CVFEM driver so its diagnostics are ALWAYS observable while it runs.
#
# Every ad-hoc `... | grep ...` pipeline gets this wrong in the same two ways: the filter
# selects end-of-run summary fields, so a long run prints nothing until it finishes, and grep
# block-buffers a pipe on top of that. A run that is working then looks identical to a run
# that has hung, and when it is killed its history is gone because nothing was written to
# disk. Use this instead of hand-rolling the pipeline.
#
#   tools/cvrun.sh <tag> <binary> [env assignments...] -- [args...]
#
# Guarantees: a complete unfiltered log at $CVRUN_DIR/<tag>.log written from the first line,
# a line-buffered filtered view on stdout that includes per-iteration residuals, and
# START/END timestamps so a slow run is distinguishable from a stuck one.
set -u
: ${CVRUN_DIR:=/tmp/cvlogs}
mkdir -p "$CVRUN_DIR"
tag=$1; shift
bin=$1; shift
# nnodes/nelements/ndof are in the default filter deliberately: a timing without the
# problem size it was measured on is not a result, and the drivers already print the
# line during setup, so this is a matter of not dropping it.
: ${CVRUN_FILTER:="^newton |^stage |^continuation|^ *upwind |band eps|smoother|sweep |^[0-9]+\	|highest Re SOLVED|newton_converged|lin_it_total|t_solve|u_l|sum of continuity|active set|fd_at_it|eps |nnodes:|nelements:|ndof:"}
echo ">>> START $tag  $(date +%H:%M:%S)  raw: $CVRUN_DIR/$tag.log"
# stdbuf on the BINARY as well as the filter. Line-buffering only the filter is not enough:
# the program's own stdout is block-buffered when it is a pipe, so its output reaches the raw
# log up to 4 KB late and a running job still looks silent. Both ends need it.
stdbuf -oL -eL env "$@" "$bin" "${CVRUN_OUT:-/tmp/cvrun_$tag}" 2>&1 \
  | tee "$CVRUN_DIR/$tag.log" \
  | stdbuf -oL grep -E "$CVRUN_FILTER"
echo "<<< END   $tag  $(date +%H:%M:%S)  raw: $CVRUN_DIR/$tag.log"
