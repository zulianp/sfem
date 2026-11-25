# shellcheck shell=bash
# The strip campaign's runs, shared by campaign.sbatch, collect.sbatch and tight.sbatch so they stay in step.
#   REDUCED  the reduced history formats, each with the remainder form (<format>) and without it
#            (<format>_plain): float32, int16_scaled, float16_md, and the packed grid codes
#            qblock5/6/7 (quant_block at history_tolerance 1e-5, 1e-6, 1e-7), _sr with stochastic
#            rounding (qblock6 against qblock6_sr: the rounding alone)
#   S32      the runs repeated with float32 evaluated stresses (<run>_s32)
# Override either from the environment (REDUCED="int16_scaled qblock6_sr").
REDUCED=${REDUCED:-float32 int16_scaled float16_md qblock5_sr qblock6 qblock6_sr qblock7_sr}
S32=${S32:-full_a int16_scaled float16_md qblock6_sr qblock7_sr}
strip_runs() {  # the main campaign's runs; DIAG=1: the diagnostics build's (lost updates)
  local f runs
  if [ "${DIAG:-0}" = 1 ]; then
    runs="full_a"; for f in $REDUCED; do runs="$runs $f ${f}_plain"; done
  else
    runs="full_a full_b"; for f in $REDUCED; do runs="$runs $f ${f}_plain"; done
    for f in $S32; do runs="$runs ${f}_s32"; done
  fi
  echo "$runs"
}
# EXTRA: options added to every run (campaign.sbatch, tight.sbatch), split like a shell command line
# by python's shlex: quotes group words, nothing expands (no braces, $, ~ or globs), no newlines.
strip_extra() {  # <folder>: opts=(the newton output and EXTRA's words); refuse if <folder>'s runs had others
  local dir=$1 f=$1/extra.txt words want="" n i a
  words=$(python3 -I -c 'import os, shlex, sys
e = os.environ.get("EXTRA", "")
if "\n" in e:
    sys.exit("EXTRA has a newline")
try:
    w = shlex.split(e)
except ValueError as x:
    sys.exit("EXTRA: %s" % x)
print(len(w))
for a in w:
    print(a)') || return 1
  opts=(--set solver.newton.verbose=true)
  { read -r n; i=0; while [ "$i" -lt "$n" ]; do IFS= read -r a; opts+=("$a"); want="$want$a"$'\n'; i=$((i+1)); done; } <<< "$words"
  # <folder>/extra.txt: EXTRA's words, one per line, written by the folder's first job (set -C: once)
  mkdir -p "$dir" || return 1
  [ ! -e "$f" ] && [ -z "$(ls -A "$dir")" ] && ( set -C; printf '%s' "$want" > "$f" ) 2> /dev/null
  if [ -e "$f" ]; then
    printf '%s' "$want" | cmp -s - "$f" && return
    echo "$dir ran with other EXTRA options ($f); give another STRIP_TAG"; return 1
  fi
  # a folder from before extra.txt: its runs had no EXTRA
  [ "$n" = 0 ] || { echo "$dir has runs without EXTRA; give another STRIP_TAG"; return 1; }
  : > "$f"
}
