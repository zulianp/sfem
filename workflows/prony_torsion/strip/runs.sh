# shellcheck shell=bash
# The strip campaign's runs, shared by campaign.sbatch, collect.sbatch and tight.sbatch so they stay in step,
# and the stamp (strip_stamp).
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
# The stamp runs/strip<tag>/stamp.txt: what all of a tag's runs share, TURNS and EXTRA's words, one
# per line (TURNS=1, EXTRA=--set, ...; REDUCED and S32 choose runs, they change none, so are not in
# it). The tag's first job writes it; campaign.sbatch (DIAG=1 too)
# and tight.sbatch stop on another, collect.sbatch without one. EXTRA (options added to every run) is
# split like a shell command line by python's shlex: quotes group words, nothing expands (braces, $, ~
# and globs stay), '#' is no comment, no newlines. Runs from before the stamp: accepted only in the
# untagged runs/strip and runs/strip_diag, as TURNS=1 without EXTRA; a tagged folder needs the stamp
# written by hand (3f487e44 and 91244183 ran EXTRA without one), e.g.
#   printf 'TURNS=1\nEXTRA=--set\nEXTRA=material.C10=1e5\n' > runs/strip_soft/stamp.txt
strip_stamp() {  # TURNS, EXTRA, STRIP_TAG: opts=(newton output, EXTRA's words), or 1 if refused
  local tag=${STRIP_TAG:-} dir=$SFEM_RUNS/strip${STRIP_TAG:-} f words want n i a old tmp
  f=$dir/stamp.txt
  words=$($SFEM_PY -I -c 'import os, shlex, sys
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
  opts=(--set solver.newton.verbose=true); want="TURNS=$TURNS"$'\n'
  { read -r n; i=0; while [ "$i" -lt "$n" ]; do IFS= read -r a; opts+=("$a"); want="${want}EXTRA=$a"$'\n'; i=$((i+1)); done; } <<< "$words"
  mkdir -p "$dir" || return 1
  # runs from before the stamp: listed before the stamp is looked for, since a job makes runs only
  # after its stamp is in place (the meshes and stamp temporaries are not runs)
  old=$( { ls -A "$dir"; ls -A "$SFEM_RUNS/strip_diag$tag" 2> /dev/null; } | grep -v -e '^meshes$' -e '^stamp\.txt$' -e '^\.stamp\.')
  if [ ! -e "$f" ]; then
    if [ -n "$old" ] && { [ -n "$tag" ] || [ "$want" != "TURNS=1"$'\n' ]; }; then
      echo "$dir: runs from before stamps, TURNS and EXTRA unknown; write $f by hand (runs.sh) or give another STRIP_TAG"
      return 1
    fi
    tmp=$dir/.stamp.${HOSTNAME:-host}.$$  # written whole, then linked: two first jobs, one stamp
    printf '%s' "$want" > "$tmp" && ln "$tmp" "$f" 2> /dev/null; rm -f "$tmp"
    [ -e "$f" ] || { echo "cannot write $f"; return 1; }
  fi
  printf '%s' "$want" | cmp -s - "$f" && return
  echo "$f holds other TURNS or EXTRA than this job's:"; cat "$f"
  echo "give another STRIP_TAG, or delete $f if none of the tag's runs has started"; return 1
}
