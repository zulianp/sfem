#!/usr/bin/env bash
# A run cut in two must be the run that was not cut.
#
# That is the whole claim of the restart, and it is not a claim a size check or a "it loaded
# without complaining" can support. What makes it falsifiable here is that the pump carries an
# exact identity -- the flux through the port equals the volume the diaphragm swept -- and a
# waveform, so a state resumed at the wrong instant, with a stale history, or with BDF2
# silently dropped to BDF1 for one step, all move the answer. Six steps in one go against
# three plus three, compared on the driver's own final numbers.
#
# The pump rather than the step because it is the cheapest case the driver has that exercises
# a time-varying boundary condition: the diaphragm velocity is a function of absolute time, so
# a resumed segment that mistook its own step index for the absolute one would be driving the
# wall at the wrong phase and the identity would say so.
set -u

DRIVER=${1:?usage: cvfem_restart_test.sh <cvfem_hex8_ns_ssgmg>}
[ -x "$DRIVER" ] || { echo "restart: no driver at $DRIVER" >&2; exit 1; }

WORK=$(mktemp -d 2>/dev/null || mktemp -d -t cvfem_restart)
trap 'rm -rf "$WORK"' EXIT

# SFEM_STATS=1 so the running moments cross the seam under exactly the conditions above. They
# are the one piece of restart state that is not the solution: the solution is checked by the
# identities below, but a mean accumulated over a window can be wrong in ways the instantaneous
# state cannot show -- a sample dropped where the segments meet, one counted twice, or moments
# resumed against a window that did not come with them.
COMMON="SFEM_CASE=pump SFEM_N=8 SFEM_MU=0.05 SFEM_U=1 SFEM_GMG=0 SFEM_PRECOND=direct
        SFEM_NL_MAX_IT=20 SFEM_DT=0.1 SFEM_PUMP_PERIOD=1 SFEM_BDF_ORDER=2 SFEM_ENABLE_OUTPUT=0
        SFEM_STATS=1"

fail=0
say() { printf '%-58s %s\n' "$1" "$2"; [ "$2" = OK ] || fail=1; }

# The reference: six steps, uninterrupted.
env $COMMON SFEM_NSTEPS=6 "$DRIVER" "$WORK/one" > "$WORK/one.log" 2>&1
[ $? -eq 0 ] || { echo "restart: the uninterrupted run failed"; tail -5 "$WORK/one.log"; exit 1; }

# The same six, as three and three.
env $COMMON SFEM_NSTEPS=3 SFEM_RESTART_OUT="$WORK/ckpt" "$DRIVER" "$WORK/a" > "$WORK/a.log" 2>&1
[ $? -eq 0 ] || { echo "restart: segment A failed"; tail -5 "$WORK/a.log"; exit 1; }
env $COMMON SFEM_NSTEPS=3 SFEM_RESTART_IN="$WORK/ckpt" "$DRIVER" "$WORK/b" > "$WORK/b.log" 2>&1
[ $? -eq 0 ] || { echo "restart: segment B failed"; tail -5 "$WORK/b.log"; exit 1; }

grep -q "restart: wrote step 3" "$WORK/a.log" && say "segment A wrote a checkpoint at step 3" OK \
    || say "segment A wrote a checkpoint at step 3" FAIL
grep -q "restart: resumed from .* at step 3" "$WORK/b.log" && say "segment B resumed at step 3" OK \
    || say "segment B resumed at step 3" FAIL
# BDF2 must survive the seam. A resumed run is mid-sequence, not starting up, so falling back
# to BDF1 for the first step of the segment would put a first-order error in the middle of a
# second-order run -- and would still produce a plausible-looking answer.
grep -q "BDF2 history" "$WORK/b.log" && say "the second history level came back (BDF2)" OK \
    || say "the second history level came back (BDF2)" FAIL

# The absolute step index, so frames from consecutive segments concatenate rather than
# overwrite each other, and the waveform is evaluated at the right phase.
grep -q "=== step 6 (segment 3/3)" "$WORK/b.log" && say "steps are numbered absolutely across segments" OK \
    || say "steps are numbered absolutely across segments" FAIL

# The moments came back with their window. Checked separately from the equality below because
# the failure it catches is specific: moments restored against a default-zero window would give
# the first step of the new segment the entire weight of everything averaged before it.
grep -q "restart: resumed statistics, window .* over 3 samples" "$WORK/b.log" \
    && say "the averaging window came back with the moments" OK \
    || say "the averaging window came back with the moments" FAIL

# The statistics line joins the identities below. Equality here is EXACT and not approximate:
# a run cut in two performs the same Welford updates on the same states in the same order, the
# moments round-trip through float64 losslessly, and the window round-trips through restart.txt
# at 17 significant digits -- which is what makes a double exact in text. Anything less than a
# character-for-character match is a real defect, not accumulated round-off.
for what in "u_linf" "pump: swept" "pump: |port - swept|" "stats: window"; do
    a=$(grep -F "$what" "$WORK/one.log" | tail -1)
    b=$(grep -F "$what" "$WORK/b.log"   | tail -1)
    if [ -n "$a" ] && [ "$a" = "$b" ]; then
        say "identical across the seam: $what" OK
    else
        say "identical across the seam: $what" FAIL
        printf '    one-shot : %s\n    segmented: %s\n' "$a" "$b"
    fi
done

# A restart must refuse a mesh it does not belong to rather than loading into a different node
# numbering, which is silent: every array is the right length and the run continues from a
# scrambled field.
env $COMMON SFEM_NSTEPS=1 SFEM_N=12 SFEM_RESTART_IN="$WORK/ckpt" "$DRIVER" "$WORK/c" > "$WORK/c.log" 2>&1
rc=$?
if [ $rc -ne 0 ] && grep -qE "restart: (size mismatch|mesh coordinate checksum)" "$WORK/c.log"; then
    say "a restart from a different mesh is refused" OK
else
    say "a restart from a different mesh is refused" FAIL
    tail -3 "$WORK/c.log"
fi

echo
[ $fail -eq 0 ] && echo PASSED || echo FAILED
exit $fail
