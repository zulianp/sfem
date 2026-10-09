#!/usr/bin/env bash
# A step that will not converge must either stop the run or take a smaller one -- never be
# accepted quietly.
#
# SFEM_STEP_ON_FAIL has two policies and they are opposites, so the test drives both against
# the same forced failure and checks they disagree in the one way they should: `abort` leaves
# the run stopped and non-zero, `cutback` shrinks dt, retries, and carries on.
#
# The failure is forced with SFEM_NL_MAX_IT, not by a hard case. A cap of one Newton iteration
# leaves the pump's first step at rel ~6e-02 -- four orders above SFEM_NL_LS_FLOOR (1e-6), so
# the residual-floor acceptance cannot swallow it and what is being tested is genuinely the
# non-convergence path rather than a step that was converged all along.
#
# The pump because it is the cheapest case the driver has with a time-varying boundary
# condition, which is what makes the reported instants worth checking: a cutback changes the
# size a step takes, so a run that still labelled its steps n*dt would be reporting times it
# never visited. That was a real defect -- the absolute time keyed on "is the CFL controller
# on" rather than "has the step size moved" -- and the banner check below is what would catch
# it coming back.
set -u

DRIVER=${1:?usage: cvfem_cutback_test.sh <cvfem_hex8_ns_ssgmg>}
[ -x "$DRIVER" ] || { echo "cutback: no driver at $DRIVER" >&2; exit 1; }

WORK=$(mktemp -d 2>/dev/null || mktemp -d -t cvfem_cutback)
trap 'rm -rf "$WORK"' EXIT

COMMON="SFEM_CASE=pump SFEM_N=8 SFEM_MU=0.05 SFEM_U=1 SFEM_GMG=0 SFEM_PRECOND=direct
        SFEM_DT=0.1 SFEM_PUMP_PERIOD=1 SFEM_BDF_ORDER=2 SFEM_ENABLE_OUTPUT=0"

fail=0
say() { printf '%-58s %s\n' "$1" "$2"; [ "$2" = OK ] || fail=1; }

# ---------------------------------------------------------------- the control: it converges
#
# The same case with a workable cap. Without this row a failing assertion below could mean the
# case is simply broken rather than that the policy misbehaved.
env $COMMON SFEM_NL_MAX_IT=20 SFEM_NSTEPS=3 "$DRIVER" "$WORK/ok" > "$WORK/ok.log" 2>&1
rc=$?
[ $rc -eq 0 ] && say "the unforced case converges (control)" OK \
              || { say "the unforced case converges (control)" FAIL; tail -5 "$WORK/ok.log"; }

# ---------------------------------------------------------------------------- abort
#
# One Newton iteration per stage cannot solve this step. The default policy stops the run, says
# so on stderr, and exits non-zero.
env $COMMON SFEM_NL_MAX_IT=1 SFEM_NSTEPS=3 "$DRIVER" "$WORK/abort" > "$WORK/abort.log" 2>&1
rc=$?
if [ $rc -ne 0 ] && grep -q "FAILED to converge" "$WORK/abort.log"; then
    say "abort: a failed step stops the run, non-zero" OK
else
    say "abort: a failed step stops the run, non-zero" FAIL
    tail -5 "$WORK/abort.log"
fi

# The policy is spelled, not guessed: anything that is neither abort nor cutback is refused
# rather than silently meaning abort.
env $COMMON SFEM_NL_MAX_IT=20 SFEM_NSTEPS=1 SFEM_STEP_ON_FAIL=shrink \
    "$DRIVER" "$WORK/bogus" > "$WORK/bogus.log" 2>&1
rc=$?
if [ $rc -ne 0 ] && grep -q "is not one of abort|cutback" "$WORK/bogus.log"; then
    say "an unknown policy is refused, not aliased to abort" OK
else
    say "an unknown policy is refused, not aliased to abort" FAIL
    tail -3 "$WORK/bogus.log"
fi

# ---------------------------------------------------------------------------- cutback
#
# The same forced failure under the other policy. Every attempt fails at this cap, so the run
# spends its whole budget and then stops -- which is the path worth testing: the retries
# happen, they are counted and bounded, the sizes halve, and the run still refuses to accept
# the unconverged state at the end.
env $COMMON SFEM_NL_MAX_IT=1 SFEM_NSTEPS=3 SFEM_STEP_ON_FAIL=cutback \
    SFEM_STEP_MAX_CUTBACK=3 "$DRIVER" "$WORK/cut.d" > "$WORK/cut.log" 2>&1
rc=$?

n_cut=$(grep -c "cutback [0-9]*/3 to dt" "$WORK/cut.log" || true)
[ "$n_cut" -eq 3 ] && say "cutback: the budget is spent, three retries" OK \
                   || { say "cutback: the budget is spent, three retries" FAIL
                        grep -n "cutback" "$WORK/cut.log" | head -5; }

# 0.1 -> 0.05 -> 0.025 -> 0.0125, each the previous halved.
if grep -q "cutback 1/3 to dt = 0.05" "$WORK/cut.log" &&
   grep -q "cutback 2/3 to dt = 0.025" "$WORK/cut.log" &&
   grep -q "cutback 3/3 to dt = 0.0125" "$WORK/cut.log"; then
    say "cutback: each retry halves the step" OK
else
    say "cutback: each retry halves the step" FAIL
    grep -n "cutback" "$WORK/cut.log" | head -5
fi

if [ $rc -ne 0 ] && grep -q "cutback exhausted after 3 of 3" "$WORK/cut.log"; then
    say "cutback: an exhausted budget still refuses the step" OK
else
    say "cutback: an exhausted budget still refuses the step" FAIL
    tail -5 "$WORK/cut.log"
fi

# The retries must not advance the clock. Each attempt rolls t_run back before shrinking, so
# every attempt at step 1 reports an instant no larger than the full step's.
bad_t=$(awk '/^=== step 1 /{ for (i=1;i<=NF;i++) if ($i=="t") { t=$(i+2)+0; if (t > 0.1000001) print t } }' \
             "$WORK/cut.log" | head -1)
[ -z "$bad_t" ] && say "cutback: a retry does not advance the clock" OK \
                || { say "cutback: a retry does not advance the clock" FAIL; echo "  saw t = $bad_t"; }

# A cutback that would fall below SFEM_DT_MIN has nowhere left to go and must stop there rather
# than halving forever towards zero.
env $COMMON SFEM_NL_MAX_IT=1 SFEM_NSTEPS=1 SFEM_STEP_ON_FAIL=cutback \
    SFEM_STEP_MAX_CUTBACK=20 SFEM_DT_MIN=0.03 "$DRIVER" "$WORK/floor.d" > "$WORK/floor.log" 2>&1
rc=$?
if [ $rc -ne 0 ] && grep -q "below SFEM_DT_MIN" "$WORK/floor.log"; then
    say "cutback: SFEM_DT_MIN is a floor, not a suggestion" OK
else
    say "cutback: SFEM_DT_MIN is a floor, not a suggestion" FAIL
    grep -n "cutback" "$WORK/floor.log" | tail -3
fi

# A malformed factor is refused up front rather than producing a step that grows or stands
# still on every retry.
env $COMMON SFEM_NSTEPS=1 SFEM_STEP_ON_FAIL=cutback SFEM_STEP_CUTBACK_FACTOR=1.5 \
    "$DRIVER" "$WORK/badf.d" > "$WORK/badf.log" 2>&1
rc=$?
if [ $rc -ne 0 ] && grep -q "SFEM_STEP_CUTBACK_FACTOR must lie in (0,1)" "$WORK/badf.log"; then
    say "cutback: the shrink factor is validated" OK
else
    say "cutback: the shrink factor is validated" FAIL
    tail -3 "$WORK/badf.log"
fi

# ------------------------------------------------- a cutback that WORKS, and then recovers
#
# The case above spends its whole budget because one Newton iteration cannot solve this step at
# any size. That exercises the retries and the refusal, but not the path a campaign actually
# takes: fail once, succeed at the smaller step, carry on. SFEM_NL_MAX_IT=3 is the cap that
# separates them -- measured on this case, it fails at dt = 0.1 and converges at dt = 0.05 --
# so the first step cuts back exactly once and every step after it runs.
#
# This is also the only place x_step_start is exercised as intended. Everywhere above, the
# restored state is handed to an attempt that fails again; here it is handed to one that
# converges, so a restore that silently wrote the wrong field would show up as a step that
# cannot solve at a size the probe says is solvable.
env $COMMON SFEM_NL_MAX_IT=3 SFEM_NSTEPS=6 SFEM_STEP_ON_FAIL=cutback \
    SFEM_STEP_RECOVER_AFTER=2 "$DRIVER" "$WORK/rec.d" > "$WORK/rec.log" 2>&1
rc=$?

[ $rc -eq 0 ] && say "recovery: a cutback run completes its steps" OK \
              || { say "recovery: a cutback run completes its steps" FAIL; tail -6 "$WORK/rec.log"; }

n_rec=$(grep -c "cutback 1/" "$WORK/rec.log" || true)
[ "$n_rec" -eq 1 ] && say "recovery: the first step cuts back exactly once" OK \
                   || { say "recovery: the first step cuts back exactly once" FAIL
                        grep -n "cutback" "$WORK/rec.log" | head -4; }

# Two converged steps at the smaller size earn the requested one back, by the same factor the
# cutback used, and it stops there rather than growing past SFEM_DT.
if grep -q "converged steps since the last cutback -- dt 0.05 to 0.1" "$WORK/rec.log"; then
    say "recovery: the requested step size is earned back" OK
else
    say "recovery: the requested step size is earned back" FAIL
    grep -n "converged steps since" "$WORK/rec.log" | head -3
fi
if grep -qE "dt 0\.1 to 0\.2|dt 0\.1 to 0\.1[0-9]" "$WORK/rec.log"; then
    say "recovery: growth stops at SFEM_DT" FAIL
    grep -n "converged steps since" "$WORK/rec.log" | head -3
else
    say "recovery: growth stops at SFEM_DT" OK
fi

# The clock follows the sizes actually taken. The cutback lands step 1 on 0.05, step 2 runs at
# that size and earns the requested one back, so steps 3 to 6 run at 0.1 and the run ends at
# 0.05*2 + 0.1*4 = 0.5 -- not the 0.6 that n*dt would report. That difference is exactly the
# defect the absolute time had when it keyed on whether the CFL controller was on rather than
# on whether the step size had moved.
#
# The step 1 banner appearing twice, at 0.1 and then at 0.05, is the other half of the same
# property: a retry rolls the clock back before it re-runs, rather than advancing it twice.
if grep -q "=== step 6 (segment 6/6)  t = 0.5 ===" "$WORK/rec.log"; then
    say "recovery: the clock follows the sizes taken, not n*dt" OK
else
    say "recovery: the clock follows the sizes taken, not n*dt" FAIL
    grep -n "=== step" "$WORK/rec.log" | tail -3
fi

# ------------------------------------------------------------- the default is untouched
#
# Everything above only fires when the policy is asked for. A run that says nothing must take
# exactly the steps it always did, at the size it was given.
if grep -q "=== step 3 (segment 3/3)  t = 0.3 ===" "$WORK/ok.log"; then
    say "default: a converging run keeps SFEM_DT and its clock" OK
else
    say "default: a converging run keeps SFEM_DT and its clock" FAIL
    grep -n "=== step" "$WORK/ok.log" | tail -3
fi

echo
[ $fail -eq 0 ] && echo PASSED || echo FAILED
exit $fail
