#!/usr/bin/env bash
# That a multi-rank run is refused, and refused rather than crashed.
#
# The spike runs on one socket: every measurement in this workstream is 72 threads at
# --ntasks=1. It nonetheless launches under mpirun, and that is the hazard this gates. Measured
# at -n 2 before the guard existed: the mesh genuinely partitions -- each rank reports its own
# elements_per_pack -- both ranks build an operator, and the run then dies inside sfem_Op.hpp.
# That abort was luck, not a check.
#
# What makes the missing guard worse than a crash is which quantities are wrong without an
# exchange. They are all reductions: the FGMRES dot and norm, the pressure gauge's two
# projections, the Newton and Armijo acceptance tests, the CFL maximum. A rank evaluates each
# over its own slice and never notices the slice is partial, so the run converges, writes
# output, and reports a per-rank answer. A wrong number that looks like a right one is exactly
# what a gate is for.
#
# Four properties, because "it exits non-zero" alone would pass for the wrong reason -- the
# unconverted-operator abort also exits non-zero:
#   1. the guard fires, and says so;
#   2. it is collective -- the job terminates instead of hanging with a rank in a barrier;
#   3. it does not fire at one rank, so the serial path is untouched;
#   4. SFEM_ALLOW_MPI=1 opts out, so the parallel stages develop against this same binary.
set -u

DRIVER=${1:?usage: cvfem_mpi_guard_test.sh <cvfem_hex8_ns_ssgmg> <mpiexec> [bench]}
MPIEXEC=${2:?usage: cvfem_mpi_guard_test.sh <cvfem_hex8_ns_ssgmg> <mpiexec> [bench]}
BENCH=${3:-}

[ -x "$DRIVER" ] || { echo "mpi_guard: no driver at $DRIVER" >&2; exit 1; }

WORK=$(mktemp -d 2>/dev/null || mktemp -d -t cvfem_mpi_guard)
trap 'rm -rf "$WORK"' EXIT

# Whether this machine can actually launch two ranks is a property of the machine, not of the
# code under test. A container without shared memory, or an oversubscribed host, fails here for
# reasons that have nothing to do with the guard -- so probe first and report SKIP (ctest
# SKIP_RETURN_CODE 77) rather than a failure nobody can act on.
LAUNCH=("$MPIEXEC" -n 2)
if ! "${LAUNCH[@]}" true > "$WORK/probe.log" 2>&1; then
    # Open MPI refuses to place more ranks than it sees slots for; --oversubscribe is the
    # documented opt-in and changes nothing about what the ranks then do.
    LAUNCH=("$MPIEXEC" -n 2 --oversubscribe)
    if ! "${LAUNCH[@]}" true >> "$WORK/probe.log" 2>&1; then
        echo "mpi_guard: SKIP -- $MPIEXEC cannot launch 2 ranks here"
        sed 's/^/    /' "$WORK/probe.log" | head -12
        exit 77
    fi
fi

fail=0
say() { printf '%-58s %s\n' "$1" "$2"; [ "$2" = OK ] || fail=1; }

# Never /dev/null as the output folder: the driver treats it as a directory to create files in
# and takes the whole job down with MPI_Abort when it cannot.
OUT="$WORK/out"

# One configuration for every invocation below, and a solvable one. The cavity at N=2 on a flat
# mesh converges in well under a second, which is what lets this run in the unit-test suite
# instead of beside the solver campaigns. (The pump, the obvious choice for a cheap case, is not
# usable here: at N=2 its port selects no faces and the problem is unsolvable by construction,
# and at N=4 it costs 29 s.)
#
# It matters that even the runs which are SUPPOSED to be refused carry this. Handed no
# SFEM_CASE the driver prints usage and exits 1 on its own, so a refusal asserted without a case
# would pass whether or not the guard existed -- it would be asserting that a driver with
# nothing to do stops. With a solvable case a missing guard means the run proceeds, and the
# assertions below can tell the two apart.
#
# SFEM_FGMRES=1 explicitly: the driver falls back to BiCGStab whenever multigrid is off, and no
# run in this spike is allowed to inherit that default.
CASE="SFEM_CASE=cavity SFEM_N=2 SFEM_ELEMENT_REFINE_LEVEL=1 SFEM_FGMRES=1 SFEM_GMG=0
      SFEM_PRECOND=direct SFEM_ENABLE_OUTPUT=0"

# The abort this spike hits when it is allowed to proceed multi-rank. Matched as a pattern
# rather than a fixed string because the wording has already moved once: the message is
# "cvfem:NavierStokes does not support scoped flat-range apply" at sfem_Op.hpp:257 in the SFEM
# installed here, where an earlier note recorded "does not support ElementScope other than ALL"
# at :238. Both spellings are accepted so this gate reports the guard rather than the wording.
UNCONVERTED="does not support scoped|does not support ElementScope"

# ---------------------------------------------------------------- 1. the guard fires and says so
env $CASE "${LAUNCH[@]}" "$DRIVER" "$OUT" > "$WORK/two.log" 2>&1
rc_two=$?

[ $rc_two -ne 0 ] \
    && say "a 2-rank run exits non-zero" OK \
    || say "a 2-rank run exits non-zero" FAIL

grep -q "refusing to run on 2 ranks" "$WORK/two.log" \
    && say "it refuses by name, rather than crashing later" OK \
    || { say "it refuses by name, rather than crashing later" FAIL; tail -15 "$WORK/two.log"; }

# The distinction the previous assertion cannot make on its own. If the guard ever stops firing,
# the run reaches the operator and aborts there -- still non-zero, still "failing". Naming that
# abort is what stops this gate passing on it. It is also why the case above must be solvable:
# against no case the run would stop at usage and this would be vacuous.
grep -qE "$UNCONVERTED" "$WORK/two.log" \
    && say "it stops BEFORE the unconverted-operator abort" FAIL \
    || say "it stops BEFORE the unconverted-operator abort" OK

# Rank 0 alone prints, so the message appears once and not once per rank. On a full node the
# alternative is 288 copies of it.
n_msg=$(grep -c "refusing to run on" "$WORK/two.log")
[ "$n_msg" = 1 ] \
    && say "only rank 0 prints the refusal (1 copy)" OK \
    || say "only rank 0 prints the refusal ($n_msg copies)" FAIL

# ------------------------------------------------------- 2. the refusal is collective, not a hang
#
# This is the property that distinguishes a guard from a bug. If only rank 0 returned non-zero
# and the others continued into a collective, the job would hang rather than exit -- and it
# would hang in CI, where a timeout reads as an infrastructure problem. That the launcher
# returned at all is the evidence, so it is asserted against the clock rather than assumed.
start=$(date +%s)
env $CASE "${LAUNCH[@]}" "$DRIVER" "$WORK/out2" > "$WORK/two2.log" 2>&1
elapsed=$(( $(date +%s) - start ))
[ "$elapsed" -lt 60 ] \
    && say "every rank exits (no barrier hang, ${elapsed}s)" OK \
    || say "every rank exits (no barrier hang, ${elapsed}s)" FAIL

# ------------------------------------------------------------ 3. one rank is entirely unaffected
env $CASE "$DRIVER" "$WORK/serial" > "$WORK/serial.log" 2>&1
rc_serial=$?
[ $rc_serial -eq 0 ] \
    && say "a 1-rank run still succeeds" OK \
    || { say "a 1-rank run still succeeds" FAIL; tail -15 "$WORK/serial.log"; }

grep -q "refusing to run on" "$WORK/serial.log" \
    && say "the guard does not fire at one rank" FAIL \
    || say "the guard does not fire at one rank" OK

# The same run launched through mpirun -n 1 rather than bare, because that is how a job script
# reaches the driver and the guard must read comm size, not the presence of a launcher.
env $CASE "$MPIEXEC" -n 1 "$DRIVER" "$WORK/serial1" > "$WORK/serial1.log" 2>&1
rc_serial1=$?
[ $rc_serial1 -eq 0 ] \
    && say "mpirun -n 1 is equally unaffected" OK \
    || { say "mpirun -n 1 is equally unaffected" FAIL; tail -15 "$WORK/serial1.log"; }

# ------------------------------------------------------------------------- 4. the opt-out works
#
# Asserted on the MESSAGE and not on the exit code. With SFEM_ALLOW_MPI=1 the run proceeds into
# the unconverted operator and exits non-zero anyway, so an exit-code assertion here would pass
# whether or not the opt-out was honoured.
#
# That run currently dies on SIGSEGV rather than a clean MPI_Abort -- SFEM_ERROR's assert(0) is
# compiled out in Release, and what follows it crashes. The noise in the log is expected and is
# precisely the state these stages exist to remove; nothing here asserts on how it dies.
env $CASE SFEM_ALLOW_MPI=1 "${LAUNCH[@]}" "$DRIVER" "$WORK/allow" > "$WORK/allow.log" 2>&1

grep -q "refusing to run on" "$WORK/allow.log" \
    && say "SFEM_ALLOW_MPI=1 opts out of the refusal" FAIL \
    || say "SFEM_ALLOW_MPI=1 opts out of the refusal" OK

# And that opting out really does reach the not-yet-parallel code, rather than failing earlier
# for some unrelated reason -- otherwise the assertion above passes trivially.
grep -qE "$UNCONVERTED" "$WORK/allow.log" \
    && say "opting out reaches the unconverted operator" OK \
    || { say "opting out reaches the unconverted operator" FAIL; tail -15 "$WORK/allow.log"; }

# --------------------------------------------- 5. the benchmarks refuse through smesh's own guard
#
# The bench drivers call sfem::initialize_serial instead of carrying a copy of the refusal
# above. That mechanism already existed and already aborts collectively, so this checks that the
# house guard is wired up -- not that a second implementation of it behaves.
if [ -n "$BENCH" ] && [ -x "$BENCH" ]; then
    "${LAUNCH[@]}" "$BENCH" > "$WORK/bench.log" 2>&1
    rc_bench=$?
    [ $rc_bench -ne 0 ] \
        && say "the packed benchmark refuses 2 ranks" OK \
        || say "the packed benchmark refuses 2 ranks" FAIL
    grep -q "does not support parallel runs" "$WORK/bench.log" \
        && say "it refuses through smesh's own guard" OK \
        || { say "it refuses through smesh's own guard" FAIL; tail -10 "$WORK/bench.log"; }
fi

echo
[ $fail -eq 0 ] && echo PASSED || echo FAILED
exit $fail
