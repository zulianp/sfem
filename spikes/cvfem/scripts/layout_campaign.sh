#!/usr/bin/env bash
# The standard-vs-packed layout campaign.
#
# This is a SWEEP, not a gate. scripts/perf_regression.sh is the gate, and it is
# deliberately not extended into this: a gate answers one question -- did this change cost
# throughput -- and a failure has to be unambiguous. Growing it into a cross product makes
# every failure ambiguous, because a red run could then mean a regression or merely a
# configuration the sweep newly covers. Two scripts, two jobs.
#
# What it answers: for one operator at a time, how the packed layout compares with the
# standard (atomic) one across problem sizes, and how both compare with applying the
# assembled matrix -- at double and at single value storage, which is the only lever a
# bandwidth-bound SpMV has.
#
#   scripts/layout_campaign.sh                run the sweep, write one csv
#   scripts/layout_campaign.sh --list         print the configurations and stop
#   scripts/layout_campaign.sh --dry-run      print the command lines and stop
#
# It writes ONE csv holding every repetition, and the reduction is done downstream by
# python/cvfem_kernel_report.py, which keeps the best reading per configuration and reports
# the spread of the rest beside it. That division is on purpose: this script must not decide
# what the numbers mean, and a median taken here would throw away the spread that says
# whether a difference is a result or noise.
#
# Environment:
#   BIN       binary under test (default: the first build*/ that has it)
#   REPS      repetitions of the whole sweep (default 5)
#   THREADS   OMP threads (default 72, the full Grace socket)
#   SIZES     cube edge counts (default "64 96 128 160 192")
#   OUT       directory for the csv (default: perf/campaign_<host>_<date>)
#   RUN_TIMEOUT  seconds any single run may take before it is killed (default 900)
#
# ON WHERE TO RUN IT. On a Grace node, at the standing node shape, and nowhere else. A
# number measured on the development laptop is not comparable with one measured on 72 Grace
# cores, and this whole sweep exists to compare numbers with each other. The laptop can run
# it with a small SIZES to check the plumbing; that output is not a result.
#
# DO NOT COMPARE TWO CAMPAIGNS AGAINST EACH OTHER. Measured, 2026-09-24/25: the same sweep
# run before and after a change that a controlled A/B had just shown to be performance-neutral
# came back 5.7%, 6.4% and 6.5% SLOWER on the packed residual at the three largest sizes, and
# by the same amount on the packed Jacobian action. The two jobs landed on nid006538 and
# nid005669. Two independent kernels moving together, only at the memory-bound sizes while the
# cache-resident ones moved slightly the other way, is the node and not the code.
#
# That is the documented 5-11% node-to-node variation, and it is larger than most differences
# worth finding. This sweep characterises ONE binary across sizes and layouts, which is what
# its rows are interleaved for. Comparing two binaries is scripts/perf_regression.sh --against,
# which measures both in one allocation with the order alternating; nothing else is entitled to
# that claim. The temptation is real -- the numbers sit in two CSVs and subtract cleanly -- so
# it is written down here rather than left to be rediscovered.
#
# ON SATURATION. n=64 and n=96 are in the default list to SHOW the knee, not to be compared
# across layouts: the packed residual saturates by n=128 (2620.2 at n=128 against 2617.0 at
# n=160 on Grace). Cross-layout conclusions are drawn at n >= 128, and the smaller sizes are
# there so a reader can see that for themselves rather than take it on trust.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
SPIKE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$SPIKE_ROOT"

REPS="${REPS:-5}"
THREADS="${THREADS:-72}"
SIZES="${SIZES:-64 96 128 160 192}"
# A cap per RUN, not per sweep. The sweep is long and it is submitted to a batch queue, so a
# single configuration that deadlocks or thrashes would otherwise consume the whole
# allocation and every configuration after it would be missing -- which reads as a failed
# sweep rather than as one bad arm. With the cap, one arm is lost and the rest still report.
RUN_TIMEOUT="${RUN_TIMEOUT:-900}"

MODE=run
case "${1:-}" in
    --list)    MODE=list ;;
    --dry-run) MODE=dry ;;
    --help|-h) sed -n '2,45p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    "") ;;
    *) echo "unknown argument: $1 (try --help)" >&2; exit 2 ;;
esac

# ------------------------------------------------------------------ the configurations
#
# key|operation|layout|extra
#
# The driver REFUSES some combinations, and it is right to: --assemble-diag dispatches on
# geometry alone and would record a --layout it never used, and --layout ecolor implements no
# assembly. A sweep that emitted them anyway would spend its allocation collecting non-zero
# exit codes and would leave holes in the table that look like measurements that failed. So the
# refusals are encoded here and a refused combination is absent by construction rather than
# filtered out afterwards.
#
# There is no kernel column. DESIGN.md's second correction leaves one micro-kernel per kernel,
# so what this sweep varies is the layout, the operation and the scheme -- which is what its
# name has always said it was for.
#
# Ordering matters as much as membership. The layouts alternate ADJACENTLY within each
# operator, so that the pair of readings a ratio is computed from sits as close together in
# the allocation as the sweep allows. Position within a run is worth 12% here (measured, see
# perf_regression.sh), which is larger than most of the differences this campaign is looking
# for, and the reps run in alternating direction to balance what is left.
CONFIGS=(
    # -- the bare element kernel, both layouts ------------------------------------------
    "residual_bare|residual|packed|"
    "residual_bare|residual|ecolor|"
    "residual_bare|residual|atomic|"
    "jac_bare|jac_action|packed|"
    "jac_bare|jac_action|ecolor|"
    "jac_bare|jac_action|atomic|"

    # The `sumfact`-against-`current` pair stood here: two different functions on the atomic
    # sweep, recorded at 887 and 888 MDOF/s, which decided nothing and was never decided at
    # saturation either. One micro-kernel per kernel settles it instead of a measurement, and
    # `current` is in subpar/.
    #
    # --layout store is deliberately ABSENT from every residual and jac_action row above: its
    # branch reads `layout == "packed" || layout == "store"` and calls the packed sweep, so a
    # store row would be the packed row measured a second time. The recorded 2929-against-2940
    # "tie" between them was exactly that. Store appears once, under assembly, which is the
    # only operation it implements.

    # -- the generated arrangements are GONE from this sweep, and that is the result ------
    #
    # They were added here to decide their retirement on fresh evidence rather than on months-
    # old numbers, they lost (0.37x-0.57x of the hand-written action; the residual fastest
    # nowhere), and they moved to subpar/. The binary now refuses them by name, so leaving
    # them in the table costs 200 of 875 launches and 40 warning lines per sweep -- and a
    # sweep whose warnings are routine is a sweep whose warnings stop being read.
    #
    # The measurement that retired them is perf/campaign_generated_arms.csv, and re-running it
    # needs -DCVFEM_ENABLE_SUBPAR=ON plus these five lines back:
    #
    #   "residual_sympy|residual|packed|"          "residual_sympy|residual|atomic|"
    #   "jac_sympy_action{,_node,_comp,_face,_geom,_geomface}|jac_action|atomic|sympy_action{...}|"

    # -- the operator the solver actually evaluates, in stages --------------------------
    #
    # Rhie-Chow first because without it there is no pressure-pressure coupling at all and
    # the bare kernel is a smaller operator than anything the Newton loop sees. Then the
    # boundary closure, then the transient term: each row adds one term to the row above
    # it, so a difference between adjacent rows is that term's cost.
    "residual_rc|residual|packed|--rhie-chow"
    "residual_rc|residual|ecolor|--rhie-chow"
    "residual_rc|residual|atomic|--rhie-chow"
    "residual_rc_bnd|residual|packed|--rhie-chow --boundary"
    "residual_rc_bnd|residual|ecolor|--rhie-chow --boundary"
    "residual_rc_bnd|residual|atomic|--rhie-chow --boundary"
    # --transient takes the timestep. Its value does not change the cost -- the term adds
    # the same mass contribution whatever dt is -- so any non-zero one measures it.
    "residual_rc_bnd_dt|residual|packed|--rhie-chow --boundary --transient 1e-2"
    "residual_rc_bnd_dt|residual|ecolor|--rhie-chow --boundary --transient 1e-2"
    "residual_rc_bnd_dt|residual|atomic|--rhie-chow --boundary --transient 1e-2"
    # The nodal pressure gradient rebuilt inside every apply instead of hoisted out of the
    # Krylov solve. A full element sweep either way, so it is a stage in its own right.
    "residual_rc_perapply|residual|packed|--rhie-chow --pgrad-per-apply"
    "residual_rc_perapply|residual|atomic|--rhie-chow --pgrad-per-apply"
    "jac_rc|jac_action|packed|--rhie-chow"
    "jac_rc|jac_action|ecolor|--rhie-chow"
    "jac_rc|jac_action|atomic|--rhie-chow"
    "jac_rc_bnd|jac_action|packed|--rhie-chow --boundary"
    "jac_rc_bnd|jac_action|ecolor|--rhie-chow --boundary"
    "jac_rc_bnd|jac_action|atomic|--rhie-chow --boundary"

    # -- assembly, where the layout ranking is not the one above ------------------------
    #
    # `colored` is here and is not in the matrix-free rows, because assembly is the one
    # operation it wins and because it is the layout the solver assembles with whenever a
    # colouring exists (cvfem_hex8_ns_core.hpp). Leaving it out would make the layout the
    # solver runs the one layout nothing measures.
    "assemble|assemble|packed|"
    "assemble|assemble|atomic|"
    "assemble|assemble|colored|"
    "assemble|assemble|store|"
    # The generated assembly arrangements stood here, on both layouts, because their ranking
    # inverted with the layout -- generated won on atomic, hand-written on colored. That is
    # recorded in subpar/README.md; the surviving kernel is the hand-written one.
    # --assemble-diag is the atomic diagonal whatever --layout says, so it appears once.
    "assemble_diag|assemble_diag|atomic|"

    # -- the third way to apply the same Jacobian --------------------------------------
    #
    # The SpMV has no layout: it reads a matrix. --layout is passed only because the driver
    # needs one to build with, and the report ignores it for these rows. The two storage
    # precisions are the measurement.
    "spmv_f64|bsr_apply|packed|--bsr-precision double"
    "spmv_f32|bsr_apply|packed|--bsr-precision single"
)

op_flag() {
    case "$1" in
        residual)      echo "" ;;
        jac_action)    echo "--jac-action" ;;
        assemble)      echo "--assemble" ;;
        assemble_diag) echo "--assemble-diag" ;;
        bsr_apply)     echo "--bsr-apply" ;;
        *) echo "unknown operation: $1" >&2; exit 2 ;;
    esac
}

if [ "$MODE" = list ]; then
    printf '%s\n' "${CONFIGS[@]}" | column -t -s'|'
    echo
    echo "sizes: $SIZES   reps: $REPS   runs: $(( ${#CONFIGS[@]} * $(echo $SIZES | wc -w) * REPS ))"
    exit 0
fi

if [ -z "${BIN:-}" ]; then
    for d in build_omp build_alps build_ab build build_cuda; do
        if [ -x "$d/cvfem_hex8_ns_upwind_bench" ]; then BIN="$d/cvfem_hex8_ns_upwind_bench"; break; fi
    done
fi
[ -n "${BIN:-}" ] && [ -x "$BIN" ] || {
    echo "error: cvfem_hex8_ns_upwind_bench not found; build it or set BIN" >&2; exit 2; }

HOST="$(hostname -s 2>/dev/null || echo unknown)"
OUT="${OUT:-perf/campaign_${HOST}_$(date '+%Y%m%d')}"
mkdir -p "$OUT"
CSV="$OUT/campaign.csv"

echo "### cvfem layout campaign"
echo "### binary : $BIN"
echo "### host   : $HOST   $(date '+%Y-%m-%d %H:%M:%S')"
echo "### threads: $THREADS   sizes: $SIZES   reps: $REPS"
echo "### csv    : $CSV"
echo "### runs   : $(( ${#CONFIGS[@]} * $(echo $SIZES | wc -w) * REPS ))"

# The first invocation in an allocation is systematically off -- 12% here -- so one is
# thrown away before anything is recorded. Same discipline as the gate, same reason.
if [ "$MODE" = run ]; then
    echo "### discarding one warm-up invocation"
    OMP_NUM_THREADS="$THREADS" OMP_PROC_BIND=true OMP_PLACES=cores \
        "$BIN" --n 96 --repeat 3 --warmup 1 --layout packed >/dev/null 2>&1
fi

measure() {  # key operation layout n extra
    local key=$1 op=$2 layout=$3 n=$4 extra=${5:-}
    # shellcheck disable=SC2046,SC2086
    local -a cmd=(env OMP_NUM_THREADS="$THREADS" OMP_PROC_BIND=true OMP_PLACES=cores
                  "$BIN" --n "$n" --repeat 20 --warmup 3
                  --layout "$layout" $(op_flag "$op") $extra
                  --csv "$CSV" --tag "${key}_n${n}")
    if [ "$MODE" = dry ]; then
        printf '%s\n' "${cmd[*]}"
        return 0
    fi
    local rc=0
    timeout "$RUN_TIMEOUT" stdbuf -oL "${cmd[@]}" >/dev/null 2>&1 || rc=$?
    if [ "$rc" -eq 124 ]; then
        echo "### TIMEOUT: ${key} n=${n} layout=${layout} exceeded ${RUN_TIMEOUT}s -- absent from the csv"
    elif [ "$rc" -ne 0 ]; then
        echo "### WARNING: ${key} n=${n} layout=${layout} returned $rc -- absent from the csv"
    fi
}

for rep in $(seq 1 "$REPS"); do
    for n in $SIZES; do
        # Alternate the direction of travel through the list every repetition, so that
        # position-in-sweep is balanced between the two members of every layout pair rather
        # than always favouring whichever is written first.
        if [ $((rep % 2)) -eq 1 ]; then
            order=$(seq 0 $(( ${#CONFIGS[@]} - 1 )))
        else
            order=$(seq $(( ${#CONFIGS[@]} - 1 )) -1 0)
        fi
        for i in $order; do
            IFS='|' read -r key op layout extra <<<"${CONFIGS[$i]}"
            [ "$MODE" = run ] && echo "### rep $rep  n=$n  $key  $layout"
            measure "$key" "$op" "$layout" "$n" "$extra"
        done
    done
done

[ "$MODE" = dry ] && exit 0

echo
echo "### wrote $CSV  ($(( $(grep -c '' "$CSV" 2>/dev/null || echo 1) - 1 )) rows)"
echo "### report it with:"
echo "###   python3 python/cvfem_kernel_report.py $CSV -o docs/CVFEM_Kernels.md --html"
