#!/usr/bin/env bash
# Profile CVFEM residual (apply) and Jacobian (assemble) with xctrace / Instruments.
#
# Examples:
#   ./profile_xctrace.sh
#   ./profile_xctrace.sh --n 48 --layout packed --kernel sympy --open
#   ./profile_xctrace.sh --mode residual --template 'CPU Counters'
#   ./profile_xctrace.sh --bench ./build/cvfem_tet4_ns_upwind_bench --out /tmp/cvfem-traces
#   ./profile_xctrace.sh --n 48 --analyze
#
# Requires: Xcode Command Line Tools (xcrun xctrace). Open .trace files in Instruments.
# Optional analysis: analyze_xctrace_export.py (Time Profiler or CPU Profiler).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH="${SCRIPT_DIR}/build/cvfem_tet4_ns_upwind_bench"
OUT_DIR="${SCRIPT_DIR}/traces"
TEMPLATE="Time Profiler"
MODE="both" # residual | jacobian | both
N=48
REPEAT=20
WARMUP=3
LAYOUT="packed"
KERNEL="current"
PACK_SIZE=""
NO_SFC=0
OPEN_TRACES=0
TIME_LIMIT=""
QUIET=0
ANALYZE=0
RECORDED_TRACES=()

usage() {
    cat <<'EOF'
usage: profile_xctrace.sh [options]

  --bench PATH         Path to cvfem_tet4_ns_upwind_bench
  --out DIR            Output directory for .trace files (default: spikes/cvfem/traces)
  --template NAME      Instruments template (default: 'Time Profiler')
                       Common: 'Time Profiler', 'CPU Profiler', 'CPU Counters', 'System Trace'
  --mode MODE          residual | jacobian | both (default: both)
  --n N                Cube cells per dim (default: 48)
  --repeat N           Timed repetitions (default: 20)
  --warmup N           Warmup repetitions (default: 3)
  --layout NAME        packed | atomic (default: packed)
  --kernel NAME        current | sympy (default: current)
  --pack-size N        Elements per pack (forwarded if set)
  --no-sfc             Disable SFC reorder
  --time-limit T       Cap recording (e.g. 30s, 2m); optional
  --open               Open resulting .trace files in Instruments
  --analyze            After recording, run analyze_xctrace_export.py on traces
  --quiet              Pass --quiet to xctrace
  -h, --help           Show this help

Notes:
  - residual  -> apply (SpMV-free residual microkernel path)
  - jacobian  -> --assemble (BSR Jacobian assembly)
  - Prefer larger --n / --repeat so setup is a small fraction of the trace.
  - 'CPU Counters' may need a signed Instruments package / elevated privileges on some Macs.
  - --analyze works with 'Time Profiler' and 'CPU Profiler' (auto-detects schema).
EOF
}

die() {
    echo "error: $*" >&2
    exit 1
}

need_cmd() {
    command -v "$1" >/dev/null 2>&1 || die "missing required command: $1"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --bench)
            [[ $# -ge 2 ]] || die "--bench needs a path"
            BENCH="$2"
            shift 2
            ;;
        --out)
            [[ $# -ge 2 ]] || die "--out needs a directory"
            OUT_DIR="$2"
            shift 2
            ;;
        --template)
            [[ $# -ge 2 ]] || die "--template needs a name"
            TEMPLATE="$2"
            shift 2
            ;;
        --mode)
            [[ $# -ge 2 ]] || die "--mode needs residual|jacobian|both"
            MODE="$2"
            shift 2
            ;;
        --n)
            [[ $# -ge 2 ]] || die "--n needs a value"
            N="$2"
            shift 2
            ;;
        --repeat)
            [[ $# -ge 2 ]] || die "--repeat needs a value"
            REPEAT="$2"
            shift 2
            ;;
        --warmup)
            [[ $# -ge 2 ]] || die "--warmup needs a value"
            WARMUP="$2"
            shift 2
            ;;
        --layout)
            [[ $# -ge 2 ]] || die "--layout needs a value"
            LAYOUT="$2"
            shift 2
            ;;
        --kernel)
            [[ $# -ge 2 ]] || die "--kernel needs a value"
            KERNEL="$2"
            shift 2
            ;;
        --pack-size)
            [[ $# -ge 2 ]] || die "--pack-size needs a value"
            PACK_SIZE="$2"
            shift 2
            ;;
        --no-sfc)
            NO_SFC=1
            shift
            ;;
        --time-limit)
            [[ $# -ge 2 ]] || die "--time-limit needs a value like 30s"
            TIME_LIMIT="$2"
            shift 2
            ;;
        --open)
            OPEN_TRACES=1
            shift
            ;;
        --analyze)
            ANALYZE=1
            shift
            ;;
        --quiet)
            QUIET=1
            shift
            ;;
        -h | --help)
            usage
            exit 0
            ;;
        *)
            die "unknown argument: $1 (try --help)"
            ;;
    esac
done

case "$MODE" in
    residual | jacobian | both) ;;
    *) die "invalid --mode '$MODE' (expected residual|jacobian|both)" ;;
esac

need_cmd xcrun
[[ -x "$BENCH" ]] || die "bench not found or not executable: $BENCH
Build first, e.g.:
  cmake -S $SCRIPT_DIR -B $SCRIPT_DIR/build -DSFEM_DIR=<sfem-build>
  cmake --build $SCRIPT_DIR/build -j"

# Confirm xctrace is available (Xcode CLT / Xcode).
if ! xcrun xctrace list templates >/dev/null 2>&1; then
    die "xctrace unavailable. Install Xcode or Command Line Tools."
fi

stamp="$(date +%Y%m%d-%H%M%S)"
safe_template="$(echo "$TEMPLATE" | tr ' /' '__')"
run_tag="n${N}_${LAYOUT}_${KERNEL}_${safe_template}_${stamp}"
mkdir -p "$OUT_DIR"

bench_args=(--n "$N" --repeat "$REPEAT" --warmup "$WARMUP" --layout "$LAYOUT" --kernel "$KERNEL")
if [[ -n "$PACK_SIZE" ]]; then
    bench_args+=(--pack-size "$PACK_SIZE")
fi
if [[ "$NO_SFC" -eq 1 ]]; then
    bench_args+=(--no-sfc)
fi

record_one() {
    local label="$1"
    shift
    local -a extra_args=()
    if [[ $# -gt 0 ]]; then
        extra_args=("$@")
    fi
    local out_trace="${OUT_DIR}/cvfem_${label}_${run_tag}.trace"
    local stdout_log="${OUT_DIR}/cvfem_${label}_${run_tag}.stdout.txt"

    local -a xctrace_args=(
        xctrace record
        --template "$TEMPLATE"
        --output "$out_trace"
        --target-stdout "$stdout_log"
        --no-prompt
    )
    if [[ "$QUIET" -eq 1 ]]; then
        xctrace_args+=(--quiet)
    fi
    if [[ -n "$TIME_LIMIT" ]]; then
        xctrace_args+=(--time-limit "$TIME_LIMIT")
    fi

    echo "==> recording $label"
    echo "    template: $TEMPLATE"
    if [[ ${#extra_args[@]} -gt 0 ]]; then
        echo "    command:  $BENCH ${bench_args[*]} ${extra_args[*]}"
    else
        echo "    command:  $BENCH ${bench_args[*]}"
    fi
    echo "    output:   $out_trace"

    if [[ ${#extra_args[@]} -gt 0 ]]; then
        xcrun "${xctrace_args[@]}" --launch -- "$BENCH" "${bench_args[@]}" "${extra_args[@]}"
    else
        xcrun "${xctrace_args[@]}" --launch -- "$BENCH" "${bench_args[@]}"
    fi

    if [[ ! -e "$out_trace" ]]; then
        die "expected trace was not created: $out_trace"
    fi

    echo "    done:     $out_trace"
    RECORDED_TRACES+=("$out_trace")
    if [[ -f "$stdout_log" ]]; then
        echo "    stdout:   $stdout_log"
        # Surface key throughput lines if present.
        if command -v rg >/dev/null 2>&1; then
            rg -n "seconds_per_|MELEM|GFLOP|elements:|OpenMP" "$stdout_log" || true
        else
            grep -E "seconds_per_|MELEM|GFLOP|elements:|OpenMP" "$stdout_log" || true
        fi
    fi

    if [[ "$OPEN_TRACES" -eq 1 ]]; then
        open "$out_trace"
    fi
}

find_python() {
    if [[ -x "${SCRIPT_DIR}/../../venv/bin/python" ]]; then
        echo "${SCRIPT_DIR}/../../venv/bin/python"
    elif command -v python3 >/dev/null 2>&1; then
        command -v python3
    else
        die "python3 not found (expected repo venv or python3)"
    fi
}

echo "Profiling CVFEM with Instruments (xctrace)"
echo "  bench:    $BENCH"
echo "  out_dir:  $OUT_DIR"
echo "  mode:     $MODE"
echo "  template: $TEMPLATE"
echo

if [[ "$MODE" == "residual" || "$MODE" == "both" ]]; then
    record_one residual
fi

if [[ "$MODE" == "jacobian" || "$MODE" == "both" ]]; then
    record_one jacobian --assemble
fi

echo
echo "Open traces in Instruments:"
echo "  open ${OUT_DIR}/cvfem_*_${run_tag}.trace"
echo
echo "Useful Instruments views:"
echo "  Time Profiler  -> Call Tree (invert, hide system libs)"
echo "  CPU Profiler   -> Hot spots by sample count"
echo "  CPU Counters   -> L1/L2 misses if counters are configured"
echo
echo "List templates:"
echo "  xcrun xctrace list templates"

if [[ "$ANALYZE" -eq 1 ]]; then
    if [[ ${#RECORDED_TRACES[@]} -eq 0 ]]; then
        die "--analyze set but no traces were recorded"
    fi
    if [[ "$TEMPLATE" != "Time Profiler" && "$TEMPLATE" != "CPU Profiler" ]]; then
        echo
        echo "warning: --analyze best supports 'Time Profiler' / 'CPU Profiler' (got '$TEMPLATE')."
    fi
    ANALYZER="${SCRIPT_DIR}/analyze_xctrace_export.py"
    [[ -f "$ANALYZER" ]] || die "missing analyzer: $ANALYZER"
    PY="$(find_python)"
    echo
    echo "==> analyzing traces with $ANALYZER"
    "$PY" "$ANALYZER" --export-dir "${OUT_DIR}/export" --force-export "${RECORDED_TRACES[@]}"
else
    echo
    echo "Analyze later:"
    echo "  ${SCRIPT_DIR}/../../venv/bin/python ${SCRIPT_DIR}/analyze_xctrace_export.py ${OUT_DIR}/cvfem_*_${run_tag}.trace"
fi
