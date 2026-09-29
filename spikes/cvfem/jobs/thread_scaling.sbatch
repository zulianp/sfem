#!/usr/bin/env bash
#SBATCH --account=c40
#SBATCH --job-name=cvfem-tscale
#SBATCH --partition=debug
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=72
#SBATCH --time=00:29:00
#SBATCH --exclusive --hint=nomultithread --uenv-passthrough=ignore
#SBATCH --output=tscale_%j.out
#SBATCH --error=tscale_%j.err

# HOW THE THREE LAYOUTS SCALE ON ONE SOCKET, AND THE MACHINE'S MEMORY BOUND. (paper F6, F8)
#
# Two things in one allocation, because they belong together: the scaling curves and the STREAM
# triad that says what bandwidth those curves are competing for. Measuring them on separate
# nodes would let a roofline be drawn against a bound the kernel never saw -- and node-to-node
# variation here is 5-11%.
#
# What the curves are expected to distinguish, and why the layouts must be swept TOGETHER:
#
#   * the atomic sweep contends on shared nodes, and contention grows with the thread count, so
#     it should scale worst and worsen as threads are added;
#   * the colored sweep pays one barrier per color regardless of problem size, a fixed cost
#     amortised over less work as threads rise;
#   * the packed sweep has neither, but its parallel work unit is the PACK, so its scaling is
#     capped by the pack count -- which is the point the pack-size sweep makes separately.
#
# The last of those is why the pack count is printed beside every row rather than left implied:
# a packed curve that flattens because there are 123 packs for 72 threads is a statement about
# the pack size, not about the layout, and the two must not be confused.
#
# Thread counts are chosen to include values that do NOT divide the pack count, because static
# scheduling quantises imbalance and a sweep over only friendly counts would hide it.

set -uo pipefail
cd "$SLURM_SUBMIT_DIR" || exit 1
source scripts/alps_env.sh || exit 1

B=$PWD/build/cvfem_hex8_ns_upwind_bench
OUT=${OUT:-$SCRATCH/cvfem_tscale/${SLURM_JOB_ID:-manual}}
mkdir -p "$OUT"
CSV=$OUT/tscale.csv

N=${N:-128}
THREADS=${THREADS:-"1 2 4 8 18 36 54 72"}
REPEAT=${REPEAT:-12}
WARMUP=${WARMUP:-3}

echo "### host $(hostname)  n=$N  repeat=$REPEAT warmup=$WARMUP  $(date '+%F %T')"
echo

# ---------------------------------------------------------------------------------------- STREAM
# The memory bound the roofline is drawn against. Built here rather than trusted from a vendor
# figure, at a size well past the 117 MB of L3 on this die so the triad is genuinely streaming.
echo "### STREAM triad (the roofline's memory bound)"
STREAM_SRC=$(cd "$PWD/../.." && pwd)/baseline/stream/stream.cpp
if [ ! -f "$STREAM_SRC" ]; then
    echo "  stream source not found at $STREAM_SRC -- SKIPPED"
else
    # Compile and run as separate steps with the status checked. Chained through a pipe, a
    # failed compile returned success, the run then failed on a missing binary, and the filter
    # printed nothing -- so the job reported no bandwidth and no error either.
    srun -u --ntasks=1 --cpus-per-task=72 --uenv="$CVFEM_UENV" --view="$CVFEM_VIEW" \
      bash -lc "cd $OUT && g++ -O3 -fopenmp -DSTREAM_ARRAY_SIZE=200000000 -DNTIMES=20 -x c++ '$STREAM_SRC' -o stream.exe" \
      > "$OUT/stream_build.log" 2>&1
    if [ ! -x "$OUT/stream.exe" ]; then
        echo "  STREAM BUILD FAILED -- first lines of $OUT/stream_build.log:"
        sed -n '1,6p' "$OUT/stream_build.log" | sed 's/^/    /'
    else
        srun -u --ntasks=1 --cpus-per-task=72 --uenv="$CVFEM_UENV" --view="$CVFEM_VIEW" \
          bash -lc "cd $OUT && OMP_NUM_THREADS=72 OMP_PROC_BIND=true OMP_PLACES=cores ./stream.exe" \
          > "$OUT/stream_run.log" 2>&1
        if grep -aqE "^Triad:" "$OUT/stream_run.log"; then
            grep -aE "^(Function|Copy|Scale|Add|Triad):" "$OUT/stream_run.log" | sed 's/^/  /'
        else
            echo "  STREAM RUN PRODUCED NO TRIAD -- tail of $OUT/stream_run.log:"
            tail -6 "$OUT/stream_run.log" | sed 's/^/    /'
        fi
    fi
fi
echo

# ------------------------------------------------------------------------------------- scaling
echo "### thread scaling"
printf '%-10s %-5s %8s %9s %8s %10s %9s\n' layout op threads MDOF_s packs dofs sec_call
# THE PACK SIZE IS PINNED ACROSS THE SWEEP, and it has to be. The default now derives from the
# thread count (cvfem_default_pack_size), so leaving it alone gives every thread count a
# different pack size -- measured on the first run of this arm list: 256 packs at one thread and
# 2048 at 72. A scaling curve drawn through that is not a statement about the layout, which is
# the confound this job's header already warns about, so the size is taken ONCE at the full
# thread count and passed to every arm. Reading it from the binary rather than typing it keeps
# the sweep in step with the rule if the rule changes.
if [ -z "${PACK:-}" ]; then
    probe=$OUT/pack_probe.log
    srun -u --ntasks=1 --cpus-per-task=72 --uenv="$CVFEM_UENV" --view="$CVFEM_VIEW" \
      bash -lc "cd $PWD && OMP_NUM_THREADS=72 $B --n $N --repeat 1 --warmup 0 --layout packed --kernel sumfact" \
      > "$probe" 2>&1
    PACK=$(sed -n 's|^  pack_size: ||p' "$probe" | head -1)
fi
[ -n "${PACK:-}" ] || { echo "could not determine the pack size" >&2; exit 1; }
echo "### pack size pinned at $PACK elements for every thread count"
echo

# An explicit arm list rather than a cross product, for two reasons. The figure plots the
# residual arms only, so they run FIRST: if the job runs out of its debug window the arms lost
# are the ones nothing reports. And the cross product contains combinations that are either
# unreported (the coloured Jacobian action, which no macro in the paper quotes) or unavailable
# (the coloured sweep has no deferred-correction kernel), so they are absent by construction
# rather than measured and dropped downstream.
#
# resho is the higher-order convective flux through the Darwish-Moukalled limiter, carrying the
# same Rhie-Chow and boundary terms as the first-order arms beside it, so the curves differ in
# the scheme and in nothing else.
for arm in "packed|res" "atomic|res" \
           "packed|resho" "atomic|resho" \
           "packed|jac" "atomic|jac" \
           "colored|res"; do
    IFS='|' read -r layout op <<<"$arm"
    flags="--layout $layout --rhie-chow --boundary"
    [ "$layout" != atomic ] && flags="$flags --pack-size $PACK"
    [ "$op" = jac ] && flags="$flags --jac-action"
    [ "$op" = resho ] && flags="$flags --conv-ho 3"
    for t in $THREADS; do
        log=$OUT/${layout}_${op}_t${t}.log
        # Capped against the serial reference: thread 1 is the slowest arm by construction, so
        # a cap sized from it protects the wall clock without truncating a legitimate run.
        timeout "${RUN_TIMEOUT:-420}" \
          srun -u --ntasks=1 --cpus-per-task=72 --uenv="$CVFEM_UENV" --view="$CVFEM_VIEW" \
          bash -lc "cd $PWD && OMP_NUM_THREADS=$t OMP_PROC_BIND=true OMP_PLACES=cores $B --n $N --repeat $REPEAT --warmup $WARMUP $flags --csv $CSV --tag tscale_${layout}_${op}_t${t}" \
          > "$log" 2>&1
        # Anchored on the exact key, not a substring. Two of these were wrong when written by
        # pattern-guessing: the driver prints "seconds_per_apply" (not _per_call), and a loose
        # MDOF/s pattern also matches MDOF/s_element_visits and MDOF/s_residual, so a tail -1
        # silently picked a different quantity for one operator and not the other.
        val() { sed -n "s|^  $1: \\(.*\\)$|\\1|p" "$log" | head -1; }
        # The per-call time is under a different key for each operation: the residual prints
        # seconds_per_apply and the Jacobian action seconds_per_jac_action. Asking for one of
        # them for both left the column blank for half the rows.
        mdof=$(val 'MDOF/s'); packs=$(val 'n_packs'); dofs=$(val 'dofs')
        sec=$(val 'seconds_per_apply'); [ -n "$sec" ] || sec=$(val 'seconds_per_jac_action')
        printf '%-10s %-5s %8s %9s %8s %10s %9s\n' \
               "$layout" "$op" "$t" "${mdof:--}" "${packs:--}" "${dofs:--}" "${sec:--}"
    done
done

echo
echo "csv: $CSV"
echo "logs in $OUT"
