#!/usr/bin/env bash
#SBATCH --account=c40
#SBATCH --job-name=cvfem-stream
#SBATCH --partition=debug
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=72
#SBATCH --time=00:15:00
#SBATCH --exclusive --hint=nomultithread --uenv-passthrough=ignore
#SBATCH --output=stream_%j.out
#SBATCH --error=stream_%j.err

# THE MEMORY BOUND THE ROOFLINE IS DRAWN AGAINST. (paper M6)
#
# Measured on the same machine as the kernels rather than taken from a vendor figure or from an
# earlier draft, because a roofline whose bound and whose points come from different nodes is
# not a roofline. The array is sized well past the shared L3 so the triad genuinely streams.
#
# Compile and run are separate steps with the status checked between them. Chained through a
# pipe they were not: a failed compile returned success, the run then failed on a missing
# binary, and the output filter printed nothing -- so the enclosing job reported neither a
# bandwidth nor an error, which is the worst of the three outcomes.
#
# Built as C, with gcc and -x c, although the file carries a .cpp extension. STREAM is C and
# declares the OpenMP entry points itself rather than including <omp.h>:
#     extern int omp_get_num_threads();
# Compiled as C++ that declaration is mangled, libgomp exports the unmangled C symbol, and the
# link fails with "undefined reference to omp_get_num_threads()" after pages of unrelated
# -Wwrite-strings warnings that make the real error easy to miss.

set -uo pipefail
cd "$SLURM_SUBMIT_DIR" || exit 1
source scripts/alps_env.sh || exit 1

OUT=${OUT:-$SCRATCH/cvfem_stream/${SLURM_JOB_ID:-manual}}
mkdir -p "$OUT"
SRC=$(cd "$PWD/../.." && pwd)/baseline/stream/stream.cpp
# 80e6 doubles per array, three arrays: ~1.9 GB, still 5.6x the 114 MB of shared L3 and so
# comfortably streaming, but inside what the aarch64 code model can address.
#
# 200e6 was the first choice and it does not link. STREAM's arrays are static globals, so three
# of them at 200e6 doubles put 4.8 GB in .bss, past the +/-4 GB reach of an ADRP relocation
# under the default (small) code model:
#     relocation truncated to fit: R_AARCH64_ADR_PREL_PG_HI21 against `.bss'
# -mcmodel=large is passed as well so that raising SIZE here does not resurrect the failure.
SIZE=${SIZE:-80000000}

echo "### host $(hostname)  $(date '+%F %T')  STREAM_ARRAY_SIZE=$SIZE"
[ -f "$SRC" ] || { echo "source not found: $SRC"; exit 1; }

srun -u --ntasks=1 --cpus-per-task=72 --uenv="$CVFEM_UENV" --view="$CVFEM_VIEW" \
  bash -lc "cd $OUT && gcc -O3 -fopenmp -mcmodel=large -DSTREAM_ARRAY_SIZE=$SIZE -DNTIMES=20 -x c '$SRC' -o stream.exe -lm" \
  > "$OUT/build.log" 2>&1
if [ ! -x "$OUT/stream.exe" ]; then
    echo "BUILD FAILED:"; sed -n '1,15p' "$OUT/build.log" | sed 's/^/  /'; exit 1
fi

for t in 72 36 18; do
    echo "### $t threads"
    srun -u --ntasks=1 --cpus-per-task=72 --uenv="$CVFEM_UENV" --view="$CVFEM_VIEW" \
      bash -lc "cd $OUT && OMP_NUM_THREADS=$t OMP_PROC_BIND=true OMP_PLACES=cores ./stream.exe" \
      > "$OUT/run_$t.log" 2>&1
    if grep -aqE "^Triad:" "$OUT/run_$t.log"; then
        grep -aE "^(Function|Copy|Scale|Add|Triad):" "$OUT/run_$t.log" | sed 's/^/  /'
    else
        echo "  NO TRIAD -- tail of run_$t.log:"; tail -8 "$OUT/run_$t.log" | sed 's/^/    /'
    fi
done
echo "logs in $OUT"
