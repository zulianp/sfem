#!/usr/bin/env bash
#SBATCH --account=c40
#SBATCH --job-name=cvfem-peak
#SBATCH --partition=debug
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=72
#SBATCH --time=00:15:00
#SBATCH --exclusive --hint=nomultithread --uenv-passthrough=ignore
#SBATCH --output=peak_%j.out
#SBATCH --error=peak_%j.err

# THE COMPUTE ROOF OF THE ROOFLINE, MEASURED. (paper M9, completing F8)
#
# STREAM already supplies the memory roof (451 GB/s at 72 threads, job 4789412). This supplies
# the other one, so that neither roof in the paper's roofline is asserted.
#
# Measured with a large double-precision GEMM, which is the standard stand-in for sustained
# fp64 throughput and is what the vendor BLAS is tuned for: 2*N^3 flops against O(N^2) traffic,
# so at N = 8192 the operational intensity is ~1400 FLOP/byte and the kernel is unambiguously
# compute bound. An architectural figure would be cheaper -- 72 cores x 4x128-bit SVE2 x 2 for
# FMA is 16 FLOP/cycle/core -- but it needs an all-core clock that is not published for this
# part, and an earlier draft in this repository quotes a "sustained" fp64 figure LARGER than the
# nominal peak it also quotes, which is impossible and is exactly what inheriting such numbers
# leads to.
#
# GEMM is an upper bound on what any real kernel reaches; the roofline is drawn with it for that
# reason and the paper says so.
#
# Linked against the shared object by path rather than with -lopenblas. The uenv provides
# OpenBLAS to the solver through CMake and does not put it on the default link path, so -l
# fails with "cannot find -lopenblas"; /usr/lib64/libopenblas.so.0 is what the solver itself
# resolves to, per ldd.

set -uo pipefail
cd "$SLURM_SUBMIT_DIR" || exit 1
source scripts/alps_env.sh || exit 1

OUT=${OUT:-$SCRATCH/cvfem_peak/${SLURM_JOB_ID:-manual}}
mkdir -p "$OUT"
N=${N:-8192}
REPS=${REPS:-5}

cat > "$OUT/peak.c" <<'EOF'
/* Sustained fp64 rate from a large DGEMM. Timed after a warm-up; best of REPS reported,
   because background activity can only slow a run down. */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
void dgemm_(const char*,const char*,const int*,const int*,const int*,const double*,
            const double*,const int*,const double*,const int*,const double*,double*,const int*);
static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t);
                          return t.tv_sec + 1e-9*t.tv_nsec; }
int main(int argc,char**argv){
    int n = argc>1?atoi(argv[1]):8192, reps = argc>2?atoi(argv[2]):5;
    double *A=malloc((size_t)n*n*8), *B=malloc((size_t)n*n*8), *C=malloc((size_t)n*n*8);
    if(!A||!B||!C){ fprintf(stderr,"alloc failed for n=%d\n",n); return 1; }
    for(size_t i=0;i<(size_t)n*n;i++){ A[i]=1.0/(i%97+1); B[i]=1.0/(i%89+1); C[i]=0.0; }
    const double one=1.0; const char N_='N';
    dgemm_(&N_,&N_,&n,&n,&n,&one,A,&n,B,&n,&one,C,&n);       /* warm-up, untimed */
    double best=0;
    for(int r=0;r<reps;r++){
        double t0=now();
        dgemm_(&N_,&N_,&n,&n,&n,&one,A,&n,B,&n,&one,C,&n);
        double dt=now()-t0;
        double gf = 2.0*(double)n*n*n/dt*1e-9;
        if(gf>best) best=gf;
        printf("  rep %d: %.3f s  %.1f GFLOP/s\n", r+1, dt, gf);
    }
    printf("PEAK_FP64_GFLOPS: %.1f\n", best);
    printf("checksum: %.6e\n", C[0]+C[(size_t)n*n-1]);       /* keep the work alive */
    return 0;
}
EOF

echo "### host $(hostname)  n=$N reps=$REPS  $(date '+%F %T')"
srun -u --ntasks=1 --cpus-per-task=72 --uenv="$CVFEM_UENV" --view="$CVFEM_VIEW" \
  bash -lc "cd $OUT && gcc -O2 peak.c -o peak.exe /usr/lib64/libopenblas.so.0 -lm" > "$OUT/build.log" 2>&1
if [ ! -x "$OUT/peak.exe" ]; then
    echo "BUILD FAILED:"; sed -n '1,12p' "$OUT/build.log" | sed 's/^/  /'; exit 1
fi
srun -u --ntasks=1 --cpus-per-task=72 --uenv="$CVFEM_UENV" --view="$CVFEM_VIEW" \
  bash -lc "cd $OUT && OMP_NUM_THREADS=72 OMP_PROC_BIND=true OMP_PLACES=cores OPENBLAS_NUM_THREADS=72 ./peak.exe $N $REPS" \
  2>&1 | grep -aE "^(  rep|PEAK_FP64_GFLOPS|checksum)"
echo "logs in $OUT"
