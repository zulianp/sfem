// The cost of an OpenMP parallel region: an empty one, and a sweep with a sum over n doubles (one
// rank's share of a QP vector), median of 2000 calls after 200, the largest median over the ranks.
//   region_cost <n>     prints: ranks threads n empty_us sweep_us
#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

static int cmp(const void *a, const void *b) {
  const double x = *(const double *)a, y = *(const double *)b;
  return (x > y) - (x < y);
}

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  const long n = argc > 1 ? atol(argv[1]) : 150000;
  double *x = malloc(n * sizeof(double)), *y = malloc(n * sizeof(double));
#pragma omp parallel for
  for (long i = 0; i < n; i++) x[i] = y[i] = 1.0 / (i + 1);
  enum { WARM = 200, N = 2000 };
  static double t_empty[N], t_sweep[N];
  volatile double sink = 0;
  for (int it = 0; it < WARM + N; it++) {
    const double t0 = omp_get_wtime();
#pragma omp parallel
    { sink += 0; }
    const double t1 = omp_get_wtime();
    double s = 0;
#pragma omp parallel for reduction(+ : s)
    for (long i = 0; i < n; i++) {
      y[i] += 0.5 * x[i];
      s += y[i] * x[i];
    }
    const double t2 = omp_get_wtime();
    sink += s;
    if (it >= WARM) {
      t_empty[it - WARM] = t1 - t0;
      t_sweep[it - WARM] = t2 - t1;
    }
  }
  qsort(t_empty, N, sizeof(double), cmp);
  qsort(t_sweep, N, sizeof(double), cmp);
  double v[2] = {t_empty[N / 2] * 1e6, t_sweep[N / 2] * 1e6}, mx[2];
  MPI_Reduce(v, mx, 2, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  if (!rank) printf("%d %d %ld %.3f %.3f\n", size, omp_get_max_threads(), n, mx[0], mx[1]);
  MPI_Finalize();
  return 0;
}
