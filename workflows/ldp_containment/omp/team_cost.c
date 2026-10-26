// A QP iteration's synchronisation, region per pass against one persistent region. The iteration
// is the contact QP's CG step on n doubles per rank, with A = S T S (T tridiagonal, SPD) in place of
// the frames and the BSR product:
//   regions:  six regions (update, from_frame, product, to_frame, curvature, step), as the solver
//   fused:    three (update + from_frame; product + to_frame + curvature; step)
//   team:     one region per sample, three omp barriers, sums from padded per-thread slots in a fixed
//             order, every thread taking the same decisions
//   diss:     team with a dissemination barrier (release/acquire flags, one writer each)
//   work:     team without barriers: the passes alone (timing only, its results are discarded)
//   chunkK:   team whose sums are over K chunks fixed by n alone, as OpenMPBLAS's deterministic
//             dot: each thread takes whole chunks, writes a partial per chunk and, after the
//             barrier, sums all K in index order (the result is the same for any thread count)
// With ranks, the sums are also allreduced (team: thread 0, then one more barrier). Median of 40
// samples of 100 iterations after 4, the largest median over the ranks, us per iteration.
//   team_cost <n>     prints: ranks threads n regions_us fused_us team_us diss_us work_us
//                     chunk256_us chunk1024_us chunk4096_us max_rel_diff
#include <math.h>
#include <mpi.h>
#include <omp.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { MAXT = 288, ROUNDS = 9, WARM = 4, SAMPLES = 40, ITERS = 100, MAXC = 4096, NV = 8 };

typedef struct {
  _Alignas(64) double v[4];
  double pad[4];
} slot_t;

typedef struct {
  _Alignas(64) atomic_long v;
  char pad[56];
} flag_t;

static slot_t slots[2][MAXT];
static flag_t flags[ROUNDS][MAXT];
static double shared_sums[2][4];
static double chunk_sums[2][MAXC][4];

static inline void relax(void) {
#if defined(__aarch64__)
  __asm__ volatile("yield");
#elif defined(__x86_64__)
  __asm__ volatile("pause");
#endif
}

static void diss_barrier(const int tid, const int nt, long *const epoch) {
  const long e = ++*epoch;
  for (int k = 0, d = 1; d < nt; k++, d <<= 1) {
    atomic_store_explicit(&flags[k][(tid + d) % nt].v, e, memory_order_release);
    while (atomic_load_explicit(&flags[k][tid].v, memory_order_acquire) < e) relax();
  }
}

static int cmp(const void *a, const void *b) {
  const double x = *(const double *)a, y = *(const double *)b;
  return (x > y) - (x < y);
}

typedef struct {
  long n;
  double *x, *g, *p, *gf, *s, *fr, *Ap, *b, *ub;
} state_t;

// T: 2 on the diagonal, -0.999 off it.
static inline double tri(const double *const f, const long i, const long n) {
  const double l = i > 0 ? f[i - 1] : 0, r = i + 1 < n ? f[i + 1] : 0;
  return 2 * f[i] - 0.999 * (l + r);
}

// x = 0, g = -b, gf = g, p = 0, beta = 0: the first update makes p = gf.
static void reset(state_t *const st, const long b, const long e) {
  for (long i = b; i < e; i++) {
    st->x[i] = 0;
    st->g[i] = -st->b[i];
    st->gf[i] = st->g[i];
    st->p[i] = 0;
    st->fr[i] = 0;
    st->Ap[i] = 0;
  }
}

static void range(const long n, const int tid, const int nt, long *const b, long *const e) {
  const long q = n / nt, r = n % nt;
  *b = tid * q + (tid < r ? tid : r);
  *e = *b + q + (tid < r);
}

static double sum_all(const double l, const int size) {
  double g = l;
  if (size > 1) MPI_Allreduce(&l, &g, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  return g;
}

static void allreduce4(double *const v, const int size, const int min_last) {
  if (size == 1) return;
  double s[3] = {v[0], v[1], v[2]}, m = v[3];
  MPI_Allreduce(MPI_IN_PLACE, s, 3, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  if (min_last)
    MPI_Allreduce(MPI_IN_PLACE, &m, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
  else
    MPI_Allreduce(MPI_IN_PLACE, &m, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  v[0] = s[0], v[1] = s[1], v[2] = s[2], v[3] = m;
}

// Region per pass (fused = 0) or three regions (fused = 1).
static void run_regions(state_t *const st, const int fused, const int size) {
  const long n = st->n;
  double beta = 0;
  for (int it = 0; it < ITERS; it++) {
    double c[4] = {0, 0, 0, INFINITY};
    double dAd = 0, dd = 0, gd = 0, am = INFINITY;
    if (fused) {
#pragma omp parallel for schedule(static)
      for (long i = 0; i < n; i++) {
        st->p[i] = st->gf[i] - beta * st->p[i];
        st->fr[i] = st->s[i] * st->p[i];
      }
#pragma omp parallel for schedule(static) reduction(+ : dAd, dd, gd) reduction(min : am)
      for (long i = 0; i < n; i++) {
        const double a = st->s[i] * tri(st->fr, i, n), pi = st->p[i];
        st->Ap[i] = a;
        dAd += pi * a;
        dd += pi * pi;
        gd += st->g[i] * pi;
        am = fmin(am, pi > 0 ? (st->ub[i] - st->x[i]) / pi : INFINITY);
      }
    } else {
#pragma omp parallel for schedule(static)
      for (long i = 0; i < n; i++) st->p[i] = st->gf[i] - beta * st->p[i];
#pragma omp parallel for schedule(static)
      for (long i = 0; i < n; i++) st->fr[i] = st->s[i] * st->p[i];
#pragma omp parallel for schedule(static)
      for (long i = 0; i < n; i++) st->Ap[i] = tri(st->fr, i, n);
#pragma omp parallel for schedule(static)
      for (long i = 0; i < n; i++) st->Ap[i] *= st->s[i];
#pragma omp parallel for schedule(static) reduction(+ : dAd, dd, gd) reduction(min : am)
      for (long i = 0; i < n; i++) {
        const double a = st->Ap[i], pi = st->p[i];
        dAd += pi * a;
        dd += pi * pi;
        gd += st->g[i] * pi;
        am = fmin(am, pi > 0 ? (st->ub[i] - st->x[i]) / pi : INFINITY);
      }
    }
    c[0] = dAd, c[1] = dd, c[2] = gd, c[3] = am;
    allreduce4(c, size, 1);
    const double alpha = fmin(c[2] / c[0], c[3]);
    double Adgf = 0, gf2 = 0, gc2 = 0, gp2 = 0;
#pragma omp parallel for schedule(static) reduction(+ : Adgf, gf2, gc2, gp2)
    for (long i = 0; i < n; i++) {
      const double xi = st->x[i] - alpha * st->p[i], gi = st->g[i] - alpha * st->Ap[i];
      const double f = xi < st->ub[i] ? gi : 0, ch = gi - f;
      st->x[i] = xi;
      st->g[i] = gi;
      st->gf[i] = f;
      Adgf += st->Ap[i] * f;
      gf2 += f * f;
      gc2 += ch * ch;
      gp2 += (f + ch) * (f + ch);
    }
    double v[4] = {Adgf, gf2, gc2, gp2};
    allreduce4(v, size, 0);
    beta = v[0] / c[0];
  }
}

// mode 0: omp barrier, 1: dissemination barrier, 2: no barrier (timing only).
static void run_team(state_t *const st, const int mode, const int size) {
  const long n = st->n;
#pragma omp parallel
  {
    const int tid = omp_get_thread_num(), nt = omp_get_num_threads();
    long b, e, epoch = 0;
    range(n, tid, nt, &b, &e);
#define BARRIER()                              \
  do {                                         \
    if (mode == 0) {                           \
      _Pragma("omp barrier")                   \
    } else if (mode == 1)                      \
      diss_barrier(tid, nt, &epoch);           \
  } while (0)
    // The flags count epochs from 0 in every sample.
    if (mode == 1) {
      for (int k = 0; k < ROUNDS; k++) atomic_store(&flags[k][tid].v, 0);
    }
#pragma omp barrier
    double beta = 0;
    for (int it = 0; it < ITERS; it++) {
      for (long i = b; i < e; i++) {
        st->p[i] = st->gf[i] - beta * st->p[i];
        st->fr[i] = st->s[i] * st->p[i];
      }
      BARRIER();
      double dAd = 0, dd = 0, gd = 0, am = INFINITY;
      for (long i = b; i < e; i++) {
        const double a = st->s[i] * tri(st->fr, i, n), pi = st->p[i];
        st->Ap[i] = a;
        dAd += pi * a;
        dd += pi * pi;
        gd += st->g[i] * pi;
        am = fmin(am, pi > 0 ? (st->ub[i] - st->x[i]) / pi : INFINITY);
      }
      slot_t *const s0 = slots[0];
      s0[tid].v[0] = dAd, s0[tid].v[1] = dd, s0[tid].v[2] = gd, s0[tid].v[3] = am;
      BARRIER();
      double c[4] = {0, 0, 0, INFINITY};
      for (int t = 0; t < nt; t++) {
        c[0] += s0[t].v[0];
        c[1] += s0[t].v[1];
        c[2] += s0[t].v[2];
        c[3] = fmin(c[3], s0[t].v[3]);
      }
      if (size > 1) {
        if (tid == 0) {
          allreduce4(c, size, 1);
          memcpy(shared_sums[0], c, sizeof(c));
        }
        BARRIER();
        memcpy(c, shared_sums[0], sizeof(c));
      }
      const double alpha = fmin(c[2] / c[0], c[3]);
      double Adgf = 0, gf2 = 0, gc2 = 0, gp2 = 0;
      for (long i = b; i < e; i++) {
        const double xi = st->x[i] - alpha * st->p[i], gi = st->g[i] - alpha * st->Ap[i];
        const double f = xi < st->ub[i] ? gi : 0, ch = gi - f;
        st->x[i] = xi;
        st->g[i] = gi;
        st->gf[i] = f;
        Adgf += st->Ap[i] * f;
        gf2 += f * f;
        gc2 += ch * ch;
        gp2 += (f + ch) * (f + ch);
      }
      slot_t *const s1 = slots[1];
      s1[tid].v[0] = Adgf, s1[tid].v[1] = gf2, s1[tid].v[2] = gc2, s1[tid].v[3] = gp2;
      BARRIER();
      double v[4] = {0, 0, 0, 0};
      for (int t = 0; t < nt; t++)
        for (int k = 0; k < 4; k++) v[k] += s1[t].v[k];
      if (size > 1) {
        if (tid == 0) {
          allreduce4(v, size, 0);
          memcpy(shared_sums[1], v, sizeof(v));
        }
        BARRIER();
        memcpy(v, shared_sums[1], sizeof(v));
      }
      beta = v[0] / c[0];
    }
#undef BARRIER
  }
}

// Sums over nc chunks fixed by n; thread t takes chunks [t nc / nt, (t + 1) nc / nt).
static void run_chunks(state_t *const st, const int nc, const int size) {
  const long n = st->n;
#pragma omp parallel
  {
    const int tid = omp_get_thread_num(), nt = omp_get_num_threads();
    const int c0 = (int)((long)tid * nc / nt), c1 = (int)((long)(tid + 1) * nc / nt);
    long b, e, unused;
    range(n, c0, nc, &b, &unused);
    range(n, c1 - 1, nc, &unused, &e);
    if (c1 == c0) b = e = 0;
    double beta = 0;
    for (int it = 0; it < ITERS; it++) {
      for (long i = b; i < e; i++) {
        st->p[i] = st->gf[i] - beta * st->p[i];
        st->fr[i] = st->s[i] * st->p[i];
      }
#pragma omp barrier
      for (int k = c0; k < c1; k++) {
        long cb, ce;
        range(n, k, nc, &cb, &ce);
        double dAd = 0, dd = 0, gd = 0, am = INFINITY;
        for (long i = cb; i < ce; i++) {
          const double a = st->s[i] * tri(st->fr, i, n), pi = st->p[i];
          st->Ap[i] = a;
          dAd += pi * a;
          dd += pi * pi;
          gd += st->g[i] * pi;
          am = fmin(am, pi > 0 ? (st->ub[i] - st->x[i]) / pi : INFINITY);
        }
        chunk_sums[0][k][0] = dAd, chunk_sums[0][k][1] = dd, chunk_sums[0][k][2] = gd, chunk_sums[0][k][3] = am;
      }
#pragma omp barrier
      double c[4] = {0, 0, 0, INFINITY};
      for (int k = 0; k < nc; k++) {
        c[0] += chunk_sums[0][k][0];
        c[1] += chunk_sums[0][k][1];
        c[2] += chunk_sums[0][k][2];
        c[3] = fmin(c[3], chunk_sums[0][k][3]);
      }
      if (size > 1) {
        if (tid == 0) {
          allreduce4(c, size, 1);
          memcpy(shared_sums[0], c, sizeof(c));
        }
#pragma omp barrier
        memcpy(c, shared_sums[0], sizeof(c));
      }
      const double alpha = fmin(c[2] / c[0], c[3]);
      for (int k = c0; k < c1; k++) {
        long cb, ce;
        range(n, k, nc, &cb, &ce);
        double Adgf = 0, gf2 = 0, gc2 = 0, gp2 = 0;
        for (long i = cb; i < ce; i++) {
          const double xi = st->x[i] - alpha * st->p[i], gi = st->g[i] - alpha * st->Ap[i];
          const double f = xi < st->ub[i] ? gi : 0, ch = gi - f;
          st->x[i] = xi;
          st->g[i] = gi;
          st->gf[i] = f;
          Adgf += st->Ap[i] * f;
          gf2 += f * f;
          gc2 += ch * ch;
          gp2 += (f + ch) * (f + ch);
        }
        chunk_sums[1][k][0] = Adgf, chunk_sums[1][k][1] = gf2, chunk_sums[1][k][2] = gc2, chunk_sums[1][k][3] = gp2;
      }
#pragma omp barrier
      double v[4] = {0, 0, 0, 0};
      for (int k = 0; k < nc; k++)
        for (int j = 0; j < 4; j++) v[j] += chunk_sums[1][k][j];
      if (size > 1) {
        if (tid == 0) {
          allreduce4(v, size, 0);
          memcpy(shared_sums[1], v, sizeof(v));
        }
#pragma omp barrier
        memcpy(v, shared_sums[1], sizeof(v));
      }
      beta = v[0] / c[0];
    }
  }
}

int main(int argc, char **argv) {
  int provided = 0;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (omp_get_max_threads() > MAXT || (1 << ROUNDS) < omp_get_max_threads()) {
    fprintf(stderr, "team_cost: at most %d threads\n", MAXT);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  state_t st;
  st.n = argc > 1 ? atol(argv[1]) : 150000;
  const long n = st.n;
  double **arrays[] = {&st.x, &st.g, &st.p, &st.gf, &st.s, &st.fr, &st.Ap, &st.b, &st.ub};
  for (size_t k = 0; k < sizeof(arrays) / sizeof(arrays[0]); k++) *arrays[k] = malloc(n * sizeof(double));
  // First touch with the static partition every variant uses.
#pragma omp parallel
  {
    long b, e;
    range(n, omp_get_thread_num(), omp_get_num_threads(), &b, &e);
    for (long i = b; i < e; i++) {
      st.s[i] = 1 / sqrt(1 + (i % 7));
      st.b[i] = sin(0.001 * (i + rank * n));
      st.ub[i] = 1e30;
    }
    reset(&st, b, e);
  }

  // Checksums of x after one sample, against the region per pass.
  double ref = 0, maxdiff = 0;
  static double t[NV][SAMPLES];
  const int chunks[3] = {256, 1024, 4096};
  for (int variant = 0; variant < NV; variant++) {
    for (int sample = 0; sample < WARM + SAMPLES; sample++) {
#pragma omp parallel
      {
        long b, e;
        range(n, omp_get_thread_num(), omp_get_num_threads(), &b, &e);
        reset(&st, b, e);
      }
      MPI_Barrier(MPI_COMM_WORLD);
      const double t0 = omp_get_wtime();
      if (variant < 2)
        run_regions(&st, variant, size);
      else if (variant < 5)
        run_team(&st, variant - 2, size);
      else
        run_chunks(&st, chunks[variant - 5], size);
      const double t1 = omp_get_wtime();
      if (sample >= WARM) t[variant][sample - WARM] = (t1 - t0) / ITERS;
    }
    if (variant == 4) continue;
    double l = 0;
#pragma omp parallel for reduction(+ : l)
    for (long i = 0; i < n; i++) l += fabs(st.x[i]);
    const double cs = sum_all(l, size);
    if (variant == 0)
      ref = cs;
    else
      maxdiff = fmax(maxdiff, fabs(cs - ref) / fabs(ref));
    if (!isfinite(cs)) {
      if (!rank) fprintf(stderr, "team_cost: variant %d is not finite\n", variant);
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
  }
  double med[NV], mx[NV];
  for (int v = 0; v < NV; v++) {
    qsort(t[v], SAMPLES, sizeof(double), cmp);
    med[v] = t[v][SAMPLES / 2] * 1e6;
  }
  MPI_Reduce(med, mx, NV, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  if (!rank)
    printf("%d %d %ld %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.2e\n", size, omp_get_max_threads(), n, mx[0], mx[1],
           mx[2], mx[3], mx[4], mx[5], mx[6], mx[7], maxdiff);
  MPI_Finalize();
  return 0;
}
