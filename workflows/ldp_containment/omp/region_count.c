// LD_PRELOAD for a process linked with libgomp: counts the parallel regions it opens
// (GOMP_parallel and the combined parallel loops) and writes the total to
// $REGION_COUNT_DIR/<pid>.txt at exit.
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

typedef void (*fn_t)(void *);
static long regions;

void GOMP_parallel(fn_t fn, void *data, unsigned n, unsigned flags) {
  static void (*real)(fn_t, void *, unsigned, unsigned);
  if (!real) real = (void (*)(fn_t, void *, unsigned, unsigned))dlsym(RTLD_NEXT, "GOMP_parallel");
  __atomic_add_fetch(&regions, 1, __ATOMIC_RELAXED);
  real(fn, data, n, flags);
}

#define LOOP(name)                                                                              \
  void name(fn_t fn, void *data, unsigned n, long s, long e, long i, long c, unsigned f) {      \
    static void (*real)(fn_t, void *, unsigned, long, long, long, long, unsigned);              \
    if (!real) real = (void (*)(fn_t, void *, unsigned, long, long, long, long, unsigned))dlsym( \
                   RTLD_NEXT, #name);                                                           \
    __atomic_add_fetch(&regions, 1, __ATOMIC_RELAXED);                                          \
    real(fn, data, n, s, e, i, c, f);                                                           \
  }
LOOP(GOMP_parallel_loop_static)
LOOP(GOMP_parallel_loop_dynamic)
LOOP(GOMP_parallel_loop_guided)
LOOP(GOMP_parallel_loop_nonmonotonic_dynamic)
LOOP(GOMP_parallel_loop_nonmonotonic_guided)

__attribute__((destructor)) static void dump(void) {
  const char *dir = getenv("REGION_COUNT_DIR");
  if (!dir || !regions) return;
  char path[4096];
  snprintf(path, sizeof path, "%s/%d.txt", dir, (int)getpid());
  FILE *out = fopen(path, "w");
  if (!out) return;
  fprintf(out, "%ld\n", regions);
  fclose(out);
}
