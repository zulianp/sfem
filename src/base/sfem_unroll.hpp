#ifndef SFEM_UNROLL_H
#define SFEM_UNROLL_H

// unroll.h

// Helpers
#if defined(_MSC_VER)
  #define PRAGMA_MSVC(x) __pragma(x)
#endif
#define PRAGMA(x) _Pragma(#x)   // works on GCC/Clang/ICX; MSVC prefers __pragma

// OpenMP 5.1+ first (202011 = OpenMP 5.1 date macro)
#if defined(_OPENMP) && _OPENMP >= 202011
  #define SFEM_UNROLL           PRAGMA(omp unroll)
  #define SFEM_UNROLL_N(N)      PRAGMA(omp unroll factor(N))
  #define SFEM_UNROLL_PARTIAL   PRAGMA(omp unroll partial)
  #define NO_UNROLL        PRAGMA(omp unroll factor(1)) /* effectively none */
#else
  // Compiler-specific fallbacks
  #if defined(__CUDACC__)
    // CUDA device/host code compiled by NVCC
    #define SFEM_UNROLL         PRAGMA(unroll)
    #define SFEM_UNROLL_N(N)    PRAGMA(unroll N)
    #define SFEM_UNROLL_PARTIAL /* not available */ 
    #define NO_UNROLL      PRAGMA(nounroll)
  #elif defined(_MSC_VER)
    #define SFEM_UNROLL         PRAGMA_MSVC(loop(unroll))
    #define SFEM_UNROLL_N(N)    PRAGMA_MSVC(loop(unroll(N)))
    #define SFEM_UNROLL_PARTIAL /* not available */
    #define NO_UNROLL      PRAGMA_MSVC(loop(nounroll))
  #elif defined(__clang__) || defined(__INTEL_CLANG_COMPILER)
    #define SFEM_UNROLL         PRAGMA(unroll)
    #define SFEM_UNROLL_N(N)    PRAGMA(unroll N)
    #define SFEM_UNROLL_PARTIAL /* not available */
    #define NO_UNROLL      PRAGMA(nounroll)
  #elif defined(__GNUC__)
    // GCC 8+ supports these pragmas
    #define SFEM_UNROLL         /* no factor form without N -> choose a default if needed */
    #define SFEM_UNROLL_N(N)    PRAGMA(GCC unroll N)
    #define SFEM_UNROLL_PARTIAL /* not available */
    #define NO_UNROLL      PRAGMA(GCC nounroll)
  #else
    // Unknown compiler: make them no-ops
    #define SFEM_UNROLL
    #define SFEM_UNROLL_N(N)
    #define SFEM_UNROLL_PARTIAL
    #define NO_UNROLL
  #endif
#endif

#endif // SFEM_UNROLL_H
