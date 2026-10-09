#ifndef CVFEM_PORTABILITY_HPP
#define CVFEM_PORTABILITY_HPP

// (moved from src/core/) This is the kernels' own wrapper over the few things that differ between
// host and device -- SFEM_HOST_DEVICE, the restrict and inline spellings, the atomic add. It
// includes nothing and names no library, which is why DESIGN.md's "no library dependencies" for
// src/kernels/ can admit it: the clause's own exception is "CUDA, OpenMP or other wrappers", and
// this is that wrapper. It sat in src/core/, so every kernel header reached outside the directory
// for it.

// Portability shims shared by the CVFEM kernels, the generated SymPy kernels and
// the layout drivers. Include this before any CVFEM kernel header.
//
// The only thing in the CVFEM element kernels that is not already portable C++ is
// the accumulation into a shared destination. Every such site in the spike routes
// through one of three places:
//
//   1. atomic_add()                        - cvfem_hex8_best_common.hpp
//   2. cvfem_hex8_acc<true>                - cvfem_hex8_ns_upwind_kernels.hpp
//   3. the emit line in                    - synthesize_cvfem_hex8_ns_upwind_sympy.py
//      synthesize_..._sympy.py, which
//      produces 4320 sites in the
//      generated header
//
// Routing all three through CVFEM_ATOMIC_ADD makes the whole body of kernel code
// compile unchanged for the host and the device.

// Host/device qualification.
//
// This mirrors the definition added to base/sfem_base.hpp, but is repeated here
// under a guard because the spike compiles against an *installed* SFEM
// (find_package(SFEM CONFIG REQUIRED)), whose headers may predate that addition.
// The guard means the library definition wins once SFEM is rebuilt, and the
// spike keeps building against an older install in the meantime.
#ifndef SFEM_HOST_DEVICE
#if defined(__CUDACC__) || defined(__HIPCC__)
#define SFEM_HOST_DEVICE __host__ __device__
#else
#define SFEM_HOST_DEVICE
#endif
#endif

#ifndef SFEM_DEVICE_INLINE
#define SFEM_DEVICE_INLINE SFEM_HOST_DEVICE inline
#endif

// THE OTHER THREE, HERE AND NOWHERE ELSE.
//
// SFEM_RESTRICT, SFEM_INLINE and SFEM_NOINLINE had nine definitions between them across the
// spike, all under #ifndef, so the effective one was decided by include order. Two of the nine
// disagreed with the rest, both in cvfem_venkata_limiter.hpp, which the HEX8 kernels header
// includes at its line 5 -- twenty-five lines before this file:
//
//   * `#define SFEM_INLINE inline`, without always_inline. Every other site spells it
//     `inline __attribute__((always_inline))`. Which one a kernel got depended on whether its
//     translation unit reached a family header before the limiter.
//   * `#define SFEM_HOST_DEVICE` unconditionally, with no __CUDACC__ test. That one is not a
//     weaker spelling but a wrong one: it made SFEM_HOST_DEVICE empty in EVERY CUDA translation
//     unit, so every kernel was host-only and the device smoke test could not compile at all --
//     "calling a __host__ function from a __global__ function". Nothing caught it because CUDA
//     is not built on the development machine.
//
// So they are defined here, with this file included where they are needed, rather than
// re-spelled per site.
#ifndef SFEM_RESTRICT
#define SFEM_RESTRICT __restrict__
#endif

#ifndef SFEM_INLINE
#define SFEM_INLINE inline __attribute__((always_inline))
#endif

#ifndef SFEM_NOINLINE
#define SFEM_NOINLINE __attribute__((noinline))
#endif

// clang-format off
#if defined(__CUDA_ARCH__)
// Device: native atomicAdd. Requires sm_60+ for the double overload.
#define CVFEM_ATOMIC_ADD(dst, val) atomicAdd(&(dst), (val))
#elif defined(_OPENMP)
// Host, threaded. `_Pragma` applies to the statement that follows it, so this
// expands to exactly the `#pragma omp atomic update` / `+=` pair it replaces.
#define CVFEM_ATOMIC_ADD(dst, val)      \
    do {                                \
        _Pragma("omp atomic update")    \
        (dst) += (val);                 \
    } while (0)
#else
// Host, serial.
#define CVFEM_ATOMIC_ADD(dst, val) \
    do {                           \
        (dst) += (val);            \
    } while (0)
#endif
// clang-format on

#endif  // CVFEM_PORTABILITY_HPP
