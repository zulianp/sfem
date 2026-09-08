#ifndef CVFEM_PORTABILITY_HPP
#define CVFEM_PORTABILITY_HPP

// Portability shims shared by the CVFEM kernels, the generated SymPy kernels and
// the layout drivers. Include this before any CVFEM kernel header.
//
// The only thing in the CVFEM element kernels that is not already portable C++ is
// the accumulation into a shared destination. Every such site in the spike routes
// through one of three places:
//
//   1. atomic_add()                        - cvfem_hex8_layout_common.hpp
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
