# CVFEMCMakeFunctions.cmake
#
# Build options, the performance flag set, and the per-target helpers that apply them.
# Lifted out of CMakeLists.txt unchanged: the flag list below and its comments are
# load-bearing and were arrived at by measurement, so they are moved verbatim rather
# than tidied. Include this after find_package(SFEM) and before any target is declared.

option(CVFEM_ENABLE_SUBPAR "Build the quarantined CVFEM variants in subpar/" OFF)

# BLAS for the element-matrix gemm path. Off by default so the build stays dependency
# free, but worth turning on wherever this is benchmarked: without it
# packed_elements_matmul falls back to a triple loop, and a "gemm" measurement that is
# really a hand-rolled loop says nothing about the gemm. Accelerate supplies the Fortran
# BLAS symbols on Apple; elsewhere the standard BLAS package is used.
option(CVFEM_ENABLE_BLAS "Use BLAS for the element-matrix gemm path" OFF)
set(_cvfem_blas_libs)
if(CVFEM_ENABLE_BLAS)
    if(APPLE)
        find_library(ACCELERATE_FRAMEWORK Accelerate REQUIRED)
        set(_cvfem_blas_libs ${ACCELERATE_FRAMEWORK})
        message(STATUS "cvfem BLAS: Accelerate")
    else()
        find_package(BLAS REQUIRED)
        set(_cvfem_blas_libs ${BLAS_LIBRARIES})
        message(STATUS "cvfem BLAS: ${BLAS_LIBRARIES}")
    endif()
endif()

# Tracing is compiled into SFEM and smesh but the define does not reach consumers: neither
# package exports SMESH_ENABLE_TRACE as an INTERFACE_COMPILE_DEFINITIONS, so every
# SFEM_TRACE_SCOPE in this spike's own translation units expands to nothing and the kernels
# are invisible in the trace. The scopes are already there -- and already name the variant
# (apply_macro_local_hoisted vs apply_macro_local_affine vs apply_naive, residual vs
# residual_naive, block_diag vs block_diag_naive) -- so switching the define on is all that
# is needed to see which kernel runs and for how long.
#
# On by default, matching SFEM_ENABLE_TRACE and SMESH_ENABLE_TRACE upstream. The tracer keeps
# a map of name -> (calls, total) and writes it at exit; the per-scope cost is a clock read.
option(CVFEM_ENABLE_TRACE "Compile SFEM_TRACE_SCOPE in this spike's own sources" ON)
function(cvfem_trace_target tgt)
    if(CVFEM_ENABLE_TRACE)
        target_compile_definitions(${tgt} PRIVATE SMESH_ENABLE_TRACE SFEM_ENABLE_TRACE)
    endif()
endfunction()

function(cvfem_blas_target tgt)
    if(CVFEM_ENABLE_BLAS)
        target_compile_definitions(${tgt} PRIVATE SFEM_ENABLE_BLAS)
        target_link_libraries(${tgt} PRIVATE ${_cvfem_blas_libs})
    endif()
endfunction()

# Applied to every target that can reach a quarantined variant. Without the option the
# subpar/ headers are not on the include path at all, so a stray #include fails loudly
# rather than silently resurrecting a removed kernel.
function(cvfem_subpar_target tgt)
    if(CVFEM_ENABLE_SUBPAR)
        target_compile_definitions(${tgt} PRIVATE CVFEM_ENABLE_SUBPAR)
        target_include_directories(${tgt} PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/subpar
            ${CMAKE_CURRENT_SOURCE_DIR}/subpar/cuda)
    endif()
endfunction()

set(_cvfem_perf_opts)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    list(APPEND _cvfem_perf_opts
        -O3
        -DNDEBUG
        -ffast-math
        # Load-bearing, do not remove. -ffast-math implies -ffinite-math-only,
        # under which the compiler may assume no NaN and no infinity ever occurs.
        # It then folds every std::isfinite()/std::isnan() guard to a constant
        # `true` -- including the ones in drivers/cvfem_hex8_ns_steady.cpp that exist to
        # catch a singular preconditioner block or a diverged Newton residual.
        # The guards failed *open*: the solver proceeded on a NaN state instead
        # of reporting failure. Verified on the generated code, where the guard
        # compiled to `mov w0, #1; ret`.
        #
        # This cannot be worked around in the source. A bit test on the IEEE
        # exponent field folds away too, because the optimizer propagates the
        # assumption through the double-typed value before the bits are ever
        # read -- so does the same test behind a pointer, once the producing
        # arithmetic is visible in the same function.
        #
        # -fno-finite-math-only restores NaN/Inf handling while keeping the rest
        # of -ffast-math (reassociation, FMA contraction, no-signed-zeros,
        # reciprocal math).
        #
        # It is not codegen-neutral, and the change is in our favour. Measured on
        # 72 Grace cores, interleaved A/B, medians of 5: assembly is flat (sympy
        # 56.8 -> 56.2, sumfact 39.5 -> 39.5, store 74.0 -> 74.1, packed 54.9 ->
        # 54.6) while the packed matrix-free operators gain (residual 2320 ->
        # 2457, +5.9%; J*v 1873 -> 2028, +8.2%). The cause is visible in the
        # object code: cvfem_hex8_conv_all_simd and _conv_all_jv_simd stop being
        # emitted out of line (4328 and 6200 bytes -> 12-byte stubs) and are
        # inlined into the packed loops instead, and the large SymPy Jacobian
        # kernels shrink 10-17%. Worst cell measured was atomic residual at
        # -1.5%, inside this machine's ~4% run-to-run band.
        -fno-finite-math-only
        -funroll-loops
        -fomit-frame-pointer
        -fno-math-errno
        -fno-trapping-math
        -ffp-contract=fast)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|ARM64")
        list(APPEND _cvfem_perf_opts -mcpu=native -mtune=native)
    else()
        list(APPEND _cvfem_perf_opts -march=native -mtune=native)
    endif()
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|AppleClang")
        list(APPEND _cvfem_perf_opts -fvectorize -fslp-vectorize)
    else()
        list(APPEND _cvfem_perf_opts -ftree-vectorize)
    endif()
    message(STATUS "cvfem performance flags: ${_cvfem_perf_opts}")
endif()

# ---------------------------------------------------------------- target defaults
# Applied to every target so the nine hand-written copies of the same four lines cannot
# drift apart. cvfem_guard_selftest is the one deliberate exception below: it links
# nothing, because what it tests is the floating-point flags in isolation.
function(cvfem_target_defaults tgt)
    target_compile_options(${tgt} PRIVATE ${_cvfem_perf_opts})
    target_include_directories(${tgt} PRIVATE ${CVFEM_INCLUDE_DIRS})
    cvfem_subpar_target(${tgt})
    cvfem_trace_target(${tgt})
endfunction()
