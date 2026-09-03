cmake_minimum_required(VERSION 3.20)

project(cvfem_tet4_ns_upwind_bench LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)

if(DEFINED SFEM_DIR)
    foreach(_dep ryml matrixio smesh SCCD ssdf)
        if(NOT DEFINED ${_dep}_DIR AND EXISTS "${SFEM_DIR}/${_dep}/${_dep}Config.cmake")
            set(${_dep}_DIR "${SFEM_DIR}/${_dep}")
        endif()
    endforeach()
endif()

find_package(ryml CONFIG REQUIRED)
find_package(matrixio CONFIG REQUIRED)
find_package(smesh CONFIG REQUIRED)
find_package(SCCD CONFIG REQUIRED)
find_package(ssdf CONFIG REQUIRED)
find_package(SFEM CONFIG REQUIRED)

# Header-only CUDA Thrust is sometimes exported by SFEM as -lThrust.
if(NOT TARGET Thrust)
    add_library(Thrust INTERFACE IMPORTED)
endif()

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

find_package(OpenMP COMPONENTS CXX)
enable_testing()

option(CVFEM_ENABLE_SUBPAR "Build the quarantined CVFEM variants in subpar/" OFF)

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
        # `true` -- including the ones in cvfem_hex8_ns_steady.cpp that exist to
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

foreach(_cvfem_tgt cvfem_tet4_ns_upwind_bench cvfem_tet4_ns_upwind_bench_baseline cvfem_hex8_ns_upwind_bench)
    add_executable(${_cvfem_tgt} ${_cvfem_tgt}.cpp)
    target_link_libraries(${_cvfem_tgt}
        PRIVATE
            SFEM::sfem
            $<$<BOOL:${OpenMP_CXX_FOUND}>:OpenMP::OpenMP_CXX>)
    target_compile_options(${_cvfem_tgt} PRIVATE ${_cvfem_perf_opts})
    target_include_directories(${_cvfem_tgt} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    cvfem_subpar_target(${_cvfem_tgt})
endforeach()

# How much of the matrix the velocity-dependent half ever touches.
add_executable(nlcover nlcover.cpp)
target_link_libraries(nlcover PRIVATE SFEM::sfem
    $<$<BOOL:${OpenMP_CXX_FOUND}>:OpenMP::OpenMP_CXX>)
target_compile_options(nlcover PRIVATE ${_cvfem_perf_opts})
target_include_directories(nlcover PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
cvfem_subpar_target(nlcover)

# Guards the guards. Built with exactly the solver's flags, so it fails if the
# build's floating-point options have folded cvfem_hex8_ns_steady.cpp's
# std::isfinite() checks into a constant `true`. See the -fno-finite-math-only
# comment above. Run it in CI; it takes microseconds.
add_executable(cvfem_guard_selftest cvfem_guard_selftest.cpp)
target_compile_options(cvfem_guard_selftest PRIVATE ${_cvfem_perf_opts})
target_include_directories(cvfem_guard_selftest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
add_test(NAME cvfem_guard_selftest COMMAND cvfem_guard_selftest)

add_executable(cvfem_hex8_ns_steady cvfem_hex8_ns_steady.cpp)
target_link_libraries(cvfem_hex8_ns_steady
    PRIVATE
        SFEM::sfem
        $<$<BOOL:${OpenMP_CXX_FOUND}>:OpenMP::OpenMP_CXX>)
target_compile_options(cvfem_hex8_ns_steady PRIVATE ${_cvfem_perf_opts})
target_include_directories(cvfem_hex8_ns_steady PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../algebra
    ${CMAKE_CURRENT_SOURCE_DIR}/../../algebra/openmp)
cvfem_subpar_target(cvfem_hex8_ns_steady)

# set_target_properties(cvfem_tet4_ns_upwind_bench PROPERTIES
#     RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")

# ---------------------------------------------------------------------- CUDA
# Off by default: the CPU targets above must keep building on a machine with no
# CUDA toolkit, which is where most of this spike is developed.
# Variants that were measured, lost, and were moved to subpar/ rather than deleted.
# OFF by default; ON keeps them compiling so the removal decisions stay reproducible on
# other hardware. subpar/README.md records what each one measured and where it lost.

option(CVFEM_ENABLE_CUDA "Build the CVFEM HEX8 CUDA targets" OFF)

if(CVFEM_ENABLE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(NOT CMAKE_CUDA_COMPILER)
        message(FATAL_ERROR "CVFEM_ENABLE_CUDA=ON but no CUDA compiler was found. "
                            "On Alps, build inside the prgenv-gnu uenv (see alps_env.sh).")
    endif()

    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        # 90 = Hopper / GH200, matching cmake/SFEMDependencies.cmake in SFEM proper.
        set(CMAKE_CUDA_ARCHITECTURES 90 CACHE STRING "CUDA architectures")
    endif()
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)

    message(STATUS "cvfem CUDA: arch ${CMAKE_CUDA_ARCHITECTURES}, "
                   "nvcc ${CMAKE_CUDA_COMPILER_VERSION}")

    # Host-side flags carry -march/-mcpu=native and -ffast-math, which nvcc rejects
    # outright, so they must be scoped to CXX only. -Xptxas -v -warn-spills is kept
    # because register spills are the main risk in the generated Jacobian kernels and
    # this is how they get reported.
    set(_cvfem_cuda_opts
        -O3
        -use_fast_math
        --expt-relaxed-constexpr
        -Xptxas=-O3,-v,-warn-spills)

    function(cvfem_cuda_target tgt)
        target_compile_definitions(${tgt} PRIVATE CVFEM_ENABLE_CUDA)
        target_include_directories(${tgt} PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/cuda)
        cvfem_subpar_target(${tgt})
        target_compile_options(${tgt} PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:${_cvfem_perf_opts}>
            $<$<COMPILE_LANGUAGE:CUDA>:${_cvfem_cuda_opts}>)
        target_link_libraries(${tgt} PRIVATE
            SFEM::sfem
            CUDA::cudart
            CUDA::cusparse
            $<$<BOOL:${OpenMP_CXX_FOUND}>:OpenMP::OpenMP_CXX>)
        set_target_properties(${tgt} PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
    endfunction()

    # Smoke target: proves the toolchain, the device, and the C-ABI/stream convention
    # before any real kernel depends on them.
    add_executable(cvfem_cuda_smoke cuda/cvfem_cuda_smoke.cpp cuda/cvfem_cuda_smoke.cu)
    cvfem_cuda_target(cvfem_cuda_smoke)

    # Packed residual: verification against the CPU reference, plus timing. Kept as its
    # own driver rather than folded into cvfem_hex8_ns_upwind_bench, so the CPU
    # benchmark's build -- and therefore its measured baseline -- is left alone.
    add_executable(cvfem_hex8_ns_cuda_verify
        cuda/cvfem_hex8_ns_cuda_verify.cpp
        cuda/cvfem_hex8_ns_cuda.cu)
    cvfem_cuda_target(cvfem_hex8_ns_cuda_verify)
endif()
