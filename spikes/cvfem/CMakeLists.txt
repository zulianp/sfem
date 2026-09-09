cmake_minimum_required(VERSION 3.20)

# A standalone project, deliberately: the SFEM root build never adds this directory, and
# the spike is developed and benchmarked against an *installed* SFEM. The layout below
# nonetheless follows SFEM's own conventions -- named module directories globbed into one
# library, each placed flat on the include path, drivers and tests discovered by globbing
# their directory -- so that moving between the two costs nothing.
project(cvfem LANGUAGES C CXX)

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

# ---------------------------------------------------------------------- options
# Declared together rather than scattered through the file. CUDA is off by default
# because the CPU targets must keep building on a machine with no CUDA toolkit, which
# is where most of this spike is developed.
option(CVFEM_ENABLE_CUDA "Build the CVFEM HEX8 CUDA targets" OFF)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
include(CVFEMCMakeFunctions)

# ---------------------------------------------------------------------- modules
# Every cvfem header is included by bare filename, never by path, so each module
# directory goes on the include path. This mirrors the SFEM_MODULES / scan_directories
# loop in SFEM's own top-level CMakeLists.txt, and reuses its function where the SFEM
# source tree is reachable -- the spike already compiles against ../../algebra.
set(CVFEM_MODULES src/core src/hex8 src/tet4 src/generated src/ss src/op src/cases)

set(_sfem_cmake_functions ${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/SFEMCMakeFunctions.cmake)
if(EXISTS ${_sfem_cmake_functions})
    include(${_sfem_cmake_functions})
    foreach(MODULE ${CVFEM_MODULES})
        scan_directories(${CMAKE_CURRENT_SOURCE_DIR}/${MODULE} "."
                         CVFEM_INCLUDE_DIRS CVFEM_HEADERS CVFEM_SOURCES)
    endforeach()
else()
    # Detached copy of the spike: same result without the SFEM source tree.
    foreach(MODULE ${CVFEM_MODULES})
        file(GLOB _mod_sources CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/${MODULE}/*.cpp")
        file(GLOB _mod_headers CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/${MODULE}/*.hpp")
        list(APPEND CVFEM_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/${MODULE})
        list(APPEND CVFEM_SOURCES ${_mod_sources})
        list(APPEND CVFEM_HEADERS ${_mod_headers})
    endforeach()
endif()

# The SFEM source tree, for the two headers that are reached in place rather than through
# the install. Kept explicit so it is obvious which targets pay for the coupling.
set(CVFEM_SFEM_ALGEBRA_INCLUDES
    ${CMAKE_CURRENT_SOURCE_DIR}/../../algebra
    ${CMAKE_CURRENT_SOURCE_DIR}/../../algebra/openmp)
set(CVFEM_SFEM_OPERATORS_INCLUDES ${CMAKE_CURRENT_SOURCE_DIR}/../../operators)

# ---------------------------------------------------------------------- library
# The sfem::Op face of the kernels, for the semi-structured GMG work. Kept as a library so
# the Op compiles and is gated on its own before any driver depends on it.
add_library(cvfem_ns_op STATIC ${CVFEM_SOURCES} ${CVFEM_HEADERS})
target_link_libraries(cvfem_ns_op
    PUBLIC
        SFEM::sfem
        $<$<BOOL:${OpenMP_CXX_FOUND}>:OpenMP::OpenMP_CXX>)
target_include_directories(cvfem_ns_op PUBLIC
    ${CVFEM_INCLUDE_DIRS}
    ${CVFEM_SFEM_ALGEBRA_INCLUDES})
cvfem_target_defaults(cvfem_ns_op)

# ---------------------------------------------------------------------- drivers
# One executable per file in drivers/, named after the file, as SFEM does for drivers/.
# Most link SFEM directly; the ones below go through the Op library.
set(CVFEM_OP_DRIVERS cvfem_hex8_ns_ssgmg cvfem_ns_apply_bench cvfem_sshex8_bench)

file(GLOB CVFEM_DRIVERS CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/drivers/*.cpp")
foreach(DRIVER ${CVFEM_DRIVERS})
    get_filename_component(DRIVER_NAME ${DRIVER} NAME_WE)
    add_executable(${DRIVER_NAME} ${DRIVER})
    if(DRIVER_NAME IN_LIST CVFEM_OP_DRIVERS)
        target_link_libraries(${DRIVER_NAME} PRIVATE cvfem_ns_op)
    else()
        target_link_libraries(${DRIVER_NAME}
            PRIVATE
                SFEM::sfem
                $<$<BOOL:${OpenMP_CXX_FOUND}>:OpenMP::OpenMP_CXX>)
    endif()
    cvfem_target_defaults(${DRIVER_NAME})
endforeach()

# The steady solver reaches into SFEM's algebra headers in place; the Op-linked drivers
# get the same directories through cvfem_ns_op's PUBLIC includes.
target_include_directories(cvfem_hex8_ns_steady PRIVATE ${CVFEM_SFEM_ALGEBRA_INCLUDES})

# T3: the macro-local gather on a semi-structured mesh, against the flat gather on the same
# mesh. The only target built against BLAS -- the gemm path it measures is the only place
# it would change a number.
target_include_directories(cvfem_sshex8_bench PRIVATE ${CVFEM_SFEM_OPERATORS_INCLUDES})
cvfem_blas_target(cvfem_sshex8_bench)
add_test(NAME cvfem_sshex8_agree COMMAND cvfem_sshex8_bench)
# SFEM_BENCH_PROBE_DIAG=1 checks the block diagonal against the operator itself rather
# than against the other block-diagonal implementation. It costs one apply per dof, so
# it is off by default and had therefore never run anywhere; at these two sizes it adds
# about 2.4 s, which is worth paying for the only check that the assembled diagonal is
# a diagonal of anything.
set_tests_properties(cvfem_sshex8_agree PROPERTIES ENVIRONMENT "SFEM_BENCH_MACROS=2;SFEM_BENCH_LEVELS=2,4;SFEM_BENCH_REPS=1;SFEM_BENCH_WARMUP=0;SFEM_BENCH_PROBE_DIAG=1")

# ------------------------------------------------------------------------ tests
# One test per file in tests/, filename == target name == ctest name, as SFEM does for
# <module>/tests/. What each one gates:
#   cvfem_ns_op_gate       the Op reproduces the standalone driver on the same mesh and
#                          state -- the gate everything downstream of the Op rests on.
#   cvfem_ns_op_isolation  compile-time proof that the operator header is usable from any
#                          driver: it leaks none of the solver core and coexists with the
#                          benchmark's layout family. Failure mode is a build error.
#   tests_lshape_check     the L-shape generator's element and node counts against their
#                          closed forms.
file(GLOB CVFEM_TESTS CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cpp")
list(REMOVE_ITEM CVFEM_TESTS ${CMAKE_CURRENT_SOURCE_DIR}/tests/cvfem_guard_selftest.cpp)

set(CVFEM_OP_TESTS cvfem_ns_op_gate cvfem_ns_op_isolation cvfem_bc_sideset_test)
foreach(TEST_SOURCE ${CVFEM_TESTS})
    get_filename_component(TEST_NAME ${TEST_SOURCE} NAME_WE)
    add_executable(${TEST_NAME} ${TEST_SOURCE})
    if(TEST_NAME IN_LIST CVFEM_OP_TESTS)
        target_link_libraries(${TEST_NAME} PRIVATE cvfem_ns_op)
    else()
        target_link_libraries(${TEST_NAME}
            PRIVATE
                SFEM::sfem
                $<$<BOOL:${OpenMP_CXX_FOUND}>:OpenMP::OpenMP_CXX>)
    endif()
    cvfem_target_defaults(${TEST_NAME})
    add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
endforeach()

# ------------------------------------------------------------ verification report machinery
#
# The report generator draws the conclusions -- it fits the convergence rates, decides which
# checks pass, and is the thing a reader will believe. A generator that quietly fits the
# wrong slope is worse than no report, so its fitter, its log parser and its plotter are
# tested like any other code here.
#
# Standard-library only and no build dependency, so this runs wherever a python3 does,
# including inside the Alps uenv where numpy, matplotlib and markdown are all absent.
find_program(CVFEM_REPORT_PYTHON NAMES python3 python)
if(CVFEM_REPORT_PYTHON)
    add_test(NAME cvfem_verify_report_selftest
             COMMAND ${CVFEM_REPORT_PYTHON}
                     ${CMAKE_CURRENT_SOURCE_DIR}/python/cvfem_verify_report.py --selftest)
endif()

# ------------------------------------------------ generated headers match their generators
#
# The three headers under src/generated/ and src/cases/ are committed artifacts compiled as
# ordinary source, and nothing regenerated or diffed them, so they could drift from the
# scripts that emit them unnoticed. Each generator now has a --check mode that regenerates
# in memory and compares, writing nothing.
#
# Registered only when a Python with SymPy is actually available: the check needs the same
# library the headers were emitted with, and a test that silently passes because the
# interpreter was missing would be worse than no test. Regeneration is slow (about a minute
# for the HEX8 kernels), so the two large synthesizers carry a longer timeout and a label,
# and only the cheap MMS case runs by default.
find_program(CVFEM_PYTHON NAMES python3 python
             HINTS ${CMAKE_CURRENT_SOURCE_DIR}/../../venv/bin ${CMAKE_CURRENT_SOURCE_DIR}/../../.venv/bin
             NO_CMAKE_SYSTEM_PATH)
if(NOT CVFEM_PYTHON)
    find_program(CVFEM_PYTHON NAMES python3 python)
endif()

if(CVFEM_PYTHON)
    execute_process(COMMAND ${CVFEM_PYTHON} -c "import sympy"
                    RESULT_VARIABLE _cvfem_no_sympy OUTPUT_QUIET ERROR_QUIET)
    if(_cvfem_no_sympy EQUAL 0)
        add_test(NAME cvfem_mms_case_is_current
                 COMMAND ${CVFEM_PYTHON} ${CMAKE_CURRENT_SOURCE_DIR}/python/gen_mms_case.py --check)
        foreach(_elem hex8 tet4)
            add_test(NAME cvfem_sympy_kernels_${_elem}_are_current
                     COMMAND ${CVFEM_PYTHON}
                             ${CMAKE_CURRENT_SOURCE_DIR}/python/synthesize_cvfem_${_elem}_ns_upwind_sympy.py
                             --check)
            set_tests_properties(cvfem_sympy_kernels_${_elem}_are_current
                                 PROPERTIES TIMEOUT 600 LABELS codegen)
        endforeach()
        message(STATUS "cvfem: generator checks enabled (${CVFEM_PYTHON})")
    else()
        message(STATUS "cvfem: generator checks skipped -- ${CVFEM_PYTHON} has no sympy")
    endif()
else()
    message(STATUS "cvfem: generator checks skipped -- no python found")
endif()

# Guards the guards, and so is the one target that must not go through
# cvfem_target_defaults: it links nothing and defines nothing, because what it tests is
# the build's floating-point options in isolation. It fails if they have folded
# drivers/cvfem_hex8_ns_steady.cpp's std::isfinite() checks into a constant `true`. See
# the -fno-finite-math-only comment in cmake/CVFEMCMakeFunctions.cmake. Run it in CI; it
# takes microseconds.
add_executable(cvfem_guard_selftest tests/cvfem_guard_selftest.cpp)
target_compile_options(cvfem_guard_selftest PRIVATE ${_cvfem_perf_opts})
target_include_directories(cvfem_guard_selftest PRIVATE ${CVFEM_INCLUDE_DIRS})
add_test(NAME cvfem_guard_selftest COMMAND cvfem_guard_selftest)

# ------------------------------------------------------------------------- CUDA
if(CVFEM_ENABLE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(NOT CMAKE_CUDA_COMPILER)
        message(FATAL_ERROR "CVFEM_ENABLE_CUDA=ON but no CUDA compiler was found. "
                            "On Alps, build inside the prgenv-gnu uenv (see scripts/alps_env.sh).")
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
            ${CVFEM_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/cuda)
        cvfem_subpar_target(${tgt})
        cvfem_trace_target(${tgt})
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
