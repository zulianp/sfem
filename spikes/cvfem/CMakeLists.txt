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

# Wanted for MPIEXEC_EXECUTABLE alone, so the multi-rank refusal can be gated on a launcher
# actually existing. MPI itself arrives transitively through matrixio's interface link
# libraries and is not requested here; QUIET because a machine with no launcher skips that one
# test rather than failing to configure.
find_package(MPI QUIET COMPONENTS CXX)

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
set(CVFEM_MODULES src/core src/hex8 src/tet4 src/generated src/ss src/op src/cases src/diag)

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

# The multigrid driver's dense coarse solve factorises with LAPACK's getrf. A required
# dependency rather than an option: the hand-written unblocked LU it replaces cost ~1 s per
# Newton step on a 2,320-dof coarsest level, and a second, portable path would be the same
# operation implemented twice. OpenBLAS on Alps, Accelerate on macOS.
find_package(LAPACK REQUIRED)
target_link_libraries(cvfem_hex8_ns_ssgmg PRIVATE LAPACK::LAPACK)

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

set(CVFEM_OP_TESTS cvfem_ns_op_gate cvfem_ns_op_isolation cvfem_bc_sideset_test cvfem_flat_vs_ss_test cvfem_flow_diag_test cvfem_ss_packed_probe cvfem_operator_consistency_test)
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
    # The kernel report's job is to keep a throughput next to the operator it was measured
    # on. A wrong rate is caught by anyone who knows the kernel; the right rate under the
    # wrong operator is not, and that is the failure the report exists to prevent -- so its
    # coverage classifier and its CSV reader are tested rather than trusted.
    add_test(NAME cvfem_kernel_report_selftest
             COMMAND ${CVFEM_REPORT_PYTHON}
                     ${CMAKE_CURRENT_SOURCE_DIR}/python/cvfem_kernel_report.py --selftest)
endif()

# ------------------------------------------------------- what the benchmark stages, and what
# it refuses. The staging work only means something if the combinations it opened are checked
# and the ones it kept shut stay shut, and both halves live in the driver's own oracles rather
# than in a reimplementation -- so this drives the driver. n=8, a couple of seconds.
add_test(NAME cvfem_bench_staging
         COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/tests/cvfem_bench_staging.sh
                 $<TARGET_FILE:cvfem_hex8_ns_upwind_bench>)

# --------------------------------------------- the packed macro path is the same operator
#
# cvfem_flat_vs_ss_test compares the flat and semi-structured operators by COORDINATE, which
# is what a renumbering demands, and it already carries the tolerance the two discretisations
# agree to. Running it again with the macro-element mesh packed asks the one question packing
# raises -- whether the faster reconstruction computes the same thing -- without introducing a
# second place for that tolerance to drift.
add_test(NAME cvfem_flat_vs_ss_packed COMMAND cvfem_flat_vs_ss_test)

# The consistency test twice, because SFEM_RC_EXACT_JAC is a process-wide static and the two
# settings have different invariants: the frozen form must have every operator form agreeing to
# round-off, the exact form deliberately does not and is bounded instead. The glob above already
# registers the default (exact) run.
add_test(NAME cvfem_operator_consistency_frozen COMMAND cvfem_operator_consistency_test)
set_tests_properties(cvfem_operator_consistency_frozen
                     PROPERTIES ENVIRONMENT "SFEM_RC_EXACT_JAC=0")

# And once more with the ATOMIC layouts forced, which is how packed and standard are compared.
# Not against each other -- against the operator. If the block diagonal reproduces the probed
# action on both layouts then they agree with the action and therefore with one another, which
# is a stronger statement than a direct diff and needs no coordinate-keyed matching across the
# renumbering a PackedMesh performs.
add_test(NAME cvfem_operator_consistency_atomic COMMAND cvfem_operator_consistency_test)
set_tests_properties(cvfem_operator_consistency_atomic
                     PROPERTIES ENVIRONMENT "SFEM_RC_EXACT_JAC=0;SFEM_DIAG_ATOMIC=1;SFEM_QGRAD_ATOMIC=1")
set_tests_properties(cvfem_flat_vs_ss_packed PROPERTIES ENVIRONMENT "SFEM_SS_PACK_SIZE=8")

# ------------------------------------------------------------------- restart round trip
#
# A run cut in two must be the run that was not cut. The pump is used because it is the
# cheapest case with an exact identity AND a time-varying boundary condition, so a segment
# that mistook its own step index for the absolute one drives the diaphragm at the wrong
# phase and the identity reports it.
add_test(NAME cvfem_restart
         COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/tests/cvfem_restart_test.sh
                 $<TARGET_FILE:cvfem_hex8_ns_ssgmg>)

# A step that will not converge is either the end of the run or a smaller step, never a step
# that is quietly accepted. Both policies are driven against one forced failure, because the
# thing worth testing is that they disagree in exactly the intended way.
add_test(NAME cvfem_cutback
         COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/tests/cvfem_cutback_test.sh
                 $<TARGET_FILE:cvfem_hex8_ns_ssgmg>)

# The transient multigrid/Vanka path, which nothing else exercised: cvfem_restart and
# cvfem_cutback both pin SFEM_GMG=0 SFEM_PRECOND=direct, so every transient number this spike
# gated came out of the dense coarse solve. A preconditioner decides how a system is solved and
# never what it solves, so the V-cycle must reach the same answer as the direct arm.
#
# N=4 at refine level 2 is the cheapest configuration that is both semi-structured (SFEM_GMG needs
# a refine level > 1) and carries the pump's exact swept-volume identity. The direct arm costs
# 36 s there against 1 s for multigrid; at N=8 the same arm costs 689 s, the dense coarse
# factorisation being cubic in a problem that grew 8x under refinement.
add_test(NAME cvfem_mg_transient
         COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/tests/cvfem_mg_transient_test.sh
                 $<TARGET_FILE:cvfem_hex8_ns_ssgmg>)
set_tests_properties(cvfem_mg_transient PROPERTIES TIMEOUT 300)

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

# The temporal order of the scheme, end to end through the driver. cvfem_ss_transient_test proves
# the BDF term is first- and second-order in isolation; it cannot see the Newton loop, FGMRES, the
# preconditioner, the time-varying boundary condition or the pressure gauge, any of which can cap
# the order while the term stays perfect.
#
# Guarded on numpy for the same reason the generator checks are guarded on sympy: the fit compares
# the solution FIELDS the restart writer drops, and a test that silently passed because numpy was
# missing would be worse than no test.
#
# Labelled slow -- it is three solves per scheme, measured at 121.6 s under ctest, and together
# with cvfem_mg_transient it adds about 152 s to a ~570 s suite. `ctest -LE slow` excludes it.
if(CVFEM_PYTHON)
    execute_process(COMMAND ${CVFEM_PYTHON} -c "import numpy"
                    RESULT_VARIABLE _cvfem_no_numpy OUTPUT_QUIET ERROR_QUIET)
    if(_cvfem_no_numpy EQUAL 0)
        add_test(NAME cvfem_richardson
                 COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/tests/cvfem_richardson_test.sh
                         $<TARGET_FILE:cvfem_hex8_ns_ssgmg> ${CVFEM_PYTHON})
        set_tests_properties(cvfem_richardson PROPERTIES TIMEOUT 900 LABELS slow)
        message(STATUS "cvfem: temporal-order gate enabled (${CVFEM_PYTHON})")
    else()
        message(STATUS "cvfem: temporal-order gate skipped -- ${CVFEM_PYTHON} has no numpy")
    endif()
endif()

# ----------------------------------------------------------------- the multi-rank refusal
#
# This spike is a one-socket code -- every measurement in the workstream is 72 threads at
# --ntasks=1 -- but nothing stopped it launching under mpirun, and at -n 2 it partitioned the
# mesh for real before dying inside sfem_Op.hpp. That abort was luck. The quantities that are
# wrong without an exchange are reductions (FGMRES dot and norm, the pressure gauge, the Newton
# and Armijo tests, the CFL maximum), and a rank evaluates each over its own slice without
# noticing it is partial -- so the failure a missing guard leaves open is a run that converges
# and reports a per-rank answer, not a crash.
#
# Registered only when a launcher exists, and the script itself reports SKIP (77) when the
# launcher is present but cannot place two ranks here, which is a property of the machine
# rather than of the code. The benchmark target is passed too: the bench drivers refuse through
# smesh's own initialize_serial rather than a second copy of the guard, and this checks that
# wiring is live.
if(MPIEXEC_EXECUTABLE)
    add_test(NAME cvfem_mpi_guard
             COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/tests/cvfem_mpi_guard_test.sh
                     $<TARGET_FILE:cvfem_hex8_ns_ssgmg>
                     ${MPIEXEC_EXECUTABLE}
                     $<TARGET_FILE:cvfem_sshex8_bench>)
    set_tests_properties(cvfem_mpi_guard PROPERTIES TIMEOUT 300 SKIP_RETURN_CODE 77)

    # That the distributed SETUP describes the same problem as the serial one: same global
    # node, element and constrained-dof counts, and the decomposition's own record of those
    # agreeing with what the owned ranges sum to.
    #
    # Everything before the solve is meant to be correct multi-rank already, and nothing
    # checked it. The failure it catches is quiet: mesh->n_nodes() is the LOCAL count on a
    # distributed mesh, so anything summed over it counts shared, ghost and aura entries once
    # per rank holding them and grows with the rank count while the problem does not.
    #
    # Integers only. The coordinate checksum is deliberately excluded -- floating-point
    # addition does not associate, so a partitioned sum need not match a sequential one bit for
    # bit even when both are right.
    add_test(NAME cvfem_distributed_setup
             COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/tests/cvfem_distributed_setup_test.sh
                     $<TARGET_FILE:cvfem_hex8_ns_ssgmg>
                     ${MPIEXEC_EXECUTABLE})
    set_tests_properties(cvfem_distributed_setup PROPERTIES TIMEOUT 300 SKIP_RETURN_CODE 77)

    message(STATUS "cvfem: MPI guard gate enabled (${MPIEXEC_EXECUTABLE})")
else()
    message(STATUS "cvfem: MPI guard gate skipped -- no MPI launcher found")
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
