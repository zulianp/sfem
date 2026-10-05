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

# THE PHASE BREAKDOWN, AND WHERE IT IS NOT WANTED.
#
# The packed sweeps carry probes that report where their time goes -- local memset, gather,
# kernel, scatter, ghost reduction -- which is what the benchmark's --breakdown prints. They are
# macros (see src/kernels/cvfem_phases.hpp) so that they can be compiled out rather than merely
# switched off at runtime: with CVFEM_PHASES=0 there is no clock read, no accumulator, no branch
# and no parameter carrying the accumulator through the signature. Measured on a probe kernel,
# the disabled build holds no reference to wall_time or g_breakdown at all.
#
# ON by default, because the benchmark is where this spike measures things and --breakdown is one
# of the ways it does. The operator library below sets it to 0 regardless: a breakdown is a
# benchmark feature, and the library SFEM links has no caller for it.
option(CVFEM_ENABLE_PHASES "Compile the packed sweeps' phase-breakdown probes" ON)
if(NOT CVFEM_ENABLE_PHASES)
    add_compile_definitions(CVFEM_PHASES=0)
endif()

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
include(CVFEMCMakeFunctions)

# ----------------------------------------------------------------------- the source tree
# ONE include root, and every cvfem header is included BY PATH relative to it:
#
#     #include "kernels/packed/affine/residual.hpp"
#
# not by bare filename. That is the whole of what lets src/ have a shape. The previous
# arrangement put each module directory flat onto the include path, which bought bare
# filenames at two prices: a directory not named in a hand-maintained list was invisible to
# the build, and two headers sharing a basename were one header, with the winner decided by
# directory order. Both failures were silent, which is why there were two FATAL_ERROR guards
# here to catch them. Neither is needed once the path is part of the include.
#
# Recursive, so a new directory is picked up by existing. The guard below is about shape,
# not about membership.
file(GLOB_RECURSE CVFEM_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
# src/drivers and src/tests are under src/ but are not library sources: each of those files
# carries a main(). They are globbed separately below.
list(FILTER CVFEM_SOURCES EXCLUDE REGEX "/src/(drivers|tests)/")
file(GLOB_RECURSE CVFEM_HEADERS CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/*.hpp")
set(CVFEM_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/src)

# WHAT MAY COMPILE INTO THE OP LIBRARY. Every .cpp the glob finds lands in
# add_library(cvfem_ns_op) below, so a translation unit under src/kernels/ would drag a whole
# kernel family into the one target that must not see it -- and the two operator families
# define sixteen of the same names with different physics. src/kernels/ is header-only, and
# this is what says so loudly rather than at link time.
foreach(_src ${CVFEM_SOURCES})
    file(RELATIVE_PATH _rel ${CMAKE_CURRENT_SOURCE_DIR} ${_src})
    if(_rel MATCHES "^src/kernels/")
        message(FATAL_ERROR
            "${_rel} is a translation unit under src/kernels/, which is header-only. Every "
            ".cpp under src/ is compiled into cvfem_ns_op; a kernel .cpp would put a kernel "
            "family into the operator library. Make it a header, or put it in src/frontend/.")
    endif()
endforeach()

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
# The breakdown probes are a benchmark feature; this library has no caller for one, so they are
# compiled out of it whatever CVFEM_ENABLE_PHASES says. The kernels it does instantiate carry no
# probes today, so this states the intent rather than changing the code -- which is the point of
# saying it on the target rather than relying on that staying true.
target_compile_definitions(cvfem_ns_op PRIVATE CVFEM_PHASES=0)

target_include_directories(cvfem_ns_op PUBLIC
    ${CVFEM_INCLUDE_DIRS}
    ${CVFEM_SFEM_ALGEBRA_INCLUDES})
cvfem_target_defaults(cvfem_ns_op)

# ---------------------------------------------------------------------- drivers
# One executable per file in drivers/, named after the file, as SFEM does for drivers/.
# Most link SFEM directly; the ones below go through the Op library.
set(CVFEM_OP_DRIVERS cvfem_hex8_ns_ssgmg cvfem_ns_apply_bench cvfem_sshex8_bench)

file(GLOB_RECURSE CVFEM_DRIVERS CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/drivers/*.cpp")
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

# THE SAME AGREEMENT ON A MESH WITH CURVED MACRO ELEMENTS, and what it does NOT gate.
#
# The box above has no curved macro element, so this arm warps its inflow half and one mesh then
# carries curved and straight macro elements side by side. What that buys is coverage of the
# curved branch and of which macro element each sweep claims: these four sweeps are split along
# the affine / isoparametric line separately, so an element dropped from one sweep's range shows
# up here as a disagreement with the other three.
#
# What it does NOT gate is whether the geometry is derived correctly, and the distinction was
# measured rather than reasoned: forcing sscvfem_macro_curved to false -- which hoists one
# geometry over a curved macro element's micro cells, the exact bug this split could introduce
# -- leaves BOTH arms of this test passing. All four sweeps read the same geometry source, so
# they agree with each other whether it is right or wrong. The oracle for that is
# cvfem_flat_vs_ss_test's warped arms, where the same break fails four checks and nothing else.
add_test(NAME cvfem_sshex8_agree_warped COMMAND cvfem_sshex8_bench)
set_tests_properties(cvfem_sshex8_agree_warped PROPERTIES ENVIRONMENT "SFEM_BENCH_MACROS=2;SFEM_BENCH_LEVELS=2,4;SFEM_BENCH_REPS=1;SFEM_BENCH_WARMUP=0;SFEM_BENCH_PROBE_DIAG=1;SFEM_BENCH_WARP=0.15")

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
# Recursive: the tests are grouped by TYPE in subfolders now. Target and ctest names stay
# NAME_WE, so the suite's names are unchanged by the move and a run before and after is
# comparable entry by entry.
file(GLOB_RECURSE CVFEM_TESTS CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/tests/*.cpp")
list(REMOVE_ITEM CVFEM_TESTS ${CMAKE_CURRENT_SOURCE_DIR}/src/tests/compile/cvfem_guard_selftest.cpp)
list(REMOVE_ITEM CVFEM_TESTS ${CMAKE_CURRENT_SOURCE_DIR}/src/tests/compile/cvfem_vectorization_gate.cpp)
# The two device tests are built by the CUDA block below, with nvcc and the CUDA libraries, and
# only when CVFEM_ENABLE_CUDA is on. Left in this glob they would be configured a second time as
# plain CPU targets -- a duplicate target name where CUDA is on, and a link failure where it is
# off. They live under src/tests/verification/ because that is what they are: a device-against-host
# comparison. Where they are built from is not where they are grouped.
list(REMOVE_ITEM CVFEM_TESTS ${CMAKE_CURRENT_SOURCE_DIR}/src/tests/verification/cvfem_cuda_smoke.cpp)
list(REMOVE_ITEM CVFEM_TESTS ${CMAKE_CURRENT_SOURCE_DIR}/src/tests/verification/cvfem_hex8_ns_cuda_verify.cpp)

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

# cvfem_sympy_action_test gates the generated Jacobian-action arrangements, which are
# quarantined in subpar/ by default, so it exits 77 (ctest's SKIP) unless CVFEM_ENABLE_SUBPAR
# is on. Declared here rather than beside the option because the glob loop above is what
# creates the test. Without this the skip would be recorded as a FAILURE, which is the worst
# of both: a red suite that means nothing, training people to ignore it.
set_tests_properties(cvfem_sympy_action_test PROPERTIES SKIP_RETURN_CODE 77)

# cvfem_subpar_compiles is the quarantine's own gate: it includes every subpar/ header so that
# a retired kernel cannot stop compiling unnoticed. Nothing else includes them -- that is the
# whole reason they rot -- and two of them had, each dating from the commit that made the
# change it missed. Same SKIP, for the same reason: without the option the headers are not on
# the include path.
set_tests_properties(cvfem_subpar_compiles PROPERTIES SKIP_RETURN_CODE 77)
set_tests_properties(cvfem_subpar_ss_compiles PROPERTIES SKIP_RETURN_CODE 77)

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
# ------------------------------------------------- src/kernels/ carries no dependency
#
# The property this checks took a long cascade to establish and one convenient #include to
# undo, which is exactly the shape of thing that needs a test rather than a note. It also
# replaces the old depth-1 directory guard, which only ever looked one level down and would
# have passed a stray src/kernels/packed/affine/ without seeing it.
if(CVFEM_REPORT_PYTHON)
    add_test(NAME cvfem_kernels_self_contained
             COMMAND ${CVFEM_REPORT_PYTHON}
                     ${CMAKE_CURRENT_SOURCE_DIR}/src/tests/compile/cvfem_kernels_self_contained.py)
    # ------------------------------------------------- an absent input is a null pointer
    #
    # The sweeps signal "this input is not there" with a null pointer: no boundary mask, no
    # prescribed pressure, no curved macro element. A caller that spells such an argument
    # `d.mask.data()` breaks that signal, because std::vector::data() is only required to
    # return null for a vector that never allocated, and the Op clears four of these masks on
    # every setup -- so a cleared-but-allocated mask arrives looking present, carrying the
    # entries of the previous setup.
    #
    # This is not a style check. Fifty-three arguments were spelled that way when the
    # semi-structured sweeps shed their staging objects, and none of them was visible one
    # level deep: the sweep itself rarely tests the pointer, it hands it to the kernel that
    # does. The test propagates the null test through calls until it stops finding new ones,
    # which is what makes the fifty-three appear at all.
    add_test(NAME cvfem_null_tested_args
             COMMAND ${CVFEM_REPORT_PYTHON}
                     ${CMAKE_CURRENT_SOURCE_DIR}/src/tests/compile/cvfem_null_tested_args.py)

    # A SWEEP CALL LEFT OUTSIDE ITS PARALLEL REGION, which nothing else can see.
    #
    # `#pragma omp parallel` applies to the one statement after it. A split sweep pair is two
    # statements, and without braces the second runs after the region closes -- once, on the
    # encountering thread. The answer is still right, every element is processed once, and the
    # fingerprints are taken at one thread, so the whole suite passes while half of a pair runs
    # serially. On a box the curved half is empty and there is nothing to notice at all.
    #
    # Found the hard way: the naive residual pair went in unbraced and was caught by a brace pass
    # run for a different sweep, with six more in the benchmark.
    add_test(NAME cvfem_parallel_region_pairs
             COMMAND ${CVFEM_REPORT_PYTHON}
                     ${CMAKE_CURRENT_SOURCE_DIR}/src/tests/compile/cvfem_parallel_region_pairs.py)

    # EVERY KERNEL THAT NAMES A COMPUTATION TYPE DECLARES IT, which the mixed-precision gates
    # cannot say. Those three show that the templated kernels run at single precision and agree;
    # they cannot show that nothing was missed, because a definition still spelling the build's
    # scalar_t compiles perfectly and is only ever reached at the other precision if some f32
    # test happens to instantiate it -- and most of src/kernels/ is not reached by them.
    #
    # It found four boundary-closure kernels, two partial-assembly convective kernels, two
    # generated BSR-slot kernels and four generated TET4 kernels after all three families had
    # been converted and all three mixed-precision gates were passing.
    #
    # The shared constant tables are the one listed exception: an accessor returning a reference
    # to a `const double[...]` table promotes an f32 chain, and templating it means a copy per
    # precision -- a decision about storage, not a missing parameter.
    add_test(NAME cvfem_kernels_are_templated
             COMMAND ${CVFEM_REPORT_PYTHON}
                     ${CMAKE_CURRENT_SOURCE_DIR}/src/tests/compile/cvfem_kernels_are_templated.py)

    # Every option a job script hands a driver must still be one that driver parses. Removing
    # the micro-kernel selector took --kernel with it and broke twenty-six job scripts in
    # silence, thirteen of which feed the paper's generated macros: a driver rejects an unknown
    # option and exits, and the jobs scrape its stdout with sed, so the failure arrived as an
    # empty table cell rather than as a non-zero exit.
    add_test(NAME cvfem_jobs_use_live_flags
             COMMAND ${CVFEM_REPORT_PYTHON}
                     ${CMAKE_CURRENT_SOURCE_DIR}/src/tests/compile/cvfem_jobs_use_live_flags.py)

    # Per-thread scratch comes from the arena, never from a std::vector declared inside a
    # parallel region, which allocates and frees once per thread per call. The arena's slot
    # register is what makes each thread's working set visible; a vector hides it, which is how
    # an nnodes-wide allocation ended up in the Vanka smoother.
    add_test(NAME cvfem_no_vector_in_parallel
             COMMAND ${CVFEM_REPORT_PYTHON}
                     ${CMAKE_CURRENT_SOURCE_DIR}/src/tests/compile/cvfem_no_vector_in_parallel.py)
endif()

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
# The TET4 benchmark's own oracles, which nothing ran. It carries verify_packed_vs_atomic,
# verify_jac_packed_vs_atomic, the element-kernel finite-difference checks and the SpMV-against-
# matrix-free chain, and it exits non-zero on any of them -- but no test invoked it, so the only
# thing holding its kernels together was a person remembering to pass --verify by hand. That
# became load-bearing when thirteen micro-kernel arms were reduced to three and the packed
# assembly changed kernel; a reduction defended by an oracle nobody runs is not defended.
# The packed layout's determinism, which is the property that makes its throughput comparable
# with the atomic layout's at all. It is checked by running the same configuration twice and
# comparing the bitwise fingerprint -- a run against ITSELF, which a race cannot satisfy by
# luck. The fingerprint sweep compares against a recorded baseline instead, so a race that
# happened to reproduce the baseline would pass it; that is how a nested parallel region
# survived long enough to be found by accident. Shown to fire before being committed: with the
# stray pragma back, it failed exactly the three affected configurations and nothing else.
add_test(NAME cvfem_packed_determinism
         COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/src/tests/regression/cvfem_packed_determinism.sh
                 $<TARGET_FILE:cvfem_hex8_ns_upwind_bench>)
set_tests_properties(cvfem_packed_determinism PROPERTIES SKIP_RETURN_CODE 77)

add_test(NAME cvfem_tet4_bench_verify
         COMMAND cvfem_tet4_ns_upwind_bench --n 10 --repeat 1 --warmup 0 --verify --verify-jac)

add_test(NAME cvfem_bench_staging
         COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/src/tests/verification/cvfem_bench_staging.sh
                 $<TARGET_FILE:cvfem_hex8_ns_upwind_bench>)
# The script asserts that every quarantined configuration is REFUSED, which is true of the
# default build and false of this one -- the flag exists to make them reachable. So it is told
# which build it is in rather than inferring it.
if(CVFEM_ENABLE_SUBPAR)
    set_tests_properties(cvfem_bench_staging PROPERTIES ENVIRONMENT CVFEM_HAS_SUBPAR=1)
endif()

# ------------------------------------------------- the geometry gate, on a NON-AFFINE mesh
#
# Every other oracle here runs on a uniform cube, where the affine edge vector and the
# difference of two node coordinates are the same number to the last bit -- so a kernel using
# one and its reference using the other agree perfectly while discretising different geometries.
# This gate warps the mesh, which separates them. It was added after exactly that inconsistency
# reached a campaign undetected and surfaced at n=160 as a 2.0e-4 disagreement; the gate was
# then checked by reintroducing the bug, which it fails at warp > 0 and passes on the cube.
add_test(NAME cvfem_warped_geometry
         COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/src/tests/verification/cvfem_warped_geometry.sh
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
# The deferred correction's freezing path, and the default that selects it. Registered
# because nothing else in this suite sets SFEM_CONV_HO: without it the entire higher-order
# convection path, and the branch its default now takes, sit outside the local gate.
add_test(NAME cvfem_conv_freeze
         COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/src/tests/verification/cvfem_conv_freeze_test.sh
                 $<TARGET_FILE:cvfem_hex8_ns_ssgmg>)
# 600, not 300, because this test gained a fourth solver arm: the LAGGED Jacobian, which is the
# point of the assertion it feeds -- it must cost Newton steps, and it does, 35 against 13, which
# makes it the most expensive run in the file. Isolated the test takes 134 s; under the suite's own
# `ctest -j 6` it has been seen at 238 s and has timed out at 300. The limit was sized for the
# shorter test, not for a slow machine.
set_tests_properties(cvfem_conv_freeze PROPERTIES TIMEOUT 600)

# A run cut in two must be the run that was not cut. The pump is used because it is the
# cheapest case with an exact identity AND a time-varying boundary condition, so a segment
# that mistook its own step index for the absolute one drives the diaphragm at the wrong
# phase and the identity reports it.
add_test(NAME cvfem_restart
         COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/src/tests/regression/cvfem_restart_test.sh
                 $<TARGET_FILE:cvfem_hex8_ns_ssgmg>)

# A step that will not converge is either the end of the run or a smaller step, never a step
# that is quietly accepted. Both policies are driven against one forced failure, because the
# thing worth testing is that they disagree in exactly the intended way.
add_test(NAME cvfem_cutback
         COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/src/tests/verification/cvfem_cutback_test.sh
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
         COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/src/tests/verification/cvfem_mg_transient_test.sh
                 $<TARGET_FILE:cvfem_hex8_ns_ssgmg>)
set_tests_properties(cvfem_mg_transient PROPERTIES TIMEOUT 300)

# ------------------------------------------------ generated headers match their generators
#
# The three generated headers, under src/upwind/ and src/cases/, are committed artifacts compiled as
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
                 COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/src/tests/verification/cvfem_richardson_test.sh
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
             COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/src/tests/verification/cvfem_mpi_guard_test.sh
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
             COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/src/tests/verification/cvfem_distributed_setup_test.sh
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
add_executable(cvfem_guard_selftest src/tests/compile/cvfem_guard_selftest.cpp)
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
            ${CVFEM_INCLUDE_DIRS}
            # The device tests include their own headers by bare name.
            ${CMAKE_CURRENT_SOURCE_DIR}/src/tests/verification)
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
    add_executable(cvfem_cuda_smoke src/tests/verification/cvfem_cuda_smoke.cpp
                               src/tests/verification/cvfem_cuda_smoke.cu)
    cvfem_cuda_target(cvfem_cuda_smoke)

    # Packed residual: verification against the CPU reference, plus timing. Kept as its
    # own driver rather than folded into cvfem_hex8_ns_upwind_bench, so the CPU
    # benchmark's build -- and therefore its measured baseline -- is left alone.
    add_executable(cvfem_hex8_ns_cuda_verify
        src/tests/verification/cvfem_hex8_ns_cuda_verify.cpp
        src/frontend/cuda/cvfem_hex8_ns_cuda.cu)
    cvfem_cuda_target(cvfem_hex8_ns_cuda_verify)
endif()

# ------------------------------------------------------------------- the vectorisation gate
#
# DESIGN.md requires the build to fail when a lane loop does not vectorise. It has to be the
# BUILD rather than a test, because a kernel that quietly went scalar still produces the right
# answers -- every oracle in this suite stays green through it, which is exactly how the one
# this gate found survived. The measured cost of that one was the entire vectorisation of the
# higher-order residual: zero vector FP instructions against 937 scalar ones.
#
# WHY IT INSPECTS OBJECTS RATHER THAN READING COMPILER REMARKS is argued at length in
# src/tests/compile/cvfem_check_vectorization.py. The short of it: a lane loop runs
# CVFEM_HEX8_VEC_SIZE iterations, a compile-time constant, so it is unrolled and vectorised by
# the SLP pass -- the loop vectoriser never engages and -Rpass-missed=loop-vectorize is silent
# whether the result is vector or scalar. -Werror=pass-failed would be a gate that cannot fire.
#
# ONE OBJECT PER KERNEL, and that is not fastidiousness. With three wrappers in one translation
# unit the compiler tail-merged them into a single shared body reached by three entry stubs, and
# a scalar kernel would have hidden behind its neighbours' vector instructions.
set(CVFEM_VEC_GATE_KERNELS
    1  residual_first_order        # the kernel the standing performance gate measures
    2  residual_rhie_chow          # the same, with the coupling this discretisation always runs
    3  residual_higher_order       # the deferred correction, entered through the sweep
    10 lane_defcor_unlimited       # the lane kernel itself, one limiter per object, which is
    11 lane_defcor_clip            # the resolution that caught the limiter-stats atomic --
    12 lane_defcor_venkatakrishnan # entering through the sweep showed 17% and read as merely
    13 lane_defcor_darwish_moukalled  # poor, where the lane kernels showed a flat zero.
    20 jv_lane_first_order         # the Jacobian action's lane sweep, which the gate had no row
    21 jv_lane_defcor_unlimited    # for while clang warned about it 72 times a build
    22 jv_lane_defcor_clip
    23 jv_lane_defcor_venkatakrishnan
    24 jv_lane_defcor_darwish_moukalled)

# Kernels that emit no vector arithmetic, and the list is SCOPED TO THE COMPILER because the
# shortfall is. On gcc 13.3 on the Grace nodes -- the only machine whose throughput decides
# anything here -- all twelve kernels above come out 99-100% vector and this list is empty, so
# that build enforces every one of them. Apple clang gives up inlining the two largest Jacobian
# arms: the directional derivative of the deferred correction runs to roughly 19,000 instructions
# per lane body with a limiter on, the lambda inside cvfem_hex8_scs_defcor_jv survives as a call
# at kernels.hpp:1609 ("call instruction cannot be vectorized"), and the lane loop then goes
# scalar. gcc inlines the same body and vectorises it.
#
# Listing them unconditionally would have weakened the gate on the machine that matters to suit
# the laptop. An entry here is still compiled, measured and reported on every build -- never
# skipped -- and the check fails if a listed kernel starts vectorising, so the list cannot
# outlive the problem.
set(CVFEM_VEC_GATE_KNOWN_SCALAR "")
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    list(APPEND CVFEM_VEC_GATE_KNOWN_SCALAR
         jv_lane_defcor_venkatakrishnan
         jv_lane_defcor_darwish_moukalled)
endif()

find_program(CVFEM_VEC_GATE_PYTHON NAMES python3 python)
if(CVFEM_VEC_GATE_PYTHON)
    set(_vec_gate_objs "")
    list(LENGTH CVFEM_VEC_GATE_KERNELS _vec_gate_n)
    math(EXPR _vec_gate_last "${_vec_gate_n} / 2 - 1")
    foreach(_i RANGE ${_vec_gate_last})
        math(EXPR _ki "${_i} * 2")
        math(EXPR _ni "${_i} * 2 + 1")
        list(GET CVFEM_VEC_GATE_KERNELS ${_ki} _kid)
        list(GET CVFEM_VEC_GATE_KERNELS ${_ni} _kname)
        set(_tgt cvfem_vecgate_${_kname})
        add_library(${_tgt} OBJECT
                    ${CMAKE_CURRENT_SOURCE_DIR}/src/tests/compile/cvfem_vectorization_gate.cpp)
        # The real flags, through the same interface every kernel target uses. A gate compiled
        # with anything else would be measuring a build nobody ships.
        target_include_directories(${_tgt} PRIVATE ${CVFEM_INCLUDE_DIRS})
        target_compile_definitions(${_tgt} PRIVATE CVFEM_VEC_GATE_KERNEL=${_kid})
        target_compile_options(${_tgt} PRIVATE ${_cvfem_perf_opts})
        # OpenMP for the `#pragma omp simd` the lane loops carry. Without it the pragma is
        # ignored and the gate would be measuring a different loop from the one that ships.
        if(OpenMP_CXX_FOUND)
            target_link_libraries(${_tgt} PRIVATE OpenMP::OpenMP_CXX)
        endif()
        list(APPEND _vec_gate_objs "${_kname}=$<TARGET_OBJECTS:${_tgt}>")
        list(APPEND _vec_gate_tgts ${_tgt})
    endforeach()

    set(_vec_gate_exempt "")
    foreach(_x ${CVFEM_VEC_GATE_KNOWN_SCALAR})
        list(APPEND _vec_gate_exempt --expect-scalar ${_x})
    endforeach()

    # ALL, so it runs on a plain `make` and a scalar kernel cannot reach a commit. CMAKE_OBJDUMP
    # is passed rather than assumed: the Apple toolchain's objdump and the uenv's binutils are
    # different programs and the script falls back through both.
    add_custom_target(cvfem_vectorization_gate ALL
        COMMAND ${CVFEM_VEC_GATE_PYTHON}
                ${CMAKE_CURRENT_SOURCE_DIR}/src/tests/compile/cvfem_check_vectorization.py
                --objdump "${CMAKE_OBJDUMP}" ${_vec_gate_exempt} ${_vec_gate_objs}
        COMMENT "checking that the lane-blocked kernels emit vector arithmetic"
        VERBATIM)
    # $<TARGET_OBJECTS:> does NOT create a dependency on the target that produces them, so
    # without this the check runs against objects that may not exist yet. It read as a pass on a
    # plain `make all`, where the object libraries happened to be built first, and failed only
    # when the gate was built as a named target on its own -- which is how it was found. The
    # script treating an unreadable object as a failure rather than a skip is what surfaced it.
    add_dependencies(cvfem_vectorization_gate ${_vec_gate_tgts})
endif()
