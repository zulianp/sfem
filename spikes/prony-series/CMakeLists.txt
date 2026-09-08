cmake_minimum_required(VERSION 3.20)

project(prony_series LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)

# A single -DSFEM_DIR=<prefix>/lib/cmake is enough to find everything SFEM was installed with.
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

add_executable(prony_visco_torsion
    prony_visco_torsion.cpp
    prony_case.cpp)

target_include_directories(prony_visco_torsion PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(prony_visco_torsion
    PRIVATE
        SFEM::sfem
        $<$<BOOL:${OpenMP_CXX_FOUND}>:OpenMP::OpenMP_CXX>)

# SFEM's YAML-facing declarations sit behind SFEM_ENABLE_RYAML, and neither SFEM nor smesh
# export the define to consumers. The installed library was built with it -- both
# DirichletConditions::create_from_yaml overloads are in libsfem.a -- so switching it on here
# only makes the declarations visible; it does not change what is linked.
target_compile_definitions(prony_visco_torsion PRIVATE SFEM_ENABLE_RYAML)

# Tracing is likewise compiled into SFEM but not exported, so the SFEM_TRACE_SCOPE in this
# spike's own translation units would expand to nothing without it.
option(PRONY_ENABLE_TRACE "Compile SFEM_TRACE_SCOPE in this spike's own sources" ON)
if(PRONY_ENABLE_TRACE)
    target_compile_definitions(prony_visco_torsion PRIVATE SMESH_ENABLE_TRACE SFEM_ENABLE_TRACE)
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    target_compile_options(prony_visco_torsion PRIVATE -O3 -DNDEBUG)
endif()

# Gate on the operator the driver rests on: the assembled tangent must be the derivative of
# the assembled gradient, at a state that carries Prony history. Cheap; run it in CI.
#
# This gate found three real defects, all now fixed in SFEM proper. Two were in the emitter for
# the history geometric-stiffness kernel, python/codegen/sr_visco_hyper_unique_Hi.py: it wrote
# the element matrix node-major into a component-major layout, and it contracted the wrong index
# of the inverse Jacobian. The third was in hex8_mooney_rivlin_visco_flexible.cpp, which read
# the element diagonal at node-major positions out of the same component-major matrix.
#
# Keep all three variants. Each one is the only thing that can see its own defect: the cube
# catches the ordering, the distorted mesh catches the Jacobian transpose (on an axis-aligned
# mesh J is diagonal and J^-1 is its own transpose, so it is invisible there), and the elastic
# variant separates the Mooney-Rivlin backbone from the viscoelastic part above it.
#
# Never make these green by raising PRONY_CHECK_TOL. The clean value is ~2e-11, set by round-off
# in the differenced gradient; the smallest structural error seen here was 3.6e-5.
add_executable(prony_tangent_check prony_tangent_check.cpp)
target_include_directories(prony_tangent_check PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(prony_tangent_check
    PRIVATE
        SFEM::sfem
        $<$<BOOL:${OpenMP_CXX_FOUND}>:OpenMP::OpenMP_CXX>)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    target_compile_options(prony_tangent_check PRIVATE -O3 -DNDEBUG)
endif()
add_test(NAME prony_tangent_check COMMAND prony_tangent_check)

# The same check with the Prony series switched off, which separates a fault in the
# viscoelastic part of the tangent from one in the Mooney-Rivlin backbone underneath it.
add_test(NAME prony_tangent_check_elastic COMMAND prony_tangent_check)
set_tests_properties(prony_tangent_check_elastic PROPERTIES ENVIRONMENT "PRONY_CHECK_TERMS=0")

# On distorted elements. Required: a transposed inverse Jacobian in an element kernel cannot be
# observed on an axis-aligned mesh at all, so the cube variant above passed throughout the time
# that defect was present.
add_test(NAME prony_tangent_check_distorted COMMAND prony_tangent_check)
set_tests_properties(prony_tangent_check_distorted PROPERTIES ENVIRONMENT "PRONY_CHECK_DISTORTION=0.15")

# The end-to-end scenario: twist, hold, release, recover. Slow enough to be excluded from a
# default `ctest`; run it with `ctest -L scenario`.
add_test(NAME prony_torsion_release
         COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/run_torsion_release.sh
         WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
set_tests_properties(prony_torsion_release PROPERTIES
    LABELS scenario
    ENVIRONMENT "PRONY_EXE=$<TARGET_FILE:prony_visco_torsion>")
