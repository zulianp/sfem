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
    # Where the dependencies install to lib64 rather than lib, they are found from the prefix.
    get_filename_component(_sfem_prefix "${SFEM_DIR}/../.." ABSOLUTE)
    list(APPEND CMAKE_PREFIX_PATH "${_sfem_prefix}")
endif()

find_package(ryml CONFIG REQUIRED)
find_package(matrixio CONFIG REQUIRED)
find_package(smesh CONFIG REQUIRED)
# Optional in SFEM, so present only when SFEM was built with them.
find_package(SCCD CONFIG QUIET)
find_package(ssdf CONFIG QUIET)
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

# The operator's gate is frontend/tests/sfem_GeneratedMooneyRivlinPronyTest.cpp in SFEM: finite-
# difference checks of value, gradient, apply and the assembled tangent after committed history
# steps, on an axis-aligned and a distorted mesh, with and without Prony terms.

# The end-to-end scenario: twist, hold, release, recover. Slow enough to be excluded from a
# default `ctest`; run it with `ctest -L scenario`.
add_test(NAME prony_torsion_release
         COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/run_torsion_release.sh
         WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
set_tests_properties(prony_torsion_release PROPERTIES
    LABELS scenario
    ENVIRONMENT "PRONY_EXE=$<TARGET_FILE:prony_visco_torsion>")
