# Compile-check the generated operator wrappers against an SFEM build.
#
# The codegen framework emits C++ that nothing in its own test suite compiles.
# `run_m9_regression.sh` says "Generated wrapper syntax checks: skipped where
# ryml.hpp is unavailable", and in a bare worktree that is always: ryml is a
# fetched dependency and lives in a configured build tree, not in the sources.
# So a change to `package/op_wrappers.py` -- the template every generated
# `sfem::Op` is stamped from -- ships unverified, and a syntax error in it
# breaks every material at once.
#
# This spike closes that hole without touching the main build. It takes an
# existing SFEM build or install directory as a dependency, compiles the
# generated sources as an OBJECT library, and stops there: no link, no run.
# Compiling is the whole point, because the failure being guarded against is a
# generated translation unit that does not compile.
#
# Usage:
#
#   cmake -S spikes/codegen_wrapper_compile -B /tmp/wrapper-check \
#         -DSFEM_ROOT=/path/to/sfem \
#         -DSFEM_BUILD_DIR=/path/to/sfem/build \
#         -DGENERATED_DIR=/path/to/generated
#   cmake --build /tmp/wrapper-check
#
# SFEM_ROOT defaults to this repository, SFEM_BUILD_DIR to <SFEM_ROOT>/build.
# GENERATED_DIR is whatever `sfem.gen` wrote; point it at one material or at a
# whole snapshot tree.

cmake_minimum_required(VERSION 3.20)
project(sfem_codegen_wrapper_compile CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# SMESH headers include <mpi.h> unconditionally, so a compile check needs MPI's
# include path even though nothing here links or runs.
find_package(MPI COMPONENTS CXX)

get_filename_component(_default_root "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
set(SFEM_ROOT "${_default_root}" CACHE PATH "SFEM source tree")
set(SFEM_BUILD_DIR "${SFEM_ROOT}/build" CACHE PATH "A configured SFEM build or install tree")
set(GENERATED_DIR "" CACHE PATH "Directory of generated sources to compile")

if(NOT GENERATED_DIR)
    message(FATAL_ERROR "GENERATED_DIR must point at a codegen output directory")
endif()
if(NOT EXISTS "${SFEM_BUILD_DIR}")
    message(FATAL_ERROR "SFEM_BUILD_DIR '${SFEM_BUILD_DIR}' does not exist; configure SFEM first")
endif()

# The generated wrappers include SFEM and SMESH headers by bare name, so every
# directory holding one has to be on the include path.  Globbing is right here
# and wrong in a real build: this spike follows whatever the generator emitted,
# which is the thing under test.
function(_collect_include_dirs out_var)
    set(_dirs "")
    foreach(_root IN LISTS ARGN)
        if(NOT EXISTS "${_root}")
            continue()
        endif()
        # The root itself, for headers included through a subdirectory prefix
        # the way ryml includes "c4/substr.hpp".
        list(APPEND _dirs "${_root}")
        file(GLOB_RECURSE _headers "${_root}/*.hpp" "${_root}/*.h")
        foreach(_header IN LISTS _headers)
            get_filename_component(_dir "${_header}" DIRECTORY)
            if(NOT _dir MATCHES "/(cuda|CMakeFiles)(/|$)")
                list(APPEND _dirs "${_dir}")
            endif()
        endforeach()
    endforeach()
    list(REMOVE_DUPLICATES _dirs)
    set(${out_var} "${_dirs}" PARENT_SCOPE)
endfunction()

_collect_include_dirs(SFEM_INCLUDE_DIRS
    "${SFEM_ROOT}/base"
    "${SFEM_ROOT}/frontend"
    "${SFEM_ROOT}/operators"
    "${SFEM_ROOT}/algebra"
    "${SFEM_ROOT}/solver"
    "${SFEM_ROOT}/external/smesh/src"
    "${SFEM_BUILD_DIR}/_deps/ryml-src/src"
    "${SFEM_BUILD_DIR}/_deps/ryml-src/ext/c4core/src"   # c4core is vendored inside ryml
    "${GENERATED_DIR}"
)
# Configured headers -- sfem_config.h and friends -- sit at the build root.
list(APPEND SFEM_INCLUDE_DIRS "${SFEM_BUILD_DIR}" "${SFEM_BUILD_DIR}/external/smesh")

file(GLOB_RECURSE GENERATED_SOURCES "${GENERATED_DIR}/*.cpp")
list(FILTER GENERATED_SOURCES EXCLUDE REGEX "/cuda/")
if(NOT GENERATED_SOURCES)
    message(FATAL_ERROR "no .cpp files under GENERATED_DIR '${GENERATED_DIR}'")
endif()
list(LENGTH GENERATED_SOURCES _n)
message(STATUS "compile-checking ${_n} generated translation units")

# OBJECT: compile every unit, link nothing.  A generated wrapper that does not
# compile is the failure this exists to catch; whether it links is the main
# build's business.
add_library(generated_wrappers OBJECT ${GENERATED_SOURCES})
target_include_directories(generated_wrappers PRIVATE ${SFEM_INCLUDE_DIRS})
target_compile_options(generated_wrappers PRIVATE -w)
if(MPI_CXX_FOUND)
    target_include_directories(generated_wrappers PRIVATE ${MPI_CXX_INCLUDE_DIRS})
else()
    message(WARNING "MPI not found; smesh headers include <mpi.h> and this will fail")
endif()
