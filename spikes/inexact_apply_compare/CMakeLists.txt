# Building the inexact-apply spike.
#
# The five benchmarks here all compile the same way: one harness translation
# unit, the generated operator sources beside it, and a family of preprocessor
# macros naming the kernels for one material and one element.  That invocation
# used to be written out by hand in each `run_*.sh`, five nearly identical
# thirty-line blocks, and the copies had drifted -- one knew about the packed
# layout and another did not, one found `mpi.h` through `mpic++` and another
# assumed Homebrew's OpenMP.  This is that invocation, once.
#
#   cmake -S spikes/inexact_apply_compare -B build-spike \
#         -DSPIKE_MATERIAL=neohookean_ogden -DSPIKE_ELEMENT=HEX8
#   cmake --build build-spike -j
#   ./build-spike/bench_split 5
#
# Kernels are generated at *configure* time, not build time, because the set of
# generated sources is not known until the generator has run and CMake resolves
# its source lists while configuring.  Reconfigure after changing the material
# or the element; the generator is skipped when its output is already there.
cmake_minimum_required(VERSION 3.20)
project(sfem_inexact_apply_compare CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# ---------------------------------------------------------------- what to build

set(SPIKE_MATERIAL "neohookean_ogden" CACHE STRING
    "Material to generate and benchmark")
set(SPIKE_ELEMENT "HEX8" CACHE STRING
    "Element type: TET4, HEX8 or TET10")
set_property(CACHE SPIKE_ELEMENT PROPERTY STRINGS TET4 HEX8 TET10)
set(SPIKE_MESH_ORDER "morton3" CACHE STRING
    "Node ordering the harness builds its mesh in")
set(SPIKE_PACK_SIZE "512" CACHE STRING
    "Default elements per pack; the harnesses also read PACK_SIZE from the environment")
set(SPIKE_ARCH_FLAGS "-march=native" CACHE STRING
    "Architecture flags; set to -mcpu=apple-m1 on Apple silicon if -march is rejected")

string(TOLOWER "${SPIKE_ELEMENT}" SPIKE_ELEMENT_LOWER)

# ------------------------------------------------------------------- where from

get_filename_component(SPIKE_WORKTREE "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
# This worktree by default, not the neighbouring checkout: the Op wrappers under
# test are the ones generated here, and `sfem::Op::inexact_supported()` exists in
# this tree and not in the other.  Point it elsewhere to measure another tree.
set(SPIKE_SFEM "${SPIKE_WORKTREE}" CACHE PATH
    "Checkout supplying base/, algebra/, operators/, frontend/ and external/smesh")
set(SPIKE_SFEM_BUILD "${SPIKE_SFEM}/build" CACHE PATH
    "Configured SFEM build tree, for its generated headers and libsfem")
set(SPIKE_GEN_ROOT "${CMAKE_CURRENT_BINARY_DIR}/gen" CACHE PATH
    "Where generated kernels are written")

find_program(SPIKE_PYTHON NAMES python3 python
             HINTS "${SPIKE_SFEM}/venv/bin" "${SPIKE_SFEM}/.venv/bin"
                   "${SPIKE_WORKTREE}/venv/bin" "${SPIKE_WORKTREE}/.venv/bin"
             NO_CMAKE_SYSTEM_PATH)
if(NOT SPIKE_PYTHON)
    find_program(SPIKE_PYTHON NAMES python3 python REQUIRED)
endif()

# `mpic++` was being guessed at by hand in every script, and one of them fell
# back to plain `c++`, which cannot find `mpi.h` -- smesh's base header includes
# it unconditionally.  Let CMake find it.
find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP)
if(NOT OpenMP_CXX_FOUND)
    message(WARNING
        "OpenMP not found: the kernels carry `#pragma omp` and will run serially. "
        "On macOS install libomp and pass -DOpenMP_ROOT=$(brew --prefix libomp).")
endif()

# ------------------------------------------------------------------- generation

set(SPIKE_GEN_DIR "${SPIKE_GEN_ROOT}/${SPIKE_MATERIAL}/d3/${SPIKE_ELEMENT_LOWER}")

# --------------------------------------------------- which element owns a kernel
#
# A mesh-ordered tensor-product element publishes no kernel body of its own: it
# forwards to its lexicographic twin, HEX8 to PROTEUS_HEX8.  So two names are in
# play -- `SPIKE_ELEMENT` for the mesh and the `extern "C"` symbols, and
# `SPIKE_KERNEL_ELEMENT` for the header holding the `..._impl` templates these
# benchmarks instantiate directly.  `SPIKE_SHAPE_ORDER` is the permutation that
# goes with it, which the harness applies because instantiating the template
# bypasses the forwarder that would otherwise have applied it.
#
# Asked of `plans/layout.cartesian_twin` rather than tabulated here, for the same
# reason `kernel_element.sh` asks: a copy of the pairing would be silent when it
# drifted -- the benchmark would still build, against whichever kernel it found.
execute_process(
    COMMAND "${SPIKE_PYTHON}" -c
"import sys
sys.path.insert(0, r'${SPIKE_WORKTREE}/python')
from codegen.framework.plans.layout import cartesian_twin
twin = cartesian_twin('${SPIKE_ELEMENT}')
print((twin.twin_name if twin is not None else '${SPIKE_ELEMENT}').upper())
print('' if twin is None else ','.join(str(n) for n in twin.shape_order))"
    OUTPUT_VARIABLE _spike_twin
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE _spike_twin_status)
if(NOT _spike_twin_status EQUAL 0)
    message(FATAL_ERROR
        "could not ask plans/layout which element owns ${SPIKE_ELEMENT}'s kernel; "
        "is SPIKE_PYTHON the venv the kernels are generated with?")
endif()
string(REPLACE "\n" ";" _spike_twin "${_spike_twin}")
list(GET _spike_twin 0 SPIKE_KERNEL_ELEMENT)
list(LENGTH _spike_twin _spike_twin_len)
if(_spike_twin_len GREATER 1)
    list(GET _spike_twin 1 SPIKE_SHAPE_ORDER)
else()
    set(SPIKE_SHAPE_ORDER "")
endif()
string(TOLOWER "${SPIKE_KERNEL_ELEMENT}" SPIKE_KERNEL_ELEMENT_LOWER)
set(SPIKE_KERNEL_DIR "${SPIKE_GEN_ROOT}/${SPIKE_MATERIAL}/d3/${SPIKE_KERNEL_ELEMENT_LOWER}")
if(NOT SPIKE_KERNEL_ELEMENT STREQUAL SPIKE_ELEMENT)
    message(STATUS
        "${SPIKE_ELEMENT} forwards its micro-kernel to ${SPIKE_KERNEL_ELEMENT} "
        "(node order ${SPIKE_SHAPE_ORDER})")
endif()

# A material with several units names its kernels after the unit, so the file
# that proves generation ran is not always the material's own.
file(GLOB _spike_inline_headers "${SPIKE_KERNEL_DIR}/*_inexact_apply_inline.hpp")
if(NOT _spike_inline_headers)
    message(STATUS "Generating ${SPIKE_MATERIAL} ${SPIKE_ELEMENT} kernels (this is slow)")
    execute_process(
        COMMAND "${CMAKE_COMMAND}" -E env "PYTHONPATH=${SPIKE_WORKTREE}/python"
                "${SPIKE_PYTHON}" -c "
import dataclasses, sys
from sfem import gen
from importlib import import_module
material = import_module('codegen.framework.materials.' + sys.argv[1]).material
gen.generate(dataclasses.replace(material, inexact_apply=True),
             sys.argv[2], elements=(sys.argv[3],), clean=False)
" "${SPIKE_MATERIAL}" "${SPIKE_GEN_ROOT}/${SPIKE_MATERIAL}" "${SPIKE_ELEMENT}"
        WORKING_DIRECTORY "${SPIKE_WORKTREE}"
        RESULT_VARIABLE _spike_gen_status)
    if(NOT _spike_gen_status EQUAL 0)
        message(FATAL_ERROR "kernel generation failed (${_spike_gen_status})")
    endif()
    file(GLOB _spike_inline_headers "${SPIKE_KERNEL_DIR}/*_inexact_apply_inline.hpp")
endif()
if(NOT _spike_inline_headers)
    message(FATAL_ERROR "no inexact-apply headers in ${SPIKE_KERNEL_DIR}")
endif()

# The operator sources, minus the inexact ones: those are headers' worth of
# inline templates and are already pulled in by the harness.
file(GLOB SPIKE_UNIT_SOURCES "${SPIKE_GEN_DIR}/*_operator.cpp")
list(FILTER SPIKE_UNIT_SOURCES EXCLUDE REGEX "_inexact_apply_operator\\.cpp$")
# A mesh-order HEX8 forwards to its Cartesian PROTEUS twin, whose translation
# unit holds the kernels the forwarding source calls.
if(SPIKE_ELEMENT STREQUAL "HEX8")
    file(GLOB _spike_proteus "${SPIKE_GEN_ROOT}/${SPIKE_MATERIAL}/d3/proteus_hex8/*_operator.cpp")
    list(FILTER _spike_proteus EXCLUDE REGEX "_inexact_apply_operator\\.cpp$")
    list(APPEND SPIKE_UNIT_SOURCES ${_spike_proteus})
endif()

# One object library so the generated translation units -- the slow ones --
# build concurrently under `-j`.  The hand-written scripts passed every source
# to a single compiler invocation, which compiles them one after another.
add_library(spike_kernels OBJECT ${SPIKE_UNIT_SOURCES})

# ----------------------------------------------------------------- common flags

file(GLOB_RECURSE _spike_smesh_dirs LIST_DIRECTORIES true
     "${SPIKE_SFEM}/external/smesh/src/*")
set(SPIKE_SMESH_INCLUDES "${SPIKE_SFEM}/external/smesh/src")
foreach(_dir IN LISTS _spike_smesh_dirs)
    if(IS_DIRECTORY "${_dir}")
        list(APPEND SPIKE_SMESH_INCLUDES "${_dir}")
    endif()
endforeach()

add_library(spike_common INTERFACE)
target_include_directories(spike_common INTERFACE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${SPIKE_WORKTREE}/python/codegen/framework/tools"
    "${SPIKE_GEN_DIR}"
    "${SPIKE_GEN_ROOT}/${SPIKE_MATERIAL}"
    "${SPIKE_GEN_ROOT}/${SPIKE_MATERIAL}/d3"
    "${SPIKE_SFEM}/base"
    "${SPIKE_SFEM}/algebra"
    "${SPIKE_SFEM}/operators"
    "${SPIKE_SFEM_BUILD}"
    "${SPIKE_SFEM_BUILD}/external/smesh"
    ${SPIKE_SMESH_INCLUDES})
target_compile_definitions(spike_common INTERFACE
    NDEBUG
    ELEMENT_${SPIKE_ELEMENT}
    PACK_SIZE=${SPIKE_PACK_SIZE})
# Not a generator expression: the order is a comma-separated list and `$<BOOL:>`
# takes exactly one parameter.
if(SPIKE_SHAPE_ORDER)
    target_compile_definitions(spike_common INTERFACE
        KERNEL_SHAPE_ORDER=${SPIKE_SHAPE_ORDER})
endif()
target_include_directories(spike_common INTERFACE "${SPIKE_KERNEL_DIR}")
target_compile_options(spike_common INTERFACE
    $<$<COMPILE_LANGUAGE:CXX>:${SPIKE_ARCH_FLAGS}>)
target_link_libraries(spike_common INTERFACE MPI::MPI_CXX)
if(OpenMP_CXX_FOUND)
    target_link_libraries(spike_common INTERFACE OpenMP::OpenMP_CXX)
endif()
target_include_directories(spike_kernels PRIVATE
    $<TARGET_PROPERTY:spike_common,INTERFACE_INCLUDE_DIRECTORIES>)
target_compile_definitions(spike_kernels PRIVATE
    $<TARGET_PROPERTY:spike_common,INTERFACE_COMPILE_DEFINITIONS>)
target_compile_options(spike_kernels PRIVATE
    $<TARGET_PROPERTY:spike_common,INTERFACE_COMPILE_OPTIONS>)
if(OpenMP_CXX_FOUND)
    target_link_libraries(spike_kernels PRIVATE OpenMP::OpenMP_CXX)
endif()
target_link_libraries(spike_kernels PRIVATE MPI::MPI_CXX)

# --------------------------------------------------------------- the benchmarks
#
# Every kernel name is `<unit>_<element>_<what>`, so a benchmark says which units
# it wants and the macros follow mechanically.  This is the part that was copied
# by hand five times.

function(spike_kernel_macros out_var unit prefix)
    # The templated bodies belong to whichever element owns the micro-kernel.
    set(_stem "${unit}_${SPIKE_KERNEL_ELEMENT_LOWER}")
    set(_macros
        ${prefix}_INEXACT_HEADER=\"${_stem}_inexact_apply_inline.hpp\"
        ${prefix}_TANGENT=sfem::codegen::${_stem}_inexact_apply_tangent_a_msoa_impl
        ${prefix}_STORED=sfem::codegen::${_stem}_inexact_apply_stored_a_msoa_impl
        ${prefix}_COMPRESS=sfem::codegen::${_stem}_inexact_apply_compressed_a_msoa_impl
        ${prefix}_PACKED_STORED=sfem::codegen::${_stem}_inexact_apply_stored_packed_two_pass_a_msoa_impl)
    set(${out_var} "${_macros}" PARENT_SCOPE)
endfunction()

# The two-unit material: an energy unit whose exact entry point is `apply` and a
# residual unit whose exact entry point is `jacobian_action`.
if(EXISTS "${SPIKE_KERNEL_DIR}/${SPIKE_MATERIAL}_elastic_${SPIKE_KERNEL_ELEMENT_LOWER}_inexact_apply_inline.hpp")
    set(_elastic "${SPIKE_MATERIAL}_elastic")
    set(_viscous "${SPIKE_MATERIAL}_viscous")
    spike_kernel_macros(_elastic_macros "${_elastic}" ELASTIC)
    spike_kernel_macros(_viscous_macros "${_viscous}" VISCOUS)
    add_executable(bench_mixed bench_mixed.cpp $<TARGET_OBJECTS:spike_kernels>)
    target_link_libraries(bench_mixed PRIVATE spike_common)
    target_compile_definitions(bench_mixed PRIVATE
        ${_elastic_macros} ${_viscous_macros}
        EXACT_ELASTIC_APPLY=${_elastic}_${SPIKE_ELEMENT_LOWER}_apply_a_msoa
        EXACT_VISCOUS_ACTION=${_viscous}_${SPIKE_ELEMENT_LOWER}_jacobian_action_a_msoa
        PACKED_STORED_APPLY)
    # The assembled Jacobian, when the material publishes one.  A material with
    # no `matrix_formats` has no BSR entry point, and the column is simply absent
    # rather than the benchmark failing to link.
    if(EXISTS "${SPIKE_GEN_DIR}/${_elastic}_${SPIKE_ELEMENT_LOWER}_matrix_format_operator.cpp")
        target_compile_definitions(bench_mixed PRIVATE
            ELASTIC_HESSIAN_BSR=${_elastic}_${SPIKE_ELEMENT_LOWER}_hessian_bsr_i_msoa
            VISCOUS_HESSIAN_BSR=${_viscous}_${SPIKE_ELEMENT_LOWER}_hessian_bsr_i_msoa)
        message(STATUS "bench_mixed: BSR column enabled")
    endif()
    message(STATUS "bench_mixed: ${_elastic} + ${_viscous} on ${SPIKE_ELEMENT}")
endif()

# The single-unit materials.
if(EXISTS "${SPIKE_KERNEL_DIR}/${SPIKE_MATERIAL}_${SPIKE_KERNEL_ELEMENT_LOWER}_inexact_apply_inline.hpp")
    # `_stem` names the C ABI symbols, which stay this element's; `_kstem` names
    # the templates, which are the owning element's.  They differ only when this
    # element forwards.
    set(_stem "${SPIKE_MATERIAL}_${SPIKE_ELEMENT_LOWER}")
    set(_kstem "${SPIKE_MATERIAL}_${SPIKE_KERNEL_ELEMENT_LOWER}")
    set(_single_macros
        MATERIAL_LABEL=\"${SPIKE_MATERIAL}\"
        MATERIAL_INEXACT_HEADER=\"${_kstem}_inexact_apply_inline.hpp\"
        EXACT_APPLY=${_stem}_apply_a_msoa
        TANGENT_KERNEL=${_kstem}_inexact_apply_tangent_a_msoa_impl
        STORED_APPLY=${_kstem}_inexact_apply_stored_a_msoa_impl
        COMPRESSED_APPLY=${_kstem}_inexact_apply_compressed_a_msoa_impl
        PACKED_EXACT_APPLY=${_stem}_apply_packed_two_pass_a_msoa
        PACKED_STORED_APPLY=${_kstem}_inexact_apply_stored_packed_two_pass_a_msoa_impl)
    # Whether the exact apply takes the current state is a property of the
    # material, not of the harness: a hyperelastic tangent needs it and a linear
    # one does not.  Omitting it does not fail to compile -- the harness simply
    # builds a different call -- it segfaults at run time, which is how this was
    # found.
    if(NOT SPIKE_MATERIAL STREQUAL "linear_elasticity")
        list(APPEND _single_macros EXACT_TAKES_STATE)
    endif()

    add_executable(bench_split bench_split.cpp $<TARGET_OBJECTS:spike_kernels>)
    target_link_libraries(bench_split PRIVATE spike_common)
    target_compile_definitions(bench_split PRIVATE
        ${_single_macros} MESH_ORDER=\"${SPIKE_MESH_ORDER}\")
    if(EXISTS "${SPIKE_GEN_DIR}/${_stem}_matrix_format_operator.cpp")
        target_compile_definitions(bench_split PRIVATE
            MATERIAL_HESSIAN_BSR=${_stem}_hessian_bsr_i_msoa)
        message(STATUS "bench_split: BSR column enabled")
    endif()

    add_executable(warp_sweep warp_sweep.cpp $<TARGET_OBJECTS:spike_kernels>)
    target_link_libraries(warp_sweep PRIVATE spike_common)
    target_compile_definitions(warp_sweep PRIVATE ${_single_macros})

    # The Op-level harnesses name the same kernels, but qualified: they are
    # compiled beside `sfem_API.hpp` rather than inside the generated namespace.
    set(SPIKE_SINGLE_UNIT_MACROS
        MATERIAL_LABEL=\"${SPIKE_MATERIAL}\"
        MATERIAL_INEXACT_HEADER=\"${_kstem}_inexact_apply_inline.hpp\"
        TANGENT_KERNEL=sfem::codegen::${_stem}_inexact_apply_tangent_a_msoa_impl
        STORED_APPLY=sfem::codegen::${_stem}_inexact_apply_stored_a_msoa_impl
        COMPRESSED_APPLY=sfem::codegen::${_stem}_inexact_apply_compressed_a_msoa_impl)
    message(STATUS "bench_split, warp_sweep: ${SPIKE_MATERIAL} on ${SPIKE_ELEMENT}")
endif()

# `scale_probe.cpp` is deliberately not a target: it names one kernel header
# literally, on line 13, because its whole point is to isolate a single kernel
# with no harness around it.  It is edited by hand for whatever is being chased
# and compiled the same way, which is why it stayed useful when the harnesses
# were the thing under suspicion.

if(NOT TARGET bench_split AND NOT TARGET bench_mixed)
    message(FATAL_ERROR
        "no harness matched the generated kernels in ${SPIKE_GEN_DIR}: "
        "expected <material>_<element>_inexact_apply_inline.hpp or the "
        "_elastic/_viscous pair")
endif()

# ------------------------------------------------- benchmarks needing all of SFEM
#
# `bench_op_split` and `op_inexact_check` go through the Op wrappers rather than
# the kernels, so they need libsfem itself.  The shell scripts got its link line
# by `sed`-ing `CMakeFiles/bench_op.dir/link.txt` out of the SFEM build tree,
# which breaks whenever that target is renamed.  Ask for the library instead.
find_library(SPIKE_SFEM_LIBRARY NAMES sfem HINTS "${SPIKE_SFEM_BUILD}"
             NO_DEFAULT_PATH)
# SFEM's frontend headers reach its FetchContent dependencies -- `ryml.hpp` and
# matrix.io -- so those directories are part of using the library.  Found by
# looking in the build tree's `_deps`, which is where FetchContent puts them,
# rather than by scraping `CMakeFiles/<target>.dir/flags.make`: the scrape breaks
# whenever the target it borrows from is renamed, and it borrowed from `bench_op`.
set(SPIKE_SFEM_DEP_INCLUDES "")
foreach(_dep_root "${SPIKE_SFEM_BUILD}/_deps" "${SPIKE_SFEM_BUILD}/external/smesh/_deps")
    file(GLOB _dep_srcs "${_dep_root}/*-src")
    foreach(_dep_src IN LISTS _dep_srcs)
        foreach(_candidate "${_dep_src}/src" "${_dep_src}/include" "${_dep_src}")
            if(IS_DIRECTORY "${_candidate}")
                list(APPEND SPIKE_SFEM_DEP_INCLUDES "${_candidate}")
                break()
            endif()
        endforeach()
        # ryml vendors c4core under `ext/`, and its own headers include
        # `c4/substr.hpp` from there.
        file(GLOB _vendored "${_dep_src}/ext/*")
        foreach(_vendor IN LISTS _vendored)
            # ryml vendors it as `ext/c4core.src`, whose headers sit directly
            # under the directory rather than in a `src/` or `include/`.
            foreach(_candidate "${_vendor}/src" "${_vendor}/include" "${_vendor}")
                if(IS_DIRECTORY "${_candidate}")
                    list(APPEND SPIKE_SFEM_DEP_INCLUDES "${_candidate}")
                    break()
                endif()
            endforeach()
        endforeach()
    endforeach()
endforeach()

# The library's public headers are its source directories -- there is no install
# step and no exported target set -- so they are collected from the checkout.
# The alternative the shell scripts used was to `sed` 111 `-I` flags out of
# `CMakeFiles/bench_op.dir/flags.make`, which breaks when that target is renamed.
set(SPIKE_SFEM_SOURCE_INCLUDES "${SPIKE_SFEM}")
foreach(_root algebra base frontend mesh operators resampling solver graphs io parametrize)
    if(IS_DIRECTORY "${SPIKE_SFEM}/${_root}")
        list(APPEND SPIKE_SFEM_SOURCE_INCLUDES "${SPIKE_SFEM}/${_root}")
        file(GLOB_RECURSE _sub LIST_DIRECTORIES true "${SPIKE_SFEM}/${_root}/*")
        foreach(_dir IN LISTS _sub)
            if(IS_DIRECTORY "${_dir}")
                list(APPEND SPIKE_SFEM_SOURCE_INCLUDES "${_dir}")
            endif()
        endforeach()
    endif()
endforeach()
list(REMOVE_DUPLICATES SPIKE_SFEM_SOURCE_INCLUDES)

# Off by default because both harnesses are stale against the current kernels:
# `bench_op_split.exe.cpp` calls the stored apply with its pre-SoA argument list,
# from before the store became component-major with a single
# `tangent_component_stride`, and fails to compile.  The include and link setup
# below is correct and was verified to get them as far as that error, so turning
# this on is the right way to start fixing them.
option(SPIKE_BUILD_OP_BENCHMARKS
       "Build the Op-level benchmarks, which link libsfem (currently stale)" OFF)

# These two reach the kernels through the Op wrappers, so they need the material's
# inexact header named the same way the kernel benchmarks name it -- and a
# single-unit material, because that is the shape their macro set describes.
if(SPIKE_SFEM_LIBRARY AND SPIKE_BUILD_OP_BENCHMARKS AND SPIKE_SINGLE_UNIT_MACROS)
    foreach(_op_bench bench_op_split op_inexact_check)
        if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${_op_bench}.exe.cpp")
            add_executable(${_op_bench} "${_op_bench}.exe.cpp")
            target_link_libraries(${_op_bench} PRIVATE spike_common "${SPIKE_SFEM_LIBRARY}")
            target_compile_definitions(${_op_bench} PRIVATE ${SPIKE_SINGLE_UNIT_MACROS})
            target_include_directories(${_op_bench} PRIVATE
                ${SPIKE_SFEM_SOURCE_INCLUDES} ${SPIKE_SFEM_DEP_INCLUDES})
        endif()
    endforeach()
    message(STATUS "Op-level benchmarks linked against ${SPIKE_SFEM_LIBRARY}")
elseif(NOT SPIKE_SFEM_LIBRARY)
    message(STATUS
        "libsfem not found under ${SPIKE_SFEM_BUILD}: skipping bench_op_split and "
        "op_inexact_check, which go through the Op wrappers rather than the kernels")
endif()
