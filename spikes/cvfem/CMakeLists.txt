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

set(_cvfem_perf_opts)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    list(APPEND _cvfem_perf_opts
        -O3
        -DNDEBUG
        -ffast-math
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

foreach(_cvfem_tgt cvfem_tet4_ns_upwind_bench cvfem_tet4_ns_upwind_bench_baseline)
    add_executable(${_cvfem_tgt} ${_cvfem_tgt}.cpp)
    target_link_libraries(${_cvfem_tgt}
        PRIVATE
            SFEM::sfem
            $<$<BOOL:${OpenMP_CXX_FOUND}>:OpenMP::OpenMP_CXX>)
    target_compile_options(${_cvfem_tgt} PRIVATE ${_cvfem_perf_opts})
    target_include_directories(${_cvfem_tgt} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
endforeach()

# set_target_properties(cvfem_tet4_ns_upwind_bench PROPERTIES
#     RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
