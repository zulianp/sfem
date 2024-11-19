# SFEMOptions

# ##############################################################################
option(BUILD_SHARED_LIBS "build shared libraries" OFF)
# option(USE_XSDK_DEFAULTS "XSDK mode" OFF)

# option(SFEM_ENABLE_SANITIZER "check for memory access problems" OFF)
option(SFEM_ENABLE_GLIBCXX_DEBUG
       "uses flags -D_GLIBCXX_DEBUG when compiling in debug mode" OFF)
option(SFEM_ENABLE_PROFILER "Allows producing profiles of the runs" OFF)
option(SFEM_ENABLE_MEM_DIAGNOSTICS "Enable mem diagonstics" ON)
option(SFEM_ENABLE_DEV_MODE
       "Add additional flags for more strict compilation" OFF)

option(SFEM_ENABLE_CUDA "Enable CUDA support" OFF)
option(SFEM_ENABLE_OPENMP "Enable OpenMP support" OFF)
option(SFEM_ENABLE_PYTHON "Enable python bindings for SFEM" OFF)

option(SFEM_ENABLE_FP16_JACOBIANS "Enable half precision jacobians when using Cuda" ON)
option(SFEM_ENABLE_AVX2 "Enable AVX2 intrinsics" OFF)
option(SFEM_ENABLE_AVX512 "Enable AVX512 intrinsics" OFF)
option(SFEM_ENABLE_METIS "Enable METIS graph-partitioning" OFF)

option(SFEM_USE_OCCUPANCY_MAX_POTENTIAL "Enable usage of cudaOccupancyMaxPotentialBlockSize" OFF)
# option(SFEM_ENABLE_AVX512 "Enable AVX2 intrinsics" OFF) # TODO

option(SFEM_ENABLE_RESAMPLING "Enable resampling features" ON)

get_directory_property(HAS_PARENT PARENT_DIRECTORY)

if(HAS_PARENT)
    option(SFEM_ENABLE_TESTING "Build the tests" OFF)
    option(SFEM_ENABLE_BENCHMARK "enable benchmark suite" OFF)
else()
    option(SFEM_ENABLE_TESTING "Build the tests" OFF)
    option(SFEM_ENABLE_BENCHMARK "enable benchmark suite" OFF)
endif()

# ##############################################################################
# Handle xSDK defaults
# ##############################################################################

include(CMakeDependentOption)
# cmake_dependent_option(BUILD_SHARED_LIBS "Build shared libraries" OFF
#                        "NOT USE_XSDK_DEFAULTS" ON)

# FIXME: LAPACK supporting long double?
cmake_dependent_option(SFEM_ENABLE_LAPACK "Enable
Lapack for dense matrix operations" ON "NOT SFEM_HAVE_QUAD_PRECISION" OFF)

# MPI is required cmake_dependent_option(SFEM_ENABLE_MPI "Enable MPI
# support" ON "NOT TPL_ENABLE_MPI" OFF)

if(NOT CMAKE_BUILD_TYPE)
    # if(USE_XSDK_DEFAULTS)
    #     set(CMAKE_BUILD_TYPE
    #         "Debug"
    #         CACHE STRING "Choose the type of build, options are: Debug Release
    #     RelWithDebInfo MinSizeRel." FORCE)

    #     message(
    #         STATUS
    #             "[Status] Since USE_XSDK_DEFAULTS=ON then CMAKE_BUILD_TYPE=Debug"
    #     )

    # else()
        set(CMAKE_BUILD_TYPE
            "Release"
            CACHE STRING "Choose the type of build, options are: Debug Release
RelWithDebInfo MinSizeRel." FORCE)

        message(STATUS "[Status] CMAKE_BUILD_TYPE=Release")

    # endif()
endif(NOT CMAKE_BUILD_TYPE)

# ##############################################################################
# ##############################################################################
# ##############################################################################

if(SFEM_ENABLE_DEV_MODE)
    set(SFEM_DEV_FLAGS
        "-Wall -Wextra -pedantic -Werror -Werror=enum-compare -Werror=delete-non-virtual-dtor -Werror=reorder -Werror=return-type" # -Werror=uninitialized
    )
endif()

if(SFEM_ENABLE_GLIBCXX_DEBUG)
    set(SFEM_SPECIAL_DEBUG_FLAGS
        "${SFEM_SPECIAL_DEBUG_FLAGS} -D_GLIBCXX_DEBUG")
endif()

if(SFEM_ENABLE_PROFILER)
    set(SFEM_PROFILING_ENABLED TRUE)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${SFEM_DEV_FLAGS}")
set(CMAKE_CXX_FLAGS_DEBUG
    "${CMAKE_CXX_FLAGS_DEBUG} ${SFEM_SPECIAL_DEBUG_FLAGS}")

if(SFEM_ENABLE_AVX2)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=core-avx2 -DSFEM_ENABLE_AVX2_SORT")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=core-avx2 -DSFEM_ENABLE_AVX2_SORT")
endif()