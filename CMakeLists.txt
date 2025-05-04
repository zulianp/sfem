cmake_minimum_required(VERSION 3.5...3.19)
cmake_policy(SET CMP0048 NEW)

project(
    sfem
    VERSION 0.1.0
    LANGUAGES "C" "CXX"
    HOMEPAGE_URL "https://bitbucket.org/zulianp/sfem"
    DESCRIPTION
    "SFEM: Simple FEM"
    )

# Std CMake imports
include(CMakePackageConfigHelpers)

# Our CMake imports
include(cmake/SFEMCompilerFeatures.cmake)
include(cmake/SFEMVersion.cmake)
include(cmake/SFEMDefaults.cmake)
include(cmake/SFEMOptions.cmake) 
include(cmake/SFEMNumbers.cmake) # More options here
include(cmake/SFEMBuildTypes.cmake)
include(cmake/SFEMCMakeFunctions.cmake)
include(cmake/SFEMDependencies.cmake)

if(SFEM_ENABLE_PYTHON)
    include(cmake/SFEMPython.cmake)
endif()

######################################################
######################################################
# Library
######################################################

find_package(MatrixIO REQUIRED)

if(SFEM_ENABLE_ISOLVER)
    find_package(ISolver REQUIRED)
endif()

set(SFEM_MODULES
    algebra base matrix 
    algebra/openmp
    mesh 
    mesh/sshex8
    operators 
    operators/beam2 
    operators/boundary_conditions
    operators/contact
    operators/cvfem 
    operators/cvfem_quad4
    operators/cvfem_tet4
    operators/cvfem_tri3
    operators/edgeshell2
    operators/hex8
    operators/hex8/dg
    operators/hierarchical
    operators/macro_tet4
    operators/macro_tri3
    operators/navier_stokes
    operators/quad4
    operators/quad4/dg
    operators/quadshell4
    operators/spectral_hex
    operators/spectral_hex/dg
    operators/sshex8
    operators/ssquad4
    operators/sstet4
    operators/stencil
    operators/tet10
    operators/tet4
    operators/tri3
    operators/tri6
    operators/trishell3
    operators/trishell6
    pizzastack 
    solver
    ssmg
)

if(SFEM_ENABLE_EXPLICIT_VECTORIZATION)
    list(APPEND SFEM_MODULES operators/tet4/vectorized)
endif()

if(SFEM_ENABLE_AMG)
    list(APPEND SFEM_MODULES amg)
endif()

if(WIN32)
set(SFEM_ENABLE_RESAMPLING OFF)
endif()

if(SFEM_ENABLE_RESAMPLING)
    list(APPEND SFEM_MODULES
        resampling
        resampling/tet10
    )
endif()

list(APPEND SFEM_MODULES 
    resampling/interpolate
    resampling/quadshell4
    resampling/trishell3
    resampling/beam2
    resampling/surface
    )

list(APPEND SFEM_MODULES frontend)

if(SFEM_ENABLE_METIS)
    list(APPEND SFEM_MODULES graphs)
endif()

if(SFEM_ENABLE_CUDA)
     list(APPEND SFEM_MODULES 
        algebra/cuda
        frontend/cuda
        operators/boundary_conditions/cuda
        operators/contact/cuda
        operators/cuda
        operators/hex8/cuda
        operators/hierarchical/cuda
        operators/macro_tet4/cuda
        operators/quadshell4/cuda
        operators/spectral_hex/cuda
        operators/sshex8/cuda
        operators/ssquad4/cuda
        operators/sstet4/cuda
        operators/tet10/cuda
        operators/tet4/cuda
        resampling/cuda
        )

    if(SFEM_ENABLE_AMG)
        list(APPEND SFEM_MODULES amg/cuda)
    endif()
endif()

foreach(MODULE ${SFEM_MODULES})
    scan_directories(${CMAKE_CURRENT_SOURCE_DIR}/${MODULE} "." SFEM_BUILD_INCLUDES
       SFEM_HEADERS SFEM_SOURCES)

    if(SFEM_ENABLE_CUDA)
        file(GLOB SFEM_MODULE_KERNEL_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/${MODULE}/*.cu")
        file(GLOB SFEM_MODULE_KERNEL_HEADERS CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/${MODULE}/*.cuh")
        # message(STATUS "${MODULE} : ${SFEM_MODULE_KERNEL_SOURCES}")
    endif()

    if(SFEM_MODULE_KERNEL_SOURCES)
        list(APPEND SFEM_KERNEL_SOURCES "${SFEM_MODULE_KERNEL_SOURCES}")
    endif()

    if(SFEM_MODULE_KERNEL_HEADERS)
        list(APPEND SFEM_KERNEL_HEADERS "${SFEM_MODULE_KERNEL_HEADERS}")
    endif()
endforeach(MODULE)

if(SFEM_ENABLE_CUDA)
    list(APPEND SFEM_SOURCES "${SFEM_KERNEL_SOURCES}")
    list(APPEND SFEM_HEADERS "${SFEM_KERNEL_HEADERS}")
endif()

# message(STATUS "SFEM_BUILD_INCLUDES: ${SFEM_BUILD_INCLUDES}")
# message(STATUS "SFEM_HEADERS: ${SFEM_HEADERS}") 
# message(STATUS "SFEM_SOURCES: ${SFEM_SOURCES}")
# message(STATUS "SFEM_KERNEL_SOURCES: ${SFEM_KERNEL_SOURCES}")

string(TOLOWER ${CMAKE_BUILD_TYPE} LBT)

if(NOT LBT STREQUAL "debug" AND NOT LBT STREQUAL "asan" AND NOT CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
    if (SFEM_VEC_SIZE STREQUAL "4" OR SFEM_CPU_ARCH STREQUAL "aaarch64")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS_AVX2}")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS_AVX2}")
    elseif (SFEM_VEC_SIZE STREQUAL "8")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS_AVX512}")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS_AVX512}")
    endif()
endif()

message(STATUS "CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")
message(STATUS "CMAKE_C_FLAGS: ${CMAKE_C_FLAGS}")


option(SFEM_TET4_CUDA "Use CUDA for TET4" OFF)
message(STATUS "SFEM_TET4_CUDA: ${SFEM_TET4_CUDA}")

option(SFEM_TET10_CUDA "Use CUDA for TET10" OFF)
message(STATUS "SFEM_TET10_CUDA: ${SFEM_TET10_CUDA}")

option(SFEM_TET10_WENO "Use WENO for TET10" ON)
message(STATUS "SFEM_TET10_WENO: ${SFEM_TET10_WENO}")

add_library(sfem ${SFEM_SOURCES} ${SFEM_HEADERS})

######################################################
# include

foreach(include ${SFEM_BUILD_INCLUDES})
    target_include_directories(sfem BEFORE
     PUBLIC $<BUILD_INTERFACE:${include}>)
endforeach()

target_include_directories(sfem BEFORE
     PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/external>)

target_include_directories(
    sfem BEFORE
    PUBLIC $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include> $<INSTALL_INTERFACE:include/kernels>)


target_include_directories(sfem PUBLIC ${SFEM_DEP_INCLUDES})

if(SFEM_ENABLE_HXTSORT)
    if(NOT SFEM_ENABLE_OPENMP)
        message(FATAL_ERROR "Option -DSFEM_ENABLE_HXTSORT=ON requires -DSFEM_ENABLE_OPENMP=ON")
    endif()
    target_include_directories(sfem PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/external/HXTSort")
    # target_compile_definitions(sfem PRIVATE "-D_XOPEN_SOURCE=600")
endif()

######################################################
# link

target_link_libraries(sfem PUBLIC ${SFEM_DEP_LIBRARIES})
# message(STATUS "SFEM_DEP_LIBRARIES: ${SFEM_DEP_LIBRARIES}")

######################################################
# properties

set_compiler_features(sfem)
set_target_properties(
    sfem PROPERTIES VERSION ${SFEM_VERSION}
    SOVERSION ${SFEM_VERSION_MAJOR})

set(SFEM_TARGETS sfem)

# option(SFEM_CUDA_ARCH "Use CUDA architecture" OFF)


if(SFEM_ENABLE_CUDA)
# set_target_properties(sfem PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
# https://gt3rs.medium.com/compile-with-nvcc-3566fbdfdbf
      set(CMAKE_CUDA_ARCHITECTURES ${SFEM_CUDA_ARCH})
      set_property(TARGET sfem PROPERTY CMAKE_CUDA_ARCHITECTURES ${SFEM_CUDA_ARCH})
      # get_property(${TEST_CUDA_ARCH} TARGET sfem PROPERTY CMAKE_CUDA_ARCHITECTURES)

      message(STATUS "SFEM_CUDA_ARCH: ${SFEM_CUDA_ARCH}")
endif()

######################################################
######################################################
# Executables

list(APPEND SFEM_DRIVER_MODULES . mesh examples mesh_functions postprocess sdf bench ale simulations contact) 

foreach(DRIVER_MODULE ${SFEM_DRIVER_MODULES})
    file(GLOB MODULE_C_DRIVERS CONFIGURE_DEPENDS "drivers/${DRIVER_MODULE}/*.c")
    file(GLOB MODULE_CPP_DRIVERS CONFIGURE_DEPENDS "drivers/${DRIVER_MODULE}/*.cpp")
    list(APPEND CPP_DRIVERS ${MODULE_CPP_DRIVERS})
    list(APPEND C_DRIVERS ${MODULE_C_DRIVERS})
endforeach()

set(DRIVERS "${C_DRIVERS};${CPP_DRIVERS}")

list(APPEND SFEM_EXCLUDED_EXEC "ssolve")

if(${SFEM_SCALAR_TYPE} STREQUAL "float32")
    # FIXME these executables need to be fixed when compiling with real_t=float32
    list (APPEND SFEM_EXCLUDED_EXEC assemble;assemble3;assemble4;cvfem_assemble;heat_equation;heat_equation;linear_elasticity_assemble;neohookean_assemble;taylor_hood_navier_stokes)
endif()

if(NOT ${SFEM_ENABLE_ISOLVER})
    message(WARNING "ISOLVER is disabled!")
     list (APPEND SFEM_EXCLUDED_EXEC heat_equation;heat_equation;linear_elasticity_assemble;neohookean_assemble;taylor_hood_navier_stokes)
endif()

if(SFEM_ENABLE_RESAMPLING)
    file(GLOB RESAMPLING_C_EXAMPLES CONFIGURE_DEPENDS "drivers/resampling/*.c")
    file(GLOB RESAMPLING_CXX_EXAMPLES CONFIGURE_DEPENDS "drivers/resampling/*.cpp")

    list(APPEND DRIVERS "${RESAMPLING_C_EXAMPLES}")
    list(APPEND DRIVERS "${RESAMPLING_CXX_EXAMPLES}")
endif()

if(SFEM_ENABLE_CUDA)
    file(GLOB CUDA_C_EXAMPLES CONFIGURE_DEPENDS "drivers/cuda/*.c")
    file(GLOB CUDA_CXX_EXAMPLES CONFIGURE_DEPENDS "drivers/cuda/*.cpp")

    list(APPEND DRIVERS "${CUDA_C_EXAMPLES}")
    list(APPEND DRIVERS "${CUDA_CXX_EXAMPLES}")
else()
    list(APPEND SFEM_EXCLUDED_EXEC "run_poisson_cuda;cuda_do_spmv;lapl_matrix_free;linear_elasticity_matrix_free;sym_spmv_bench")
endif()

if(NOT SFEM_ENABLE_CUSPARSE)
    # FIXME
    list(APPEND SFEM_EXCLUDED_EXEC "cuda_do_spmv")
endif()

foreach(DRIVER ${DRIVERS})
    get_filename_component(DRIVER_NAME ${DRIVER} NAME_WE)
    list(FIND SFEM_EXCLUDED_EXEC "${DRIVER_NAME}" LIST_INDEX)

    if (${LIST_INDEX} GREATER -1)
        message(STATUS "Skipped DRIVER_NAME: ${DRIVER_NAME}")
    else()
        add_executable(${DRIVER_NAME} ${DRIVER})
        target_link_libraries(${DRIVER_NAME} PRIVATE sfem)
        list(APPEND SFEM_TARGETS ${DRIVER_NAME})
    endif()
endforeach()

if(SFEM_ENABLE_TESTING)
    enable_testing()

    list(APPEND SFEM_UNIT_TESTS
        operators/ssquad4/tests
        operators/sshex8/tests
        operators/stencil/tests
        mesh/tests
        frontend/tests
        ssmg/tests
        amg/tests
    )

    foreach(UNIT_TEST ${SFEM_UNIT_TESTS})
    
    file(GLOB C_UNIT_TESTS CONFIGURE_DEPENDS "${UNIT_TEST}/*.c")
    file(GLOB CXX_UNIT_TESTS CONFIGURE_DEPENDS "${UNIT_TEST}/*.cpp")

    list(APPEND TEST_DRIVERS "${C_UNIT_TESTS}")
    list(APPEND TEST_DRIVERS "${CXX_UNIT_TESTS}")

    endforeach(UNIT_TEST)

    message(STATUS "TEST_DRIVERS=${TEST_DRIVERS}")

    foreach(TEST_DRIVER ${TEST_DRIVERS})
        get_filename_component(TEST_DRIVER_NAME ${TEST_DRIVER} NAME_WE)
        add_executable(${TEST_DRIVER_NAME} ${TEST_DRIVER})
        target_link_libraries(${TEST_DRIVER_NAME} PRIVATE sfem)
        add_test(NAME ${TEST_DRIVER_NAME} COMMAND ${TEST_DRIVER_NAME})
    endforeach(TEST_DRIVER ${TEST_DRIVERS})
endif()

# if(SFEM_ENABLE_BENCHMARK)
#     include(benchmarks/benchmarks.cmake)
# endif()

if(SFEM_ENABLE_PYTHON)
    add_subdirectory(python/bindings)
    set(SFEM_TARGETS ${SFEM_TARGETS} pysfem)
endif()

# message(STATUS "SFEM_TARGETS: ${SFEM_TARGETS}")

if(SFEM_SUBMODULES)
    target_link_libraries(sfem PUBLIC ${SFEM_SUBMODULES})
endif()

######################################################
######################################################
######################################################
# Installation
######################################################

install(
    TARGETS ${SFEM_TARGETS}
    EXPORT SFEMTargets
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib)

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/configuration/sfem_config.h.in
    ${CMAKE_BINARY_DIR}/sfem_config.h)

install(FILES ${CMAKE_BINARY_DIR}/sfem_config.h DESTINATION include)

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/configuration/sfem_config.py.in
    ${CMAKE_BINARY_DIR}/sfem_config.py)

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/configuration/sfem_config.sh.in
    ${CMAKE_BINARY_DIR}/sfem_config.sh)

install(FILES ${CMAKE_BINARY_DIR}/sfem_config.py DESTINATION scripts/sfem)
install(FILES ${CMAKE_BINARY_DIR}/sfem_config.sh DESTINATION workflows)

install(FILES ${SFEM_HEADERS} DESTINATION include)

install(
    EXPORT SFEMTargets
    FILE SFEMTargets.cmake
    NAMESPACE SFEM::
    DESTINATION lib/cmake)

######################################################
# Python scripts
######################################################

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/python/sfem DESTINATION scripts USE_SOURCE_PERMISSIONS)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/python/requirements.txt DESTINATION scripts)


######################################################
######################################################

# generate the config file that is includes the exports
configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/SFEMConfig.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/SFEMConfig.cmake"
    INSTALL_DESTINATION "lib/cmake")

# generate the version file for the config file
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/SFEMConfigVersion.cmake"
    VERSION "${SFEM_VERSION}"
    COMPATIBILITY AnyNewerVersion)

# install the configuration file
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/SFEMConfig.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/SFEMConfigVersion.cmake
  DESTINATION lib/cmake/)


if(NOT SFEM_ENABLE_SUBMODULE)
    include(cmake/PostInstall.cmake)
endif()

######################################################
######################################################
# Extra targets
######################################################

######################################################

if(TARGET cuda_do_spmv)
    add_custom_command(TARGET cuda_do_spmv  POST_BUILD
                       COMMAND ${CMAKE_COMMAND} -E copy
                           $<TARGET_FILE:cuda_do_spmv>
                           cuspmv
                       COMMENT "Copying 'cuda_do_spmv' executable to 'cuspmv'")
endif()


add_custom_command(TARGET do_spmv  POST_BUILD
                   COMMAND ${CMAKE_COMMAND} -E copy
                       $<TARGET_FILE:do_spmv>
                       spmv
                   COMMENT "Copying 'do_spmv' executable to 'spmv'")
