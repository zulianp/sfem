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
find_package(ISolver REQUIRED)

set(SFEM_MODULES
    algebra base matrix mesh pizzastack solver
    algebra/openmp
    amg
    operators 
    operators/beam2 
    operators/cvfem 
    # operators/kernels 
    operators/navier_stokes
    # operators/phase_field_for_fracture
    operators/tet10
    operators/tet4
    operators/tet4/vectorized
    operators/tri3
    operators/tri6
    operators/trishell3
    operators/quadshell4
    operators/cvfem_tri3
    operators/cvfem_tet4
    operators/cvfem_quad4
    operators/macro_tri3
    operators/macro_tet4
    operators/proteus_tet4
    operators/proteus_hex8
    operators/hierarchical
    operators/hex8
)

if(SFEM_ENABLE_RESAMPLING)
    list(APPEND SFEM_MODULES
        resampling
        resampling/tet10
    )
endif()

list(APPEND SFEM_MODULES frontend)

if(SFEM_ENABLE_METIS)
    list(APPEND SFEM_MODULES graphs)
endif()

if(SFEM_ENABLE_CUDA)
     list(APPEND SFEM_MODULES 
        algebra/cuda
        operators/cuda
        frontend/cuda
        operators/tet4/cuda
        operators/macro_tet4/cuda
        operators/tet10/cuda
        operators/proteus_tet4/cuda
        operators/hex8/cuda
        operators/proteus_hex8/cuda
        operators/hierarchical/cuda
        )
endif()

foreach(MODULE ${SFEM_MODULES})
    scan_directories(${CMAKE_SOURCE_DIR}/${MODULE} "." SFEM_BUILD_INCLUDES
       SFEM_HEADERS SFEM_SOURCES)

    if(SFEM_ENABLE_CUDA)
        file(GLOB SFEM_MODULE_KERNEL_SOURCES CONFIGURE_DEPENDS "${CMAKE_SOURCE_DIR}/${MODULE}/*.cu")
        # message(STATUS "${MODULE} : ${SFEM_MODULE_KERNEL_SOURCES}")
    endif()

    if(SFEM_MODULE_KERNEL_SOURCES)
        list(APPEND SFEM_KERNEL_SOURCES "${SFEM_MODULE_KERNEL_SOURCES}")
    endif()
endforeach(MODULE)

if(SFEM_ENABLE_CUDA)
    list(APPEND SFEM_SOURCES "${SFEM_KERNEL_SOURCES}")
endif()

# message(STATUS "SFEM_BUILD_INCLUDES: ${SFEM_BUILD_INCLUDES}")
# message(STATUS "SFEM_HEADERS: ${SFEM_HEADERS}") 
# message(STATUS "SFEM_SOURCES: ${SFEM_SOURCES}")
# message(STATUS "SFEM_KERNEL_SOURCES: ${SFEM_KERNEL_SOURCES}")
message(STATUS "CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")
message(STATUS "CMAKE_C_FLAGS: ${CMAKE_C_FLAGS}")


add_library(sfem ${SFEM_SOURCES} ${SFEM_HEADERS})

######################################################
# include

foreach(include ${SFEM_BUILD_INCLUDES})
    target_include_directories(sfem BEFORE
     PUBLIC $<BUILD_INTERFACE:${include}>)
endforeach()

target_include_directories(sfem BEFORE
     PUBLIC $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/external>)

target_include_directories(
    sfem BEFORE
    PUBLIC $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include> $<INSTALL_INTERFACE:include/kernels>)


target_include_directories(sfem PUBLIC ${SFEM_DEP_INCLUDES})

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

if(SFEM_ENABLE_CUDA)
# set_target_properties(sfem PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
# https://gt3rs.medium.com/compile-with-nvcc-3566fbdfdbf
      set_property(TARGET sfem PROPERTY CUDA_ARCHITECTURES 60)
endif()

######################################################
######################################################
# Executables

file(GLOB C_DRIVERS CONFIGURE_DEPENDS "drivers/*.c")
file(GLOB CPP_DRIVERS CONFIGURE_DEPENDS "drivers/*.cpp")
# file(GLOB CPP_EXAMPLES CONFIGURE_DEPENDS "examples/*.cpp")
file(GLOB C_EXAMPLES CONFIGURE_DEPENDS "examples/*.c")

set(DRIVERS "${C_DRIVERS};${CPP_DRIVERS};${C_EXAMPLES}")

# list(APPEND SFEM_EXCLUDED_EXEC "grad_and_project;heat_equation;integrate_divergence;partition_mesh_based_on_operator;taylor_hood_navier_stokes;ssolve")

list(APPEND SFEM_EXCLUDED_EXEC "ssolve")

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
    list(APPEND SFEM_EXCLUDED_EXEC "run_poisson_cuda;cuda_do_spmv;lapl_matrix_free;linear_elasticity_matrix_free")
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

# if(SFEM_ENABLE_TESTING)
#     include(tests/tests.cmake)
# endif()

# if(SFEM_ENABLE_BENCHMARK)
#     include(benchmarks/benchmarks.cmake)
# endif()

if(SFEM_ENABLE_PYTHON)
    add_subdirectory(python/bindings)
    set(SFEM_TARGETS ${SFEM_TARGETS} pysfem)
endif()

# message(STATUS "SFEM_TARGETS: ${SFEM_TARGETS}")

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

include(cmake/PostInstall.cmake)

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
