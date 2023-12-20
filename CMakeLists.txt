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
include(cmake/SFEMBuildTypes.cmake)
include(cmake/SFEMCMakeFunctions.cmake)
include(cmake/SFEMDependencies.cmake)

# ######################################################
# ######################################################
# Library
# ######################################################

find_package(MatrixIO)

set(SFEM_MODULES
    algebra base matrix mesh pizzastack resampling solver
    operators 
        operators/beam2 
        operators/cvfem 
        operators/kernels 
        operators/navier_stokes
        operators/phase_field_for_fracture
        operators/tet10
        operators/tet4
        operators/tri3
        operators/tri6
        operators/trishell3
            )

if(SFEM_ENABLE_METIS)
    list(APPEND SFEM_MODULES graphs)
endif()

foreach(MODULE ${SFEM_MODULES})
    scan_directories(${CMAKE_SOURCE_DIR}/${MODULE} "." SFEM_BUILD_INCLUDES
                     SFEM_HEADERS SFEM_SOURCES)
endforeach(MODULE)

# message(STATUS "SFEM_BUILD_INCLUDES: ${SFEM_BUILD_INCLUDES}")
# message(STATUS "SFEM_HEADERS: ${SFEM_HEADERS}") 
# message(STATUS "SFEM_SOURCES: ${SFEM_SOURCES}")

add_library(sfem ${SFEM_SOURCES} ${SFEM_HEADERS})

# ######################################################
# include

foreach(include ${SFEM_BUILD_INCLUDES})
    target_include_directories(sfem BEFORE
                               PUBLIC $<BUILD_INTERFACE:${include}>)
endforeach()

target_include_directories(
    sfem BEFORE
    PUBLIC $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}>
           $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
           $<INSTALL_INTERFACE:include> $<INSTALL_INTERFACE:include/kernels>)

target_include_directories(sfem PUBLIC ${SFEM_DEP_INCLUDES})

# ######################################################
# link

target_link_libraries(sfem PUBLIC ${SFEM_DEP_LIBRARIES})

# ######################################################
# properties

set_compiler_features(sfem)
set_target_properties(
    sfem PROPERTIES VERSION ${SFEM_VERSION}
                             SOVERSION ${SFEM_VERSION_MAJOR})

# ######################################################
# ######################################################
# Executables

file(GLOB C_DRIVERS CONFIGURE_DEPENDS "drivers/*.c")
file(GLOB CPP_DRIVERS CONFIGURE_DEPENDS "drivers/*.cpp")

set(DRIVERS "${C_DRIVERS};${CPP_DRIVERS}")

list(APPEND SFEM_EXCLUDED_EXEC "grad_and_project;heat_equation;integrate_divergence;partition_mesh_based_on_operator;taylor_hood_navier_stokes;ssolve")

foreach(DRIVER ${DRIVERS})
    get_filename_component(DRIVER_NAME ${DRIVER} NAME_WE)
    list(FIND SFEM_EXCLUDED_EXEC "${DRIVER_NAME}" LIST_INDEX)
    
    if (${LIST_INDEX} GREATER -1)
    else()
        add_executable(${DRIVER_NAME} ${DRIVER})
        target_link_libraries(${DRIVER_NAME} PRIVATE sfem)
    endif()
endforeach()

# if(SFEM_ENABLE_TESTING)
#     include(tests/tests.cmake)
# endif()

# if(SFEM_ENABLE_BENCHMARK)
#     include(benchmarks/benchmarks.cmake)
# endif()

# ######################################################
# ######################################################
# ######################################################
# Installation
# ######################################################

install(
    TARGETS sfem
    EXPORT SFEMTargets
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib)

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/configuration/sfem_config.hpp.in
    ${CMAKE_BINARY_DIR}/sfem_config.hpp)

install(FILES ${CMAKE_BINARY_DIR}/sfem_config.hpp DESTINATION include)
install(FILES ${SFEM_HEADERS} DESTINATION include)

install(
    EXPORT SFEMTargets
    FILE SFEMTargets.cmake
    NAMESPACE SFEM::
    DESTINATION lib/cmake)

######################################################
# Python scripts
######################################################

file(GLOB PYTHON_SCRIPTS CONFIGURE_DEPENDS
     "${CMAKE_SOURCE_DIR}/python/mesh/*.py")
install(FILES ${PYTHON_SCRIPTS} DESTINATION scripts/mesh)

file(GLOB PYTHON_SCRIPTS CONFIGURE_DEPENDS
     "${CMAKE_SOURCE_DIR}/python/sdf/*.py")
install(FILES ${PYTHON_SCRIPTS} DESTINATION scripts/sdf)

file(GLOB PYTHON_SCRIPTS CONFIGURE_DEPENDS
     "${CMAKE_SOURCE_DIR}/python/grid/*.py")
install(FILES ${PYTHON_SCRIPTS} DESTINATION scripts/grid)

file(GLOB PYTHON_SCRIPTS CONFIGURE_DEPENDS
     "${CMAKE_SOURCE_DIR}/python/algebra/*.py")
install(FILES ${PYTHON_SCRIPTS} DESTINATION scripts/algebra)

######################################################

# ######################################################

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

# ######################################################
# ######################################################
# Extra targets
# ######################################################

# ######################################################
