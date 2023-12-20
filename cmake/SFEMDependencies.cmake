# SFEMDependencies.cmake

if(SFEM_ENABLE_CUDA)
    enable_language(CUDA)

    if(NOT DEFINED CMAKE_CUDA_STANDARD)
        set(CMAKE_CUDA_STANDARD 17)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    endif()

endif()

# ##############################################################################

if(SFEM_ENABLE_LAPACK OR TPL_ENABLE_LAPACK)
    if(TPL_LAPACK_LIBRARIES)
        list(APPEND SFEM_DEP_LIBRARIES ${TPL_LAPACK_LIBRARIES})
        set(SFEM_HAVE_LAPACK TRUE)
    else()
        find_package(LAPACK)
        if(LAPACK_FOUND)
            set(SFEM_HAVE_LAPACK TRUE)
            list(APPEND SFEM_DEP_LIBRARIES ${LAPACK_LIBRARIES})
        endif()
    endif()
endif()

# ##############################################################################

find_package(MPIExtended REQUIRED)

if(MPI_FOUND)
    set(SFEM_HAVE_MPI TRUE)

    if(MPI_C_INCLUDE_PATH)
        set(SFEM_DEP_INCLUDES
            "${SFEM_DEP_INCLUDES};${MPI_C_INCLUDE_PATH}")
    endif()

    if(MPI_CXX_INCLUDE_PATH)
        set(SFEM_DEP_INCLUDES
            "${SFEM_DEP_INCLUDES};${MPI_CXX_INCLUDE_PATH}")
    endif()

    if(MPI_LIBRARIES)
        set(SFEM_DEP_LIBRARIES
            "${SFEM_DEP_LIBRARIES};${MPI_LIBRARIES}")
    endif()

    if(MPI_C_LIBRARIES)
        set(SFEM_DEP_LIBRARIES
            "${SFEM_DEP_LIBRARIES};${MPI_C_LIBRARIES}")
    endif()

    if(MPI_CXX_LIBRARIES)
        set(SFEM_DEP_LIBRARIES
            "${SFEM_DEP_LIBRARIES};${MPI_CXX_LIBRARIES}")
    endif()
else()
    message(
        FATAL_ERROR
            "We should never end up here, because find_package above is REQUIRED"
    )
endif()

# ##############################################################################

find_package(Doxygen QUIET)

if(DOXYGEN_FOUND)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.txt ${CMAKE_BINARY_DIR}
                   @ONLY IMMEDIATE)
    add_custom_target(
        docs
        COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_BINARY_DIR}/Doxyfile.txt
        SOURCES ${CMAKE_BINARY_DIR}/Doxyfile.txt)

endif()

# ##############################################################################

if(CMAKE_BUILD_TYPE MATCHES "[Cc][Oo][Vv][Ee][Rr][Aa][Gg][Ee]")
    include(cmake/CodeCoverage.cmake)
    add_codecov(sfem_coverage sfem_test coverage)
endif()
