# SFEMDependencies.cmake

if(SFEM_ENABLE_CUDA)
    enable_language(CUDA)
    if(NOT DEFINED CMAKE_CUDA_STANDARD)
        set(CMAKE_CUDA_STANDARD 17)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    endif()
endif()

if(SFEM_ENABLE_OPENMP)
    if(OPENMP_DIR)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Xpreprocessor -fopenmp")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Xpreprocessor -fopenmp")
        list(APPEND SFEM_DEP_INCLUDES "${OPENMP_DIR}/include")
        list(APPEND SFEM_DEP_LIBRARIES "-L${OPENMP_DIR}/lib -lomp" )
    else()
        find_package(OpenMP REQUIRED)
        if(OpenMP_FOUND)
            message(STATUS "OpenMP: ${OpenMP_INCLUDES}")
            set(SFEM_DEP_LIBRARIES "${SFEM_DEP_LIBRARIES};OpenMP::OpenMP_C;OpenMP::OpenMP_CXX")
            set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        endif()
    endif()
endif()

# ##############################################################################

if(SFEM_ENABLE_METIS)
  find_package(METIS REQUIRED)
  if(METIS_FOUND)
    list(APPEND SFEM_BUILD_INCLUDES ${METIS_INCLUDES})
    list(APPEND SFEM_DEP_LIBRARIES ${METIS_LIBRARIES})
    set(SFEM_ENABLE_METIS ON)
  else()
    message(FATAL_ERROR "[Warning] Metis not found")
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

if(SFEM_ENABLE_MPI)
    find_package(MPI REQUIRED)

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

# ##############################################################################

if(SFEM_ENABLE_CUDA)
    enable_language(CUDA)

    if(NOT DEFINED CMAKE_CUDA_STANDARD)
      set(CMAKE_CUDA_STANDARD 17)
      set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    endif()

    set(SFEM_DEP_INCLUDES "${SFEM_DEP_INCLUDES};${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
    #set(SFEM_DEP_LIBRARIES "${SFEM_DEP_LIBRARIES};${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES}")

    include(CheckLanguage)
    check_language(CUDA)

    find_package(CUDAToolkit REQUIRED)

    list(APPEND SFEM_DEP_LIBRARIES "CUDA::cudart")

    set(_SFEM_CUDA_MODULES "CUDA::cusparse;CUDA::cublas;CUDA::nvToolsExt")
    set(SFEM_ENABLE_CUBLAS TRUE)
    set(SFEM_ENABLE_CUSPARSE TRUE)

    set(SFEM_CUDA_MATH_LIBS_FOUND FALSE)

    foreach(CUDA_MODULE ${_SFEM_CUDA_MODULES})
        if(TARGET ${CUDA_MODULE})
            list(APPEND SFEM_DEP_LIBRARIES "${CUDA_MODULE}")
            set(SFEM_CUDA_MATH_LIBS_FOUND TRUE)
        else()
            message(WARNING "[Warning] CUDAToolkit does not have module ${CUDA_MODULE} in a standard location!")
            
        endif()
    endforeach()

    if(NOT SFEM_CUDA_MATH_LIBS_FOUND)
        message("Trying with: CRAY_CUDATOOLKIT_POST_LINK_OPTS=$ENV{CRAY_CUDATOOLKIT_POST_LINK_OPTS}")
        message("Trying with: CRAY_CUDATOOLKIT_INCLUDE_OPTS=$ENV{CRAY_CUDATOOLKIT_INCLUDE_OPTS}")
        list(APPEND SFEM_DEP_LIBRARIES "$ENV{CRAY_CUDATOOLKIT_POST_LINK_OPTS} -lcublas -lcusparse")
        include_directories($ENV{CRAY_CUDATOOLKIT_INCLUDE_OPTS})
    endif()

    #https://github.com/NVIDIA/thrust/blob/main/thrust/cmake/README.md
    find_package(Thrust CONFIG)
    if(Thrust_FOUND)
        thrust_create_target(Thrust)
        list(APPEND SFEM_DEP_LIBRARIES Thrust)
    else()
        message(WARNING "Thrust not found!")
    endif()
endif()



if(SFEM_ENABLE_RYAML)
    set(RYML_REPO_URL https://github.com/biojppm/rapidyaml CACHE STRING "")
    set(RYML_BRANCH_NAME master CACHE STRING "")
    include(FetchContent)
    FetchContent_Declare(ryml
        GIT_REPOSITORY ${RYML_REPO_URL}
        GIT_TAG ${RYML_BRANCH_NAME}
        GIT_SHALLOW FALSE  # ensure submodules are checked out
    )
    FetchContent_MakeAvailable(ryml)
    list(APPEND SFEM_SUBMODULES ryml::ryml)
endif()

if(WIN32)
    set(GLOB_REPO_URL https://github.com/p-ranav/glob.git CACHE STRING "")
    set(GLOB_BRANCH_NAME master CACHE STRING "")
    include(FetchContent)
    FetchContent_Declare(Glob
        GIT_REPOSITORY ${GLOB_REPO_URL}
        GIT_TAG ${GLOB_BRANCH_NAME}
        GIT_SHALLOW FALSE  # ensure submodules are checked out
    )
    FetchContent_MakeAvailable(Glob)
    list(APPEND SFEM_SUBMODULES Glob)
endif()

if(SFEM_ENABLE_BLAS)
    if(APPLE)
        # Add Accelerate framework for macOS BLAS/LAPACK
        find_library(ACCELERATE_FRAMEWORK Accelerate REQUIRED)
        list(APPEND SFEM_DEP_LIBRARIES ${ACCELERATE_FRAMEWORK})
    else()
        find_package(BLAS REQUIRED)
    endif()
endif()

if(SFEM_ENABLE_LAPACK)
    find_package(LAPACK REQUIRED)
    list(APPEND SFEM_DEP_LIBRARIES  LAPACK::LAPACK)
endif()

# if(SFEM_ENABLE_HIP)
# # TODO
# endif()

if(SFEM_ENABLE_AVX512_SORT)
	include_directories("${CMAKE_CURRENT_SOURCE_DIR}/external/x86-simd-sort/src") 
endif()

