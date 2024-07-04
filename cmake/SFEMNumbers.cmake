# ##############################################################################
# XSDK_PRECISION
# ##############################################################################

option(SFEM_ENABLE_CUSTOM_NUMBERS "Enable custom numbers support. (this option will be removed in the future)" ON)

set(SFEM_REAL_TYPE          "float64"   CACHE STRING "Set SFEM real_t type. Used for solution vectors.")
set(SFEM_SCALAR_TYPE        "float64"   CACHE STRING "Set SFEM scalar_t type. Used for local kernel computations")
set(SFEM_GEOM_TYPE          "float32"   CACHE STRING "Set SFEM geom_t type")
set(SFEM_JACOBIAN_CPU_TYPE  "float32"   CACHE STRING "Set SFEM jacobian_t type")
set(SFEM_JACOBIAN_GPU_TYPE  "half"      CACHE STRING "Set SFEM cu_jacobian_t type")
set(SFEM_ACCUMULATOR_TYPE   "float64"   CACHE STRING "Set SFEM accumulator_t type")

set(SFEM_IDX_TYPE           "int32"     CACHE STRING "Set SFEM idx_t type")
set(SFEM_COUNT_TYPE         "int64"     CACHE STRING "Set SFEM count_t type")
set(SFEM_ELEMENT_IDX_TYPE   "int32"     CACHE STRING "Set SFEM element_idx_t type")
set(SFEM_LOCAL_IDX_TYPE     "int16"     CACHE STRING "Set SFEM local_idx_t type")

function(sfem_simd_vector_size type vec_size)
    if("${type}" STREQUAL "float32")
        set(${vec_size} 8 PARENT_SCOPE)
    elseif("${type}" STREQUAL "float64")
        set(${vec_size} 4 PARENT_SCOPE)
    else()
        message(FATAL_ERROR "Not simd vector size type for `${type}`!")
    endif()
endfunction()

sfem_simd_vector_size(${SFEM_SCALAR_TYPE} SFEM_VEC_SIZE_DEFAUT)
set(SFEM_VEC_SIZE  ${SFEM_VEC_SIZE_DEFAUT}   CACHE STRING "Set number of simd lanes for vec_t type")

function(sfem_c_type type c_type)
    if("${type}" STREQUAL "int16")
        set(${c_type} "int16_t " PARENT_SCOPE)
    elseif("${type}" STREQUAL "int32")
        set(${c_type} "int32_t " PARENT_SCOPE)
    elseif("${type}" STREQUAL "int64")
        set(${c_type} "int64_t " PARENT_SCOPE)
    # elseif("${type}" STREQUAL "float16")
    #     set(${c_type} "NO direct MPI Support" PARENT_SCOPE)
    elseif("${type}" STREQUAL "float32")
        set(${c_type} "float" PARENT_SCOPE)
    elseif("${type}" STREQUAL "float64")
        set(${c_type} "double" PARENT_SCOPE)
    else()
        message(FATAL_ERROR "Not C type for `${type}`!")
    endif()
endfunction()

function(sfem_mpi_type type mpi_type)
    if("${type}" STREQUAL "int16")
        set(${mpi_type} "MPI_INT16_T " PARENT_SCOPE)
    elseif("${type}" STREQUAL "int32")
        set(${mpi_type} "MPI_INT32_T " PARENT_SCOPE)
    elseif("${type}" STREQUAL "int64")
        set(${mpi_type} "MPI_INT64_T " PARENT_SCOPE)
    # elseif("${type}" STREQUAL "float16")
    #     set(${mpi_type} "NO direct MPI Support" PARENT_SCOPE)
    elseif("${type}" STREQUAL "float32")
        set(${mpi_type} "MPI_FLOAT" PARENT_SCOPE)
    elseif("${type}" STREQUAL "float64")
        set(${mpi_type} "MPI_DOUBLE" PARENT_SCOPE)
    else()
        message(FATAL_ERROR "Not MPI type for `${type}`!")
    endif()
endfunction()

function(sfem_cusparse_type type cusparse_type)
    # if("${type}" STREQUAL "int16")
    #     set(${cusparse_type} "int16_t " PARENT_SCOPE)
    # else
    if("${type}" STREQUAL "int32")
        set(${cusparse_type} "CUSPARSE_INDEX_32I " PARENT_SCOPE)
    elseif("${type}" STREQUAL "int64")
        set(${cusparse_type} "CUSPARSE_INDEX_64I " PARENT_SCOPE)
    elseif("${type}" STREQUAL "float16")
        set(${cusparse_type} "CUDA_R_16F" PARENT_SCOPE)
    elseif("${type}" STREQUAL "float32")
        set(${cusparse_type} "CUDA_R_32F" PARENT_SCOPE)
    elseif("${type}" STREQUAL "float64")
        set(${cusparse_type} "CUDA_R_64F" PARENT_SCOPE)
    else()
        message(FATAL_ERROR "Not CuSparse type for `${type}`!")
    endif()
endfunction()

function(sfem_print_type type print_type)
    if("${type}" STREQUAL "int16")
        set(${print_type} "hd" PARENT_SCOPE)
    elseif("${type}" STREQUAL "int32")
        set(${print_type} "d" PARENT_SCOPE)
    elseif("${type}" STREQUAL "int64")
        set(${print_type} "ld" PARENT_SCOPE)
    # elseif("${type}" STREQUAL "float16")
    #     set(${print_type} "NO direct MPI Support" PARENT_SCOPE)
    elseif("${type}" STREQUAL "float32")
        set(${print_type} "f" PARENT_SCOPE)
    elseif("${type}" STREQUAL "float64")
        set(${print_type} "g" PARENT_SCOPE)
    else()
        message(FATAL_ERROR "No print type for `${type}`!")
    endif()
endfunction()

# C types
sfem_c_type(${SFEM_REAL_TYPE} SFEM_REAL_C_TYPE)
sfem_c_type(${SFEM_SCALAR_TYPE} SFEM_SCALAR_C_TYPE)
sfem_c_type(${SFEM_GEOM_TYPE} SFEM_GEOM_C_TYPE)
sfem_c_type(${SFEM_JACOBIAN_CPU_TYPE} SFEM_JACOBIAN_CPU_C_TYPE)
sfem_c_type(${SFEM_ACCUMULATOR_TYPE} SFEM_ACCUMULATOR_C_TYPE)
sfem_c_type(${SFEM_IDX_TYPE} SFEM_IDX_C_TYPE)
sfem_c_type(${SFEM_COUNT_TYPE} SFEM_COUNT_C_TYPE)
sfem_c_type(${SFEM_ELEMENT_IDX_TYPE} SFEM_ELEMENT_IDX_C_TYPE)
sfem_c_type(${SFEM_LOCAL_IDX_TYPE} SFEM_LOCAL_IDX_C_TYPE)

# CuSparse types
sfem_cusparse_type(${SFEM_REAL_TYPE} SFEM_REAL_CUSPARSE_TYPE)
sfem_cusparse_type(${SFEM_SCALAR_TYPE} SFEM_SCALAR_CUSPARSE_TYPE)
sfem_cusparse_type(${SFEM_GEOM_TYPE} SFEM_GEOM_CUSPARSE_TYPE)
sfem_cusparse_type(${SFEM_JACOBIAN_CPU_TYPE} SFEM_JACOBIAN_CPU_CUSPARSE_TYPE)
sfem_cusparse_type(${SFEM_IDX_TYPE} SFEM_IDX_CUSPARSE_TYPE)
sfem_cusparse_type(${SFEM_COUNT_TYPE} SFEM_COUNT_CUSPARSE_TYPE)
sfem_cusparse_type(${SFEM_ELEMENT_IDX_TYPE} SFEM_ELEMENT_IDX_CUSPARSE_TYPE)
# sfem_cusparse_type(${SFEM_LOCAL_IDX_TYPE} SFEM_LOCAL_IDX_CUSPARSE_TYPE)

# MPI types
sfem_mpi_type(${SFEM_REAL_TYPE} SFEM_REAL_MPI_TYPE)
sfem_mpi_type(${SFEM_SCALAR_TYPE} SFEM_SCALAR_MPI_TYPE)
sfem_mpi_type(${SFEM_GEOM_TYPE} SFEM_GEOM_MPI_TYPE)
sfem_mpi_type(${SFEM_JACOBIAN_CPU_TYPE} SFEM_JACOBIAN_CPU_MPI_TYPE)
sfem_mpi_type(${SFEM_IDX_TYPE} SFEM_IDX_MPI_TYPE)
sfem_mpi_type(${SFEM_COUNT_TYPE} SFEM_COUNT_MPI_TYPE)
sfem_mpi_type(${SFEM_ELEMENT_IDX_TYPE} SFEM_ELEMENT_IDX_MPI_TYPE)
sfem_mpi_type(${SFEM_LOCAL_IDX_TYPE} SFEM_LOCAL_IDX_MPI_TYPE)

# Print format
sfem_print_type(${SFEM_REAL_TYPE} SFEM_REAL_PRINT_TYPE)
sfem_print_type(${SFEM_SCALAR_TYPE} SFEM_SCALAR_PRINT_TYPE)
sfem_print_type(${SFEM_GEOM_TYPE} SFEM_GEOM_PRINT_TYPE)
sfem_print_type(${SFEM_JACOBIAN_CPU_TYPE} SFEM_JACOBIAN_CPU_PRINT_TYPE)
sfem_print_type(${SFEM_IDX_TYPE} SFEM_IDX_PRINT_TYPE)
sfem_print_type(${SFEM_COUNT_TYPE} SFEM_COUNT_PRINT_TYPE)
sfem_print_type(${SFEM_ELEMENT_IDX_TYPE} SFEM_ELEMENT_IDX_PRINT_TYPE)
sfem_print_type(${SFEM_LOCAL_IDX_TYPE} SFEM_LOCAL_IDX_PRINT_TYPE)

message(STATUS 
    "--------------------------------------------------------------------------------------\n"
    "\nSFEM Numbers\n"
    "--------------------------------------------------------------------------------------\n"
    "type\t\tC\t\tId\tMPI\t\tprintf\t(CMake option)\n"
    "--------------------------------------------------------------------------------------\n"
    "real_t\t\t${SFEM_REAL_C_TYPE}\t\t${SFEM_REAL_TYPE}\t${SFEM_REAL_MPI_TYPE}\t${SFEM_REAL_PRINT_TYPE}\t(SFEM_REAL_TYPE)\n"
    "scalar_t\t${SFEM_SCALAR_C_TYPE}\t\t${SFEM_SCALAR_TYPE}\t${SFEM_SCALAR_MPI_TYPE}\t${SFEM_SCALAR_PRINT_TYPE}\t(SFEM_SCALAR_TYPE)\n"
    "geom_t\t\t${SFEM_GEOM_C_TYPE}\t\t${SFEM_GEOM_TYPE}\t${SFEM_GEOM_MPI_TYPE}\t${SFEM_GEOM_PRINT_TYPE}\t(SFEM_GEOM_TYPE)\n"
    "jacobian_t\t${SFEM_JACOBIAN_CPU_C_TYPE}\t\t${SFEM_JACOBIAN_CPU_TYPE}\t${SFEM_JACOBIAN_CPU_MPI_TYPE}\t${SFEM_JACOBIAN_CPU_PRINT_TYPE}\t(SFEM_JACOBIAN_CPU_TYPE)\n"
    "cu_jacobian_t\t${SFEM_JACOBIAN_GPU_TYPE}\t\t-\t-\t\t-\t(SFEM_JACOBIAN_GPU_TYPE)\n"
    "idx_t\t\t${SFEM_IDX_C_TYPE}\t${SFEM_IDX_TYPE}\t${SFEM_IDX_MPI_TYPE}\t${SFEM_IDX_PRINT_TYPE}\t(SFEM_IDX_TYPE)\n"
    "count_t\t\t${SFEM_COUNT_C_TYPE}\t${SFEM_COUNT_TYPE}\t${SFEM_COUNT_MPI_TYPE}\t${SFEM_COUNT_PRINT_TYPE}\t(SFEM_COUNT_TYPE)\n"
    "element_idx_t\t${SFEM_ELEMENT_IDX_C_TYPE}\t${SFEM_ELEMENT_IDX_TYPE}\t${SFEM_ELEMENT_IDX_MPI_TYPE}\t${SFEM_ELEMENT_IDX_PRINT_TYPE}\t(SFEM_ELEMENT_IDX_TYPE)\n"
    "local_idx_t\t${SFEM_LOCAL_IDX_C_TYPE}\t${SFEM_LOCAL_IDX_TYPE}\t${SFEM_LOCAL_IDX_MPI_TYPE}\t${SFEM_LOCAL_IDX_PRINT_TYPE}\t(SFEM_LOCAL_IDX_TYPE)\n"
    "--------------------------------------------------------------------------------------\n"
    "SFEM_VEC_SIZE=${SFEM_VEC_SIZE}\n"
    "--------------------------------------------------------------------------------------\n"

)

# if(NOT XSDK_PRECISION OR USE_XSDK_DEFAULTS)
#     set(XSDK_PRECISION "DOUBLE")
# endif()

# string(COMPARE EQUAL ${XSDK_PRECISION} "DOUBLE" SFEM_HAVE_DOUBLE_PRECISION)
# string(COMPARE EQUAL ${XSDK_PRECISION} "SINGLE" SFEM_HAVE_SINGLE_PRECISION)
# string(COMPARE EQUAL ${XSDK_PRECISION} "QUAD" SFEM_HAVE_QUAD_PRECISION)

# # ##############################################################################
# # XSDK_INDEX_SIZE
# # ##############################################################################


# if(NOT SFEM_INDEX_BITSIZE)
#     set(SFEM_INDEX_BITSIZE 32 CACHE STRING "Choice of idx_t size between 32 or 64 bits" FORCE)
# endif()

# if(NOT SFEM_COUNT_BITSIZE)
#     set(SFEM_COUNT_BITSIZE 32 CACHE STRING "Choice of count_t size between 32 or 64 bits" FORCE)
# endif()


# if(USE_XSDK_DEFAULTS)
#     set(XSDK_INDEX_SIZE 32)
#     set(SFEM_INDEX_BITSIZE 32)
# else()
#     if(XSDK_INDEX_SIZE)
#         set(SFEM_INDEX_BITSIZE ${XSDK_INDEX_SIZE})
#     elseif(NOT SFEM_INDEX_BITSIZE)
#         set(XSDK_INDEX_SIZE 64)
#         set(SFEM_INDEX_BITSIZE 64)
#     endif()
# endif()