# FindMatrixIO.cmake

if(NOT MatrixIO_DIR)

find_path(
    MatrixIO_DIR  matrixio_array.h
    HINTS ${CMAKE_SOURCE_DIR}/../matrix.io
    )

endif()

if(NOT MatrixIO_DIR)
	message(FATAL_ERROR "MatrixIO not found!")
endif()

scan_directories(${MatrixIO_DIR} "." SFEM_BUILD_INCLUDES
                 SFEM_HEADERS SFEM_SOURCES)

# set(SFEM_BUILD_INCLUDES
#     ${SFEM_BUILD_INCLUDES}
#     PARENT_SCOPE)

# set(SFEM_HEADERS
#     ${SFEM_HEADERS}
#     PARENT_SCOPE)

# set(SFEM_SOURCES
#     ${SFEM_SOURCES}
#     PARENT_SCOPE)