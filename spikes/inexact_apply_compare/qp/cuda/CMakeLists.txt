# The qp formats on the GPU: kernel-shape tests, bench_qp_cuda and newton_qp_cuda.  Device code is
# compiled without fast math, so the set-up rounds as the CPU's does; the register and spill report
# (-Xptxas -v) stays in the build log.
find_package(CUDAToolkit REQUIRED)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(_qp_cuda_options $<$<COMPILE_LANGUAGE:CUDA>:-O3 --default-stream per-thread -lineinfo -Xptxas=-v>)

# Device code apart from spike_common: nvcc splits a definition's value at its commas, and
# KERNEL_SHAPE_ORDER is a comma-separated list.
add_library(qbsr_kernels OBJECT qbsr_kernels.cu)
target_compile_options(qbsr_kernels PRIVATE ${_qp_cuda_options})

# Quantized BSR: which layout the device format takes.
add_executable(qbsr_shape qbsr_shape.cpp $<TARGET_OBJECTS:qbsr_kernels>)
target_link_libraries(qbsr_shape PRIVATE spike_common "${SPIKE_SFEM_LIBRARY}" ${SPIKE_SMESH_LIBRARY}
    CUDA::cudart CUDA::cusparse)
target_include_directories(qbsr_shape PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}" ${SPIKE_SFEM_SOURCE_INCLUDES}
    ${SPIKE_SFEM_DEP_INCLUDES})
target_compile_options(qbsr_shape PRIVATE ${_qp_cuda_options})

# The stores' device kernels for this build's element, from the generated stages (qp_gen/).
add_library(qp_device OBJECT qp_device.cu)
target_include_directories(qp_device PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}" "${_qp_gen}"
    "${SPIKE_SFEM}/src/frontend/ops/generated/cuda")
target_compile_options(qp_device PRIVATE ${_qp_cuda_options} $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fopenmp>)

# bench_qp's problem and references, every operator on the device; host stores of 32 lanes for
# the set-up gate.
get_target_property(_qp_bench_defs bench_qp COMPILE_DEFINITIONS)
get_target_property(_qp_bench_incs bench_qp INCLUDE_DIRECTORIES)
add_executable(bench_qp_cuda bench_qp_cuda.cpp $<TARGET_OBJECTS:qp_device> $<TARGET_OBJECTS:qbsr_kernels>
    $<TARGET_OBJECTS:spike_kernels>)
target_link_libraries(bench_qp_cuda PRIVATE spike_common "${SPIKE_SFEM_LIBRARY}" ${SPIKE_SMESH_LIBRARY} CUDA::cudart
    CUDA::cusparse)
target_include_directories(bench_qp_cuda PRIVATE ${_qp_bench_incs} "${CMAKE_CURRENT_SOURCE_DIR}"
    "${SPIKE_SFEM}/src/frontend/ops/prony" ${SPIKE_SFEM_SOURCE_INCLUDES} ${SPIKE_SFEM_DEP_INCLUDES})
target_compile_definitions(bench_qp_cuda PRIVATE ${_qp_bench_defs} LANE_VS=32)

# Newton on the device with a compressed format as the Krylov operator (newton_qp's flow).
if(TARGET newton_qp)
    get_target_property(_qp_newton_defs newton_qp COMPILE_DEFINITIONS)
    get_target_property(_qp_newton_incs newton_qp INCLUDE_DIRECTORIES)
    get_target_property(_qp_newton_libs newton_qp LINK_LIBRARIES)
    add_executable(newton_qp_cuda newton_qp_cuda.exe.cpp $<TARGET_OBJECTS:qp_device> $<TARGET_OBJECTS:qbsr_kernels>)
    target_link_libraries(newton_qp_cuda PRIVATE ${_qp_newton_libs} CUDA::cudart)
    target_include_directories(newton_qp_cuda PRIVATE ${_qp_newton_incs} "${CMAKE_CURRENT_SOURCE_DIR}")
    target_compile_definitions(newton_qp_cuda PRIVATE ${_qp_newton_defs})
endif()
