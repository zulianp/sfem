cmake_minimum_required(VERSION 3.10)

project(sfem_hardware_tools LANGUAGES C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

find_package(OpenCL REQUIRED)

add_executable(opencl_devices opencl_devices.c)
target_link_libraries(opencl_devices PRIVATE OpenCL::OpenCL)
target_include_directories(opencl_devices PRIVATE ${OpenCL_INCLUDE_DIRS})

if(APPLE)
    target_compile_definitions(opencl_devices PRIVATE CL_TARGET_OPENCL_VERSION=120)
    target_link_libraries(opencl_devices PRIVATE "-framework OpenCL")
endif()

if(CMAKE_C_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(opencl_devices PRIVATE -Wall -Wextra -Wpedantic)
endif()

install(TARGETS opencl_devices RUNTIME DESTINATION bin)
