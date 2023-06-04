cmake_minimum_required(VERSION 3.15)
project(tensorrt_yolo)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

find_package(OpenCV)
message(STATUS "opencv include path: ${OpenCV_INCLUDE_DIRS}")

include_directories(
    /usr/local/cuda/targets/x86_64-linux/include
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/install/include
)

link_directories(
    ${CMAKE_SOURCE_DIR}/install/lib   
)

set(CMAKE_BUILD_TYPE Release)
# setup CUDA
enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)
message(STATUS "cuda include path: ${CUDAToolkit_INCLUDE_DIRS}")
message(STATUS "cuda library path: ${CUDAToolkit_LIBRARY_DIR}")

include_directories(${CUDAToolkit_INCLUDE_DIRS})
link_directories(${CUDAToolkit_LIBRARY_DIR})

set(CMAKE_CUDA_ARCHITECTURES 86)
add_subdirectory(src)
