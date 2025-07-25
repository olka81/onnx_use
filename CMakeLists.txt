cmake_minimum_required(VERSION 3.15)
project(onnx_gpu_demo)

set(CMAKE_CXX_STANDARD 17)

# vcpkg paths
set(VCPKG_INSTALLED_PATH "D:/vcpkg/installed/x64-windows")

include_directories(${VCPKG_INSTALLED_PATH}/include)
link_directories(${VCPKG_INSTALLED_PATH}/lib)

add_executable(onnx_gpu_demo src/main.cpp)

target_link_libraries(onnx_gpu_demo PRIVATE onnxruntime)

set_target_properties(onnx_gpu_demo PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
)