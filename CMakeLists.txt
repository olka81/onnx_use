cmake_minimum_required(VERSION 3.15)
project(onnx_cpu_demo)

set(CMAKE_CXX_STANDARD 17)

set(ORT_ROOT "${CMAKE_SOURCE_DIR}/third_party/onnxruntime-win-x64-1.19.0")
include_directories(${ORT_ROOT}/include)
link_directories(${ORT_ROOT}/lib)

add_executable(onnx_cpu_demo src/main.cpp)
target_link_libraries(onnx_cpu_demo onnxruntime)


set_target_properties(onnx_cpu_demo PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/bin/Release
    RUNTIME_OUTPUT_DIRECTORY_DEBUG   ${CMAKE_BINARY_DIR}/bin/Debug
)

add_custom_command(TARGET onnx_cpu_demo POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${ORT_ROOT}/lib/onnxruntime.dll"
        "$<TARGET_FILE_DIR:onnx_cpu_demo>"
    COMMENT "Copying onnxruntime.dll to output directory"
)