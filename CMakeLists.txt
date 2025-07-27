cmake_minimum_required(VERSION 3.15)
project(onnx_cpu_demo)
set(CMAKE_CXX_STANDARD 17)

include(FetchContent)

set(ORT_VERSION "1.19.0")
set(ORT_ARCHIVE_NAME "onnxruntime-win-x64-${ORT_VERSION}.zip")
set(ORT_URL "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/${ORT_ARCHIVE_NAME}")
set(ORT_LOCAL_ZIP "${CMAKE_BINARY_DIR}/${ORT_ARCHIVE_NAME}")
set(ORT_EXTRACT_DIR "${CMAKE_SOURCE_DIR}/third_party/onnxruntime-win-x64-${ORT_VERSION}")

# Скачать архив, если он ещё не загружен
if(NOT EXISTS "${ORT_LOCAL_ZIP}")
    message(STATUS "Downloading ONNX Runtime ${ORT_VERSION}...")
    file(DOWNLOAD
        "${ORT_URL}"
        "${ORT_LOCAL_ZIP}"
        SHOW_PROGRESS
    )
endif()

# Распаковать, если не распаковано
if(NOT EXISTS "${ORT_EXTRACT_DIR}/include")
    message(STATUS "Extracting ONNX Runtime...")
    file(MAKE_DIRECTORY "${CMAKE_SOURCE_DIR}/third_party")
    execute_process(
        COMMAND ${CMAKE_COMMAND} -E tar xf "${ORT_LOCAL_ZIP}"
        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/third_party"
    )
endif()

# Подключить include и lib
include_directories("${ORT_EXTRACT_DIR}/include")
link_directories("${ORT_EXTRACT_DIR}/lib")

# Скомпилировать основной исполняемый файл
add_executable(onnx_cpu_demo src/main.cpp)
target_link_libraries(onnx_cpu_demo onnxruntime)

foreach(OUTPUTCONFIG ${CMAKE_CONFIGURATION_TYPES})
    string(TOUPPER ${OUTPUTCONFIG} OUTPUTCONFIG_UPPER)
    set(MODEL_PATH_${OUTPUTCONFIG_UPPER} "${CMAKE_BINARY_DIR}/bin/${OUTPUTCONFIG}/model-8.onnx")
endforeach()

# Передать макрос в компилятор
target_compile_definitions(onnx_cpu_demo PRIVATE
    $<$<CONFIG:Debug>:MODEL_PATH=\"${MODEL_PATH_DEBUG}\">
    $<$<CONFIG:Release>:MODEL_PATH=\"${MODEL_PATH_RELEASE}\">
)

# Выходной путь
set_target_properties(onnx_cpu_demo PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/bin/Release
    RUNTIME_OUTPUT_DIRECTORY_DEBUG   ${CMAKE_BINARY_DIR}/bin/Debug
)

# Копировать DLL рядом с .exe после сборки
add_custom_command(TARGET onnx_cpu_demo POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${ORT_EXTRACT_DIR}/lib/onnxruntime.dll"
        "$<TARGET_FILE_DIR:onnx_cpu_demo>"
    COMMENT "Copying onnxruntime.dll to output directory"
)
add_custom_command(TARGET onnx_cpu_demo POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_SOURCE_DIR}/third_party/mnist-8.onnx"
        "$<TARGET_FILE_DIR:onnx_cpu_demo>/model-8.onnx"
    COMMENT "Copying ONNX model to output directory"
)