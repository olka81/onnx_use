#include <onnxruntime_cxx_api.h>
#include <iostream>

int main() {
    try {
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "gpu_test");

        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);

        // Попробуем включить CUDA Execution Provider
        OrtCUDAProviderOptions cuda_options;
        session_options.AppendExecutionProvider_CUDA(cuda_options);

        std::cout << "ONNX Runtime with CUDA loaded!" << std::endl;
    } catch (const Ort::Exception& e) {
        std::cerr << "ONNX Runtime error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}