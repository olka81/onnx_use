#include <onnxruntime_cxx_api.h>
#include <iostream>

int main() {
    std::cout << "Compiled against ORT API version: " << ORT_API_VERSION << std::endl;

    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "test"};
    std::cout << "ONNX Runtime initialized." << std::endl;

    return 0;
}