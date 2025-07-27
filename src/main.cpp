#include <onnxruntime_cxx_api.h>
#include <iostream>

int main() {

    std::wcout << L"Compiled against ORT API version: " << ORT_API_VERSION << std::endl;

    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "demo"};

    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    std::wstring model_path = L"model-8.onnx";
    Ort::Session session(env, model_path.c_str(), session_options);

    std::wcout << L"Model loaded successfully." << std::endl;
    return 0;
}