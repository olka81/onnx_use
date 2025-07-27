#include <onnxruntime_cxx_api.h>
#include <iostream>

#ifndef MODEL_PATH
#ifdef _DEBUG
    #define MODEL_PATH L"build/bin/Debug/model-8.onnx";
#else
    #define MODEL_PATH L"build/bin/Release/model-8.onnx";
#endif
#endif

int main() {
    std::wcout << L"Compiled against ORT API version: " << ORT_API_VERSION << std::endl;

    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "demo"};

    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    std::wstring model_path = L"" MODEL_PATH;
    std::wcout << L"Model path: " << model_path << std::endl;

    try {
        Ort::Session session(env, model_path.c_str(), session_options);
        std::wcout << L"Model loaded successfully." << std::endl;
    } catch (const Ort::Exception& e) {
        std::wcerr << L"ORT Exception: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}