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
    std::unique_ptr<Ort::Session> session;

    try {
        session = std::make_unique<Ort::Session>(env, model_path.c_str(), session_options);
        std::wcout << L"Model loaded successfully." << std::endl;
    } catch (const Ort::Exception& e) {
        std::wcerr << L"ORT Exception: " << e.what() << std::endl;
        return 1;
    }
    Ort::AllocatorWithDefaultOptions allocator;

    // Входы
    size_t num_inputs = session->GetInputCount();
    std::wcout << L"Number of inputs: " << num_inputs << std::endl;

    for (size_t i = 0; i < num_inputs; ++i) {
        Ort::AllocatedStringPtr input_name = session->GetInputNameAllocated(i, allocator);
        std::wcout << L"Input " << i << L" name: " << input_name.get() << std::endl;

        Ort::TypeInfo type_info = session->GetInputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        std::vector<int64_t> shape = tensor_info.GetShape();

        std::wcout << L"  Shape: [ ";
        for (auto dim : shape) std::wcout << dim << L" ";
        std::wcout << L"]" << std::endl;
    }
    
    return 0;
}