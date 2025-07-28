// onnx_classifier.cpp
#include "onnx_classifier.h"
#include <iostream>

OnnxClassifier::OnnxClassifier()
    : env(ORT_LOGGING_LEVEL_WARNING, "onnx_classifier") {
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
}

bool OnnxClassifier::load(const std::wstring& model_path) {
    try {
        session = std::make_unique<Ort::Session>(env, model_path.c_str(), session_options);
        input_name_holder = session->GetInputNameAllocated(0, allocator);
        output_name_holder = session->GetOutputNameAllocated(0, allocator);

        auto type_info = session->GetInputTypeInfo(0);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        input_shape = tensor_info.GetShape();

        std::wcout << L"Model loaded successfully." << std::endl;
        return true;
    } catch (const Ort::Exception& e) {
        std::wcerr << L"Failed to load model: " << e.what() << std::endl;
        return false;
    }
}

std::vector<int64_t> OnnxClassifier::getInputShape() const {
    return input_shape;
}

std::vector<float> OnnxClassifier::run(const std::vector<float>& input_tensor_data) {
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info,
        const_cast<float*>(input_tensor_data.data()),
        input_tensor_data.size(),
        input_shape.data(),
        input_shape.size()
    );

    try {
        const char* input_name = input_name_holder->get();
        const char* output_name = output_name_holder->get();

        auto output_tensors = session->Run(
            Ort::RunOptions{nullptr},
            &input_name, &input_tensor, 1,
            &output_name, 1
        );

        float* output_data = output_tensors[0].GetTensorMutableData<float>();
        auto output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
        size_t output_size = 1;
        for (auto dim : output_shape) output_size *= dim;

        return std::vector<float>(output_data, output_data + output_size);
    } catch (const Ort::Exception& e) {
        std::wcerr << L"Run() failed: " << e.what() << std::endl;
        return {};
    }
}
