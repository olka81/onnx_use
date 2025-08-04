// onnx_classifier.cpp
#include "onnx_classifier.h"
#include <iostream>
#include <cmath>
#include <numeric>
#include <algorithm>

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

int OnnxClassifier::predict(const std::vector<float> &input_tensor_data)
{
    std::vector<float> output = run(input_tensor_data);

    if (output.empty()) {
        std::wcerr << L"Prediction failed: empty output" << std::endl;
        return -1; // сигнал ошибки
    }

    auto max_iter = std::max_element(output.begin(), output.end());
    return static_cast<int>(std::distance(output.begin(), max_iter));
}

std::vector<float> OnnxClassifier::apply_softmax(const std::vector<float>& logits) {
    std::vector<float> exps(logits.size());
    float max_logit = *std::max_element(logits.begin(), logits.end());

    float sum = 0.0f;
    for (size_t i = 0; i < logits.size(); ++i) {
        exps[i] = std::exp(logits[i] - max_logit);
        sum += exps[i];
    }

    for (float& x : exps) x /= sum;
    return exps;
}

std::vector<std::pair<int, float>> OnnxClassifier::predict_top_k(const std::vector<float>& input_tensor_data, int k) {
    std::vector<float> output = run(input_tensor_data);
    std::vector<float> probs = apply_softmax(output);

    std::vector<int> indices(probs.size());
    std::iota(indices.begin(), indices.end(), 0);

    std::partial_sort(indices.begin(), indices.begin() + k, indices.end(),
        [&](int a, int b) { return probs[a] > probs[b]; });

    std::vector<std::pair<int, float>> top_k;
    for (int i = 0; i < k; ++i) {
        top_k.emplace_back(indices[i], probs[indices[i]]);
    }
    return top_k;
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
