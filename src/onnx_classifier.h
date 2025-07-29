#pragma once

#include <onnxruntime_cxx_api.h>
#include <memory>
#include <string>
#include <vector>
#include <optional>

class OnnxClassifier {
public:
    OnnxClassifier();
    bool load(const std::wstring& model_path);
    std::vector<float> run(const std::vector<float>& input_tensor);
    std::vector<int64_t> getInputShape() const;
    int predict(const std::vector<float>& input_tensor_data);

private:
    Ort::Env env;
    Ort::SessionOptions session_options;
    Ort::AllocatorWithDefaultOptions allocator;
    std::unique_ptr<Ort::Session> session;

    std::optional<Ort::AllocatedStringPtr> input_name_holder;
    std::optional<Ort::AllocatedStringPtr> output_name_holder;

    std::vector<int64_t> input_shape;
};
