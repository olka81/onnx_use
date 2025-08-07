#pragma once

#include <onnxruntime_cxx_api.h>
#include <string>
#include <vector>
#include <optional>

class Gpt2Generator {
public:
    explicit Gpt2Generator(const std::wstring& model_path);

    int64_t generate_next_token(const std::vector<int64_t>& input_ids,
                                 const std::vector<int64_t>& attention_mask);

    std::vector<int64_t> generate_sequence(const std::vector<int64_t>& input_ids,
                                           const std::vector<int64_t>& attention_mask,
                                           size_t max_tokens = 20);

private:
    Ort::Env env;
    Ort::SessionOptions session_options;
    std::unique_ptr<Ort::Session> session;

    Ort::AllocatorWithDefaultOptions allocator;
    std::optional<Ort::AllocatedStringPtr> input_name_holder;
    std::optional<Ort::AllocatedStringPtr> attention_mask_name_holder;
    std::optional<Ort::AllocatedStringPtr> output_name_holder;

    std::vector<int64_t> input_shape;
};
