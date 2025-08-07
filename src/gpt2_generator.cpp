#include "gpt2_generator.h"
#include <iostream>
#include <algorithm>

Gpt2Generator::Gpt2Generator(const std::wstring& model_path)
    : env(ORT_LOGGING_LEVEL_WARNING, "gpt2_generator") {
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    session = std::make_unique<Ort::Session>(env, model_path.c_str(), session_options);

    input_name_holder = session->GetInputNameAllocated(0, allocator);
    attention_mask_name_holder = session->GetInputNameAllocated(1, allocator);
    output_name_holder = session->GetOutputNameAllocated(0, allocator);

    auto type_info = session->GetInputTypeInfo(0);
    input_shape = type_info.GetTensorTypeAndShapeInfo().GetShape();

    std::wcout << L"GPT2 model loaded successfully." << std::endl;
}

int64_t Gpt2Generator::generate_next_token(const std::vector<int64_t>& input_ids,
                                           const std::vector<int64_t>& attention_mask) {
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    std::vector<int64_t> shape = {1, static_cast<int64_t>(input_ids.size())};

    Ort::Value input_tensor = Ort::Value::CreateTensor<int64_t>(
        memory_info,
        const_cast<int64_t*>(input_ids.data()),
        input_ids.size(),
        shape.data(), shape.size()
    );

    Ort::Value attention_tensor = Ort::Value::CreateTensor<int64_t>(
        memory_info,
        const_cast<int64_t*>(attention_mask.data()),
        attention_mask.size(),
        shape.data(), shape.size()
    );

    const char* input_name = input_name_holder.value().get();
    const char* mask_name = attention_mask_name_holder.value().get();
    const char* output_name = output_name_holder.value().get();

    std::array<const char*, 2> input_names = {input_name, mask_name};
    std::array<Ort::Value, 2> input_tensors = {std::move(input_tensor), std::move(attention_tensor)};

    auto output_tensors = session->Run(
        Ort::RunOptions{nullptr},
        input_names.data(), input_tensors.data(), input_tensors.size(),
        &output_name, 1
    );

    auto& logits_tensor = output_tensors[0];
    float* logits_data = logits_tensor.GetTensorMutableData<float>();

    auto shape_out = logits_tensor.GetTensorTypeAndShapeInfo().GetShape();
    if (shape_out.size() != 3) {
        std::wcerr << L"Unexpected output shape" << std::endl;
        return -1;
    }

    int64_t vocab_size = shape_out[2];
    int64_t last_index = (shape_out[1] - 1) * vocab_size;

    const float* last_logits = logits_data + last_index;
    auto max_iter = std::max_element(last_logits, last_logits + vocab_size);
    return static_cast<int64_t>(std::distance(last_logits, max_iter));
}

std::vector<int64_t> Gpt2Generator::generate_sequence(const std::vector<int64_t>& input_ids,
                                                      const std::vector<int64_t>& attention_mask,
                                                      size_t max_tokens) {
    std::vector<int64_t> ids = input_ids;
    std::vector<int64_t> mask = attention_mask;

    for (size_t i = 0; i < max_tokens; ++i) {
        int64_t next_token = generate_next_token(ids, mask);
        if (next_token < 0) break;
        ids.push_back(next_token);
        mask.push_back(1); // Extend attention mask

        if (next_token == 50256) break;
    }

    return ids;
}
