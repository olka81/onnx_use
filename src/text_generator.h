#pragma once
#include <string>
#include <vector>

class TextGenerator {
public:
    TextGenerator(const std::wstring& gpt2_model_path);
    std::wstring generate(int digit) const;

private:
    std::wstring model_path;
    std::wstring tokenizer_path;
    std::wstring run_prepare_input_script(const std::wstring& prompt) const;
    std::vector<int64_t> load_input_ids(const std::string& filename) const;
    std::wstring decode_output_tokens(const std::vector<int64_t>& token_ids) const;
};
