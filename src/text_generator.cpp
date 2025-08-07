#include "text_generator.h"
#include <windows.h>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cstdlib>

TextGenerator::TextGenerator(const std::wstring& gpt2_model_path)
    : model_path(gpt2_model_path) {
    tokenizer_path = L"../third_party/gpt2_tokenizer";
}

std::wstring TextGenerator::generate(int digit) const {
    std::wstring prompt = L"Tell me a joke about the number " + std::to_wstring(digit);
    if (run_prepare_input_script(prompt).empty()) {
        return L"Failed to prepare input.";
    }

    std::vector<int64_t> input_ids = load_input_ids("input_ids.txt");
    if (input_ids.empty()) {
        return L"Failed to load input_ids.";
    }

    // TODO: GPT2 inference
    return L"GPT2 inference not yet implemented.";
}

std::wstring TextGenerator::run_prepare_input_script(const std::wstring& prompt) const {
    std::wstring command = L"python ./src/prepare_input.py \"" + prompt + L"\"";
    int result = _wsystem(command.c_str());
    return result == 0 ? L"ok" : L"";
}

std::vector<int64_t> TextGenerator::load_input_ids(const std::string& filename) const {
    std::ifstream file(filename);
    if (!file) return {};

    std::vector<int64_t> ids;
    int64_t id;
    while (file >> id) {
        ids.push_back(id);
    }
    return ids;
}
