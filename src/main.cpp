#include <windows.h>
#include "image_preprocessor.h"
#include "onnx_classifier.h"
#include "gpt2_generator.h"

#include <iostream>
#include <fstream>
#include <filesystem>
#include <sstream>
#include <cstdlib>

bool load_input_ids(const std::string& path, std::vector<int64_t>& ids) {
    std::ifstream in(path);
    if (!in.is_open()) return false;
    ids.clear();
    int64_t id;
    while (in >> id) ids.push_back(id);
    return true;
}

std::wstring getExecutableDir() {
    wchar_t buffer[MAX_PATH];
    GetModuleFileNameW(nullptr, buffer, MAX_PATH);
    std::wstring path(buffer);
    return path.substr(0, path.find_last_of(L"\\/") + 1);
}

int main(int argc, char* argv[]) {
    std::string image_path = "src/digit.png";
    if (argc >= 2) {
        image_path = std::filesystem::path(argv[1]).string();
    }

    if (!std::filesystem::exists(image_path)) {
        std::cerr << "Error: Image file does not exist: " << image_path << std::endl;
        return 1;
    }

    std::wstring exe_dir = getExecutableDir();
    std::wstring model_path = exe_dir + L"model-8.onnx";

    OnnxClassifier classifier;
    if (!classifier.load(model_path)) {
        std::wcerr << L"Failed to load model." << std::endl;
        return 1;
    }

    auto input_shape = classifier.getInputShape();
    std::vector<float> input_tensor;
    if (!load_and_preprocess_image(image_path, input_tensor, input_shape, 28, 28, true, true)) {
        std::wcerr << L"Failed to load or preprocess image." << std::endl;
        return 1;
    }

    int digit = classifier.predict(input_tensor);
    std::wcout << L"Recognized digit: " << digit << std::endl;

    {
        std::stringstream cmd;
        cmd << "python src/prepare_input.py " << digit;
        int ret = std::system(cmd.str().c_str());
        if (ret != 0) {
            std::cerr << "Failed to run prepare_input.py" << std::endl;
            return 1;
        }
    }

    std::vector<int64_t> input_ids, attention_mask;
    if (!load_input_ids("input_ids.txt", input_ids)) {
        std::wcerr << L"Failed to load input_ids.txt" << std::endl;
        return 1;
    }
    if (!load_input_ids("attention_mask.txt", attention_mask)) {
        std::wcerr << L"Failed to load attention_mask.txt" << std::endl;
        return 1;
    }

    std::wstring gpt2_path = exe_dir + L"gpt2.onnx";
    Gpt2Generator generator(gpt2_path);
    auto output_ids = generator.generate_sequence(input_ids, attention_mask);

    std::wcout << L"Generated token IDs:";
    for (int64_t id : output_ids) std::wcout << L" " << id;
    std::wcout << std::endl;

    {
        std::stringstream cmd;
        cmd << "python src/decode_output.py";
        for (int64_t id : output_ids) cmd << " " << id;
        std::cout << "Decoded text:\n";
        int ret = std::system(cmd.str().c_str());
        if (ret != 0) {
            std::cerr << "Failed to run decode_output.py" << std::endl;
            return 1;
        }
    }

    return 0;
}
