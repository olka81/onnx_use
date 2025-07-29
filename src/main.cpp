#include "onnx_classifier.h"
#include "image_preprocessor.h"
#include <iostream>
#include <windows.h>
#include <filesystem>

std::wstring getExecutableDir() {
    wchar_t buffer[MAX_PATH];
    GetModuleFileNameW(NULL, buffer, MAX_PATH);
    std::filesystem::path exePath(buffer);
    return exePath.parent_path().wstring();
}

int main(int argc, char* argv[]) {
    std::string image_path = "src/digit.png"; 

    if (argc >= 2) {
        image_path = argv[1];
    }

    if (!std::filesystem::exists(image_path)) {
        std::cerr << "Error: Image file does not exist: " << image_path << std::endl;
        return 1;
    }

    std::wstring exe_dir = getExecutableDir();
    std::wstring model_path = exe_dir + L"\\model-8.onnx";

    OnnxClassifier classifier;
    if (!classifier.load(model_path)) {
        std::wcerr << L"Failed to load model." << std::endl;
        return 1;
    }

    std::vector<float> input_tensor;
    std::vector<int64_t> input_shape;

    if (!load_and_preprocess_image(image_path, input_tensor, input_shape, 28, 28, true, true)) {
        std::cerr << "Failed to load or preprocess image!" << std::endl;
        return 1;
    }

    if (input_shape != classifier.getInputShape()) {
        std::cerr << "Image shape doesn't match model input shape!" << std::endl;
        return 1;
    }

    int predicted_digit = classifier.predict(input_tensor);
    std::wcout << L"Predicted digit: " << predicted_digit << std::endl;

    return 0;
}