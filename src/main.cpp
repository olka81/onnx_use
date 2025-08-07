#include "onnx_classifier.h"
#include "image_preprocessor.h"
#include "text_generator.h"
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

    std::wstring gpt2_model_path = exe_dir + L"\\gpt2.onnx";
    TextGenerator generator(gpt2_model_path);
    std::wcout << generator.generate(predicted_digit) << std::endl;

    // auto top_predictions = classifier.predict_top_k(input_tensor, 3);

    // if (!top_predictions.empty()) {
    //     std::wcout << L"Predicted digit: " << top_predictions[0].first
    //            << L" (" << top_predictions[0].second * 100.0f << L"%)" << std::endl;

    //     std::wcout << L"Top 3 predictions:\n";
    //     for (const auto& [label, prob] : top_predictions) {
    //         std::wcout << L"  " << label << L": " << prob * 100.0f << L"%\n";
    //     }
    // } else {
    //     std::wcerr << L"Prediction failed." << std::endl;
    // }

    return 0;
}