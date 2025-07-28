#include "onnx_classifier.h"
#include <iostream>

#ifndef MODEL_PATH
#ifdef _DEBUG
#define MODEL_PATH L"build/bin/Debug/model-8.onnx"
#else
#define MODEL_PATH L"build/bin/Release/model-8.onnx"
#endif
#endif

int main() {
    OnnxClassifier classifier;
    std::wstring model_path = L"" MODEL_PATH;
    if (!classifier.load(model_path)) {
        std::wcerr << L"Failed to load model." << std::endl;
        return 1;
    }

    std::vector<int64_t> input_shape = classifier.getInputShape();
    size_t input_size = 1;
    for (auto dim : input_shape) input_size *= dim;

    std::vector<float> input(input_size, 0.0f);  // dummy input
    std::vector<float> output = classifier.run(input);

    std::wcout << L"Output: ";
    for (float val : output) {
        std::wcout << val << L" ";
    }
    std::wcout << std::endl;

    return 0;
}