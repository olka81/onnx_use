# ONNX Digit Classifier & Text Generator (C++)

This is a C++ project that demonstrates:
- Inference with an ONNX model for digit classification (MNIST)
- Fake or real text generation (e.g., jokes or proverbs) based on a digit
- Integration with ONNX Runtime (CPU)
- Planned upgrade to GPT-based generation using ONNX-formatted GPT-2

## Prerequisites

1. CMake  
2. C++17 compatible compiler (MSVC, Clang, etc.)  
3. Python 3.x (only for downloading tokenizer, optional)  
4. ONNX Runtime 1.19.0 for Windows x64

## Directory Structure

```
onnx_use/
├── build/                  # CMake build output (Debug/Release)
├── src/                   # C++ sources
├── third_party/
│   ├── onnxruntime-win-x64-1.19.0/  ← Required
│   ├── mnist-8.onnx                ← Required
│   ├── gpt2.onnx                   ← Optional (for text generation)
│   └── gpt2_tokenizer/             ← Optional (for GPT2)
└── CMakeLists.txt
```

## Setup Instructions

### 1. Prepare `third_party/`

You must manually create the `third_party/` folder and place the following files:

- ONNX Runtime 1.19.0 for Windows x64  
  Download: https://github.com/microsoft/onnxruntime/releases/tag/v1.19.0  
  Unpack it as: `third_party/onnxruntime-win-x64-1.19.0/`

- MNIST classification model  
  File: `mnist-8.onnx`  
  Download: https://github.com/onnx/models/blob/main/vision/classification/mnist/model/mnist-8.onnx  
  Save to: `third_party/mnist-8.onnx`

- GPT-2 model for text generation (optional)  
  File: `gpt2.onnx`  
  Download: https://huggingface.co/hlhr202/gpt2-onnx  
  Save to: `third_party/gpt2.onnx`

- GPT-2 tokenizer files (optional)  
  Generate using `transformers` (see below)

## Generate GPT-2 Tokenizer (Optional)

pip install transformers

# Run from project root:
python src/download_tokenizer.py

This will create `third_party/gpt2_tokenizer/` with files like `vocab.json`, `merges.txt`, etc.

## Building the Project

You can build the project either manually using CMake or via the provided batch script.

build.bat Debug

or

build.bat Release

## Running

run.bat Debug path\to\image.png

- The first argument must be either `Debug` or `Release` (depending on your build)
- The second argument is the path to a 28×28 grayscale PNG image containing a digit

## Cleaning the Build

clean.bat

This will remove the `build/` directory and all build artifacts.

## License

MIT or Unlicense — this is an educational demo.
