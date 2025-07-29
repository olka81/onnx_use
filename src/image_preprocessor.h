#pragma once
#include <string>
#include <vector>

bool load_and_preprocess_image(
    const std::string& filename,
    std::vector<float>& out_tensor,
    std::vector<int64_t>& shape,     // [1, C, H, W]
    int target_width,
    int target_height,
    bool grayscale,
    bool normalize
);