#include "image_preprocessor.h"
#include <opencv2/opencv.hpp>
#include <iostream>

bool load_and_preprocess_image(
    const std::string& filename,
    std::vector<float>& out_tensor,
    std::vector<int64_t>& shape,
    int target_width,
    int target_height,
    bool grayscale,
    bool normalize
) {
    int flags = grayscale ? cv::IMREAD_GRAYSCALE : cv::IMREAD_COLOR;
    cv::Mat image = cv::imread(filename, flags);

    if (image.empty()) {
        std::cerr << "Failed to load image: " << filename << std::endl;
        return false;
    }

    cv::resize(image, image, cv::Size(target_width, target_height));

    int channels = image.channels();
    shape = {1, channels, target_height, target_width};

    out_tensor.resize(channels * target_height * target_width);

    if (channels == 1) {
        for (int i = 0; i < target_height; ++i) {
            for (int j = 0; j < target_width; ++j) {
                float value = static_cast<float>(image.at<uchar>(i, j));
                if (normalize) value /= 255.0f;
                out_tensor[i * target_width + j] = value;
            }
        }
    } else if (channels == 3) {
        std::vector<cv::Mat> bgr;
        cv::split(image, bgr);
        size_t channel_size = target_height * target_width;
        for (int c = 0; c < 3; ++c) {
            for (int i = 0; i < target_height; ++i) {
                for (int j = 0; j < target_width; ++j) {
                    float value = static_cast<float>(bgr[c].at<uchar>(i, j));
                    if (normalize) value /= 255.0f;
                    out_tensor[c * channel_size + i * target_width + j] = value;
                }
            }
        }
    } else {
        std::cerr << "Unsupported channel count: " << channels << std::endl;
        return false;
    }

    return true;
}
