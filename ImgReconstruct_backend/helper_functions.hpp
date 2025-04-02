#include <stdio.h>
#include "lbfgs.hpp"
#include <mutex>
#include <vector>
#include <string>
#include <chrono>
#include <stdlib.h>
#include <random>
#include <iostream>

struct TileCoord {
    int x;
    int y;
};

struct indices {
    std::vector<int> ri_x_g, ri_y_g;
};

std::vector<unsigned long> generate_seeds(const std::string input);

int nextClosestDivisible(const int& x, const int& y);

cv::Mat reconstructImage(const std::vector<std::vector<cv::Mat>>& tiles,
    const std::vector<std::vector<TileCoord>>& coordinates);
// Function to update the image
void updateImage(const std::string& windowName, const cv::Mat& newImage);
std::vector<cv::Mat> splitMat(cv::Mat& image, int M, int N);
inline void updateAxb2AndComputeFx(float* x_copy, const int* ri_x, const int* ri_y,
    float* Axb2_vec, const float* b, int cols, float& fx, int n);
inline void eval_g(float* Axb2, float* g, int n);
inline void copy_x(float* x_copy, float* x, float* Axb2_vec, int n);
float evaluate(
    void* instance,
    const float* x,
    eval_data data,
    float* g,
    const int n,
    const float step
);
int progress(
    void* instance,
    const float* x,
    const float* g,
    const float fx,
    const float xnorm,
    const float gnorm,
    const float step,
    int n,
    int k,
    int ls
);
std::vector<cv::Mat> createRefSolutions(const int& rows, const int& cols);
void reconstruct_color_channel(const cv::Mat& measurement, const int& k, const float& param_c, const int& rows, const int& cols, const std::vector<int>& ri_x, const std::vector<int>& ri_y, const int& iterations, cv::Mat& ref);
std::vector<std::string> splitString(const std::string& str, const char& delimiter);
std::string removeCharacter(const std::string& str, const char& ch);
void storeStringInColorMat(const std::string& text, cv::Mat& colorMat);
std::string retrieveStringFromColorMat(const cv::Mat& colorMat);
std::vector<cv::Mat> splitImageIntoTiles(const cv::Mat& image, const int& tile_width, const int& tile_height, const int& rows, const int& cols);
std::vector<std::string> spiralOrder(std::vector<std::vector<std::string>>& matrix);
void splitImageIntoTiles(const cv::Mat& inputImage,
    std::vector<std::vector<cv::Mat>>& tiles,
    std::vector<std::vector<TileCoord>>& coordinates,
    const int& tileCountN,
    const int& overlap);
cv::Mat blendTilesWithImage(const std::vector<std::vector<cv::Mat>>& tiles,
    const std::vector<std::vector<TileCoord>>& coordinates,
    const cv::Mat& targetImage,
    float alpha);

struct display {
    std::thread display_output_thread;
    bool g_dsp = true;

    void display_output(std::string windowName, cv::Mat& temp, const std::vector<std::vector<TileCoord>>& coordinates, const std::vector<std::vector<cv::Mat>>& reconfigured_cropped_out) {
        while (g_dsp) {
            cv::waitKey(33);
            temp = reconstructImage(reconfigured_cropped_out, coordinates);
            updateImage(windowName, temp);
            cv::waitKey(1);
        }
    }

    void display_image(const std::string& windowName, cv::Mat& reconstructed, const std::vector<std::vector<TileCoord>>& coordinates, const std::vector<std::vector<cv::Mat>>& reconfigured_cropped_out) {
        display_output_thread = std::thread(&display::display_output, this, windowName, std::ref(reconstructed), std::ref(coordinates), std::ref(reconfigured_cropped_out));
    }

    void stop_display() {
        g_dsp = false;
        display_output_thread.join();
    }
};