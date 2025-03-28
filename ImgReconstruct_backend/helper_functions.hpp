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

int nextClosestDivisible(int x, int y);

cv::Mat reconstructImage(const std::vector<std::vector<cv::Mat>>& tiles,
    const std::vector<std::vector<TileCoord>>& coordinates);
// Function to update the image
void updateImage(const std::string& windowName, const cv::Mat& newImage);
std::vector<cv::Mat> splitMat(cv::Mat& image, int M, int N);
inline void updateAxb2AndComputeFx(float* x_copy, int* ri_x, int* ri_y,
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
std::vector<cv::Mat> createRefDCT(int rows, int cols);
void reconstruct_color_chanel(cv::Mat& out, cv::Mat& measurement, int k, float param_c, float optimal_value, int rows, int cols, std::vector<int>& ri_x, std::vector<int>& ri_y, int iterations, std::vector<cv::Mat> ref, bool opt, int tile_index, bool copy_result);
std::vector<std::string> splitString(const std::string& str, char delimiter);
std::string removeCharacter(const std::string& str, char ch);
void storeStringInColorMat(const std::string& text, cv::Mat& colorMat);
std::string retrieveStringFromColorMat(const cv::Mat& colorMat);
std::vector<cv::Mat> splitImageIntoTiles(const cv::Mat& image, int tile_width, int tile_height, int rows, int cols);
std::vector<std::string> spiralOrder(std::vector<std::vector<std::string>>& matrix);
void splitImageIntoTiles(const cv::Mat& inputImage,
    std::vector<std::vector<cv::Mat>>& tiles,
    std::vector<std::vector<TileCoord>>& coordinates,
    int tileCountN,
    int overlap);
cv::Mat blendTilesWithImage(const std::vector<std::vector<cv::Mat>>& tiles,
    const std::vector<std::vector<TileCoord>>& coordinates,
    const cv::Mat& targetImage,
    float alpha);