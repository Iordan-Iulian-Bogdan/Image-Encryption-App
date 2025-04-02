#include "image_encryption.hpp"

encrypt_image::encrypt_image(std::string input_path) {
    input_img = cv::imread(input_path, cv::IMREAD_COLOR);
    rows = input_img.rows;
    cols = input_img.cols;
    org_size.width = cols;
    org_size.height = rows;
    cv::resize(input_img, input_img, cv::Size(nextClosestDivisible(input_img.cols, 24), nextClosestDivisible(input_img.rows, 24)));
}

encrypt_image::encrypt_image(const cv::Mat& input) {
    input.copyTo(input_img);
    rows = input_img.rows;
    cols = input_img.cols;
    org_size.width = cols;
    org_size.height = rows;
}

void encrypt_image::get_mat(cv::Mat& dest) {
    encrypted_img.copyTo(dest);
}

cv::Mat encrypt_image::get_mat() {
    return encrypted_img;// .clone();
}

void encrypt_image::encrypt(const float& pixel_p, const std::string& passwrod) {
    bm = pixel_p;
    m = rows * cols * bm;
    int n = rows * cols;
    ri_x.resize(m);
    ri_y.resize(m);
    returnRandomIndices(ri_x, ri_y, rows, cols, m, passwrod);
    encrypted_img = cv::Mat(1, (int)(sqrt(m + 6) + 1) * (int)(sqrt(m + 6) + 1), CV_8UC3);

    std::string text = std::to_string(m) + "|" + std::to_string(rows) + "|" + std::to_string(cols) + "|" + std::to_string(org_size.height) + "|" + std::to_string(org_size.width); // Example string 
    std::string padding;

    int paddingSize = 32 - text.size();
    for (int i = 0; i < paddingSize; i++) {
        padding += "-";
    }
    text = padding + text;
    cv::Mat colorMat;
    storeStringInColorMat(text, colorMat);
    std::string retrievedText = retrieveStringFromColorMat(colorMat);

    int i = 0;

    for (; i < colorMat.total(); i++) {
        encrypted_img.at<cv::Vec3b>(i) = colorMat.at<cv::Vec3b>(i);
    }

    int k = 0;

    for (; (i < encrypted_img.rows * encrypted_img.cols) && k < m - 1; i++) {
        encrypted_img.at<cv::Vec3b>(k + colorMat.total()) = input_img.at<cv::Vec3b>(ri_x[k], ri_y[k]);
        k++;
    }

    encrypted_img = encrypted_img.reshape(0, (int)sqrt(encrypted_img.total()));
}

void encrypt_image::encrypt(const std::vector<int>& ri_x_g, const std::vector<int>& ri_y_g) {
    m = ri_x_g.size();
    int n = rows * cols;
    encrypted_img = cv::Mat(1, (int)(sqrt(m + 6) + 1) * (int)(sqrt(m + 6) + 1), CV_8UC3);

    std::string text = std::to_string(m) + "|" + std::to_string(rows) + "|" + std::to_string(cols) + "|" + std::to_string(org_size.height) + "|" + std::to_string(org_size.width); // Example string 
    std::string padding;

    int paddingSize = 32 - text.size();
    for (int i = 0; i < paddingSize; i++) {
        padding += "-";
    }
    text = padding + text;
    cv::Mat colorMat;
    storeStringInColorMat(text, colorMat);
    std::string retrievedText = retrieveStringFromColorMat(colorMat);

    int i = 0;

    for (; i < colorMat.total(); i++) {
        encrypted_img.at<cv::Vec3b>(i) = colorMat.at<cv::Vec3b>(i);
    }

    int k = 0;

    for (; (i < encrypted_img.rows * encrypted_img.cols) && k < m - 1; i++) {
        encrypted_img.at<cv::Vec3b>(k + colorMat.total()) = input_img.at<cv::Vec3b>(ri_x_g[k], ri_y_g[k]);
        k++;
    }

    encrypted_img = encrypted_img.reshape(0, (int)sqrt(encrypted_img.total()));
}

cv::Size encrypt_image::get_size() {
    cv::Size tile_size;
    tile_size.width = cols;
    tile_size.height = rows;

    return tile_size;
}

void encrypt_image::writeEncryptedImageToDisk(const std::string& output_path) {
    cv::imwrite(output_path, encrypted_img);
}