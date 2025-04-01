#include "image_encryption.hpp"

encrypt_image::encrypt_image(std::string input_path, bool remove_noise, int noise_level) {
    input_img = cv::imread(input_path, cv::IMREAD_COLOR);
    if (remove_noise) {
        cv::fastNlMeansDenoisingColored(input_img, input_img, noise_level);
    }
    rows = input_img.rows;
    cols = input_img.cols;
    org_size.width = cols;
    org_size.height = rows;
    cv::resize(input_img, input_img, cv::Size(nextClosestDivisible(input_img.cols, 24), nextClosestDivisible(input_img.rows, 24)));
}

encrypt_image::encrypt_image(cv::Mat input, bool remove_noise, int noise_level) {
    input.copyTo(input_img);
    if (remove_noise) {
        cv::fastNlMeansDenoisingColored(input_img, input_img, noise_level);
    }
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

cv::Mat encrypt_image::get_sampled_mat() {
    cv::Mat sampled_mat = cv::Mat::zeros(rows, cols, CV_8UC3);

    for (int i = 0; i < ri_x.size(); i++) {
        sampled_mat.at<cv::Vec3b>(ri_x[i], ri_y[i]) = input_img.at<cv::Vec3b>(ri_x[i], ri_y[i]);
    }

    return sampled_mat.clone();
}

cv::Mat encrypt_image::get_sampled_mask() {
    cv::Mat sampled_mat = cv::Mat::zeros(rows, cols, CV_8UC3);
    sampled_mat.setTo(cv::Scalar(0, 0, 0));

    for (int i = 0; i < ri_x.size(); i++) {
        sampled_mat.at<cv::Vec3b>(ri_x[i], ri_y[i]) = cv::Vec3b(1, 1, 1);
    }

    return sampled_mat.clone();
}

void encrypt_image::encrypt(float pixel_p, std::string seed) {
    bm = pixel_p;
    m = rows * cols * bm;
    int n = rows * cols;
    ri_x.resize(m);
    ri_y.resize(m);
    returnRandomIndices(ri_x, ri_y, rows, cols, m, seed);
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

void encrypt_image::encrypt(std::vector<int> ri_x_g, std::vector<int> ri_y_g) {
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

void encrypt_image::writeEncryptedImageToDisk(std::string output_path) {
    cv::imwrite(output_path, encrypted_img);
}