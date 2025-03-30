#include "image_decryption.hpp"

decrypt_image::decrypt_image(std::string input_path) {
    encrypted_img = cv::imread(input_path, cv::IMREAD_COLOR);
    encrypted_img = encrypted_img.reshape(0, encrypted_img.total());
    cv::Mat retrievedMat(11, 1, CV_8UC3);

    for (int i = 0; i < 11; i++) {
        retrievedMat.at<cv::Vec3b>(i) = encrypted_img.at<cv::Vec3b>(i);
    }

    std::string retrievedInfo = retrieveStringFromColorMat(retrievedMat);
    retrievedInfo = removeCharacter(retrievedInfo, '-');

    std::vector<std::string> splitText = splitString(retrievedInfo, '|');
    m = std::stoi(splitText[0]);
    rows = std::stoi(splitText[1]);
    cols = std::stoi(splitText[2]);
    optimal_values.resize(3);
    org_size.height = (float)std::stoi(splitText[3]);
    org_size.width = (float)std::stoi(splitText[4]);
}

decrypt_image::decrypt_image(cv::Mat input) {
    input.copyTo(encrypted_img);
    encrypted_img = encrypted_img.reshape(0, encrypted_img.total());
    cv::Mat retrievedMat(11, 1, CV_8UC3);

    for (int i = 0; i < 11; i++) {
        retrievedMat.at<cv::Vec3b>(i) = encrypted_img.at<cv::Vec3b>(i);
    }

    std::string retrievedInfo = retrieveStringFromColorMat(retrievedMat);
    retrievedInfo = removeCharacter(retrievedInfo, '-');

    std::vector<std::string> splitText = splitString(retrievedInfo, '|');
    m = std::stoi(splitText[0]);
    rows = std::stoi(splitText[1]);
    cols = std::stoi(splitText[2]);
    optimal_values.resize(3);
    org_size.height = (float)std::stoi(splitText[3]);
    org_size.width = (float)std::stoi(splitText[4]);
}

void decrypt_image::decrypt(std::vector<cv::Mat> ref, std::vector<int>& ri_x_g, std::vector<int>& ri_y_g, int num_iterations, float coef, cv::Mat& out) {
    ri_x = ri_x_g;
    ri_y = ri_y_g;
    //std::vector<cv::Mat> ref = createRefDCT(rows, cols);
    int n = rows * cols;
    std::vector<std::thread> CPUProcessing(3);

    for (int i = 0; i < 3; i++) {
        reconstruct_color_chanel(c[i], encrypted_img, i, coef, rows, cols, ri_x, ri_y, num_iterations, ref);
    }

    cv::merge(c, 3, decrypted_img);
    decrypted_img.convertTo(decrypted_img, CV_8UC3);
    out = decrypted_img.clone();
}

void decrypt_image::get_mat(cv::Mat& dest) {
    decrypted_img.copyTo(dest);
}

cv::Mat decrypt_image::get_mat() {
    return decrypted_img.clone();
}

void decrypt_image::writeDecryptedImageToDisk(std::string output_path, bool remove_noise, bool noise_level) {

    cv::imwrite(output_path, decrypted_img);

    if (remove_noise) {
        cv::Mat decrypted_img_noisless = cv::imread(output_path, cv::IMREAD_COLOR);

        cv::fastNlMeansDenoisingColored(decrypted_img_noisless, decrypted_img_noisless, noise_level);
        cv::imwrite(output_path, decrypted_img_noisless);
    }
}


cv::Mat decrypt_image::get_sampled_mat(int seed) {
    cv::Mat sampled_mat = cv::Mat::zeros(rows, cols, CV_8UC3);
    sampled_mat.setTo(cv::Scalar(255, 255, 255));
    ri_x.resize(m);
    ri_y.resize(m);
    returnRandomIndices(ri_x, ri_y, rows, cols, m, seed);

    for (int i = 11; i < ri_x.size() + 11; ++i) {
        sampled_mat.at<cv::Vec3b>(ri_x[i - 11], ri_y[i - 11]) = encrypted_img.at<cv::Vec3b>(i);
    }

    return sampled_mat;
}

cv::Mat decrypt_image::get_sampled_mask(int seed) {

    cv::Mat sampled_mask = cv::Mat::zeros(rows, cols, CV_8UC3);

    for (int i = 0; i < ri_x.size(); i++) {
        sampled_mask.at<cv::Vec3b>(ri_x[i], ri_y[i]) = cv::Vec3b(1, 1, 1);
    }

    return sampled_mask.clone();
}

void decrypt_image::get_sampled_mask_mats(int seed, cv::Mat& sampled_mat, cv::Mat& sampled_mask) {
    sampled_mask = cv::Mat::zeros(rows, cols, CV_8UC3);
    sampled_mat = cv::Mat::zeros(rows, cols, CV_8UC3);
    ri_x.resize(m);
    ri_y.resize(m);
    returnRandomIndices(ri_x, ri_y, rows, cols, m, seed);

    for (int i = 11, j = 0; i < ri_x.size() + 11, j < ri_x.size(); i++, j++) {
        sampled_mat.at<cv::Vec3b>(ri_x[i - 11], ri_y[i - 11]) = encrypted_img.at<cv::Vec3b>(i);
        sampled_mask.at<cv::Vec3b>(ri_x[j], ri_y[j]) = cv::Vec3b(1, 1, 1);
    }
}

cv::Size decrypt_image::get_org_size() {
    return org_size;
}