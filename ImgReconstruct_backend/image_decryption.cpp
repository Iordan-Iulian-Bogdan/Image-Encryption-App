#include "image_decryption.hpp"

decrypt_image::decrypt_image(const std::string input_path) {
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

    org_size.height = (float)std::stoi(splitText[3]);
    org_size.width = (float)std::stoi(splitText[4]);
}

void decrypt_image::decrypt(cv::Mat ref[3], const std::vector<int>& ri_x_g, const std::vector<int>& ri_y_g, const int num_iterations, const float coef, cv::Mat& out) {

    reconstruct_color_channel(encrypted_img, 0, coef, rows, cols, ri_x_g, ri_y_g, num_iterations, ref[0]);
    reconstruct_color_channel(encrypted_img, 1, coef, rows, cols, ri_x_g, ri_y_g, num_iterations, ref[1]);
    reconstruct_color_channel(encrypted_img, 2, coef, rows, cols, ri_x_g, ri_y_g, num_iterations, ref[2]);

    cv::merge(ref, 3, out);
    out.convertTo(out, CV_8UC3);
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


void decrypt_image::get_sampled_mat(const std::string& password, cv::Mat& sampled_mat, cv::Mat& masked_mat) {
    sampled_mat = cv::Mat(rows, cols, CV_8UC3);
    masked_mat = cv::Mat(rows, cols, CV_8UC3);
    ri_x.resize(m);
    ri_y.resize(m);
    returnRandomIndices(ri_x, ri_y, rows, cols, m, password);

    for (int i = 11; i < ri_x.size() + 11 - 8; i = i + 8) {
        sampled_mat.at<cv::Vec3b>(ri_x[i - 11], ri_y[i - 11]) = encrypted_img.at<cv::Vec3b>(i);
        masked_mat.at<cv::Vec3b>(ri_x[i - 11], ri_y[i - 11]) = cv::Vec3b(1, 1, 1);

        sampled_mat.at<cv::Vec3b>(ri_x[i + 1 - 11], ri_y[i + 1 - 11]) = encrypted_img.at<cv::Vec3b>(i + 1);
        masked_mat.at<cv::Vec3b>(ri_x[i + 1 - 11], ri_y[i + 1 - 11]) = cv::Vec3b(1, 1, 1);

        sampled_mat.at<cv::Vec3b>(ri_x[i + 2 - 11], ri_y[i + 2 - 11]) = encrypted_img.at<cv::Vec3b>(i + 2);
        masked_mat.at<cv::Vec3b>(ri_x[i + 2 - 11], ri_y[i + 2 - 11]) = cv::Vec3b(1, 1, 1);

        sampled_mat.at<cv::Vec3b>(ri_x[i + 3 - 11], ri_y[i + 3 - 11]) = encrypted_img.at<cv::Vec3b>(i + 3);
        masked_mat.at<cv::Vec3b>(ri_x[i + 3 - 11], ri_y[i + 3 - 11]) = cv::Vec3b(1, 1, 1);

        sampled_mat.at<cv::Vec3b>(ri_x[i + 4 - 11], ri_y[i + 4 - 11]) = encrypted_img.at<cv::Vec3b>(i + 4);
        masked_mat.at<cv::Vec3b>(ri_x[i + 4 - 11], ri_y[i + 4 - 11]) = cv::Vec3b(1, 1, 1);

        sampled_mat.at<cv::Vec3b>(ri_x[i + 5 - 11], ri_y[i + 5 - 11]) = encrypted_img.at<cv::Vec3b>(i + 5);
        masked_mat.at<cv::Vec3b>(ri_x[i + 5 - 11], ri_y[i + 5 - 11]) = cv::Vec3b(1, 1, 1);

        sampled_mat.at<cv::Vec3b>(ri_x[i + 6 - 11], ri_y[i + 6 - 11]) = encrypted_img.at<cv::Vec3b>(i + 6);
        masked_mat.at<cv::Vec3b>(ri_x[i + 6 - 11], ri_y[i + 6 - 11]) = cv::Vec3b(1, 1, 1);

        sampled_mat.at<cv::Vec3b>(ri_x[i + 7 - 11], ri_y[i + 7 - 11]) = encrypted_img.at<cv::Vec3b>(i + 7);
        masked_mat.at<cv::Vec3b>(ri_x[i + 7 - 11], ri_y[i + 7 - 11]) = cv::Vec3b(1, 1, 1);
    }
}

cv::Size decrypt_image::get_org_size() {
    return org_size;
}