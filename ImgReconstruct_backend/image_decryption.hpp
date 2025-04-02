#ifndef DECRYPTION_IMAGE_HPP
#define DECRYPTION_IMAGE_HPP

#include "CS_encryption.hpp"

class decrypt_image : CSencryption {
private:

public:
    decrypt_image(const std::string input_path);

    decrypt_image() {}

    decrypt_image(cv::Mat input);

    void decrypt(cv::Mat ref[3], const std::vector<int>& ri_x_g, const std::vector<int>& ri_y_g, const int num_iterations, const float coef, cv::Mat& out);

    void get_mat(cv::Mat& dest);

    cv::Mat get_mat();

    void writeDecryptedImageToDisk(const std::string output_path, bool remove_noise = false, bool noise_level = 3);

    void get_sampled_mat(const std::string& password, cv::Mat& sampled_mat, cv::Mat& masked_mat);

    cv::Size get_org_size();
};

#endif