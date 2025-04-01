#ifndef DECRYPTION_IMAGE_HPP
#define DECRYPTION_IMAGE_HPP

#include "CS_encryption.hpp"

class decrypt_image : CSencryption {
private:
    cv::Mat c[3];
    std::vector<float> optimal_values;

public:
    decrypt_image(std::string input_path);

    decrypt_image() {}

    decrypt_image(cv::Mat input);

    void decrypt(std::vector<cv::Mat> ref, std::vector<int>& ri_x_g, std::vector<int>& ri_y_g, int num_iterations, float coef, cv::Mat& out);

    void get_mat(cv::Mat& dest);

    cv::Mat get_mat();

    void writeDecryptedImageToDisk(std::string output_path, bool remove_noise = false, bool noise_level = 3);


    cv::Mat get_sampled_mat(std::string password, cv::Mat& sampled_mat, cv::Mat& masked_mat);

    cv::Mat get_sampled_mask(int seed);

    void get_sampled_mask_mats(std::string password, cv::Mat& sampled_mat, cv::Mat& sampled_mask);

    cv::Size get_org_size();
};

#endif