#ifndef ENCRYPT_IMAGE_HPP
#define ENCRYPT_IMAGE_HPP

#include "CS_encryption.hpp"

class encrypt_image : CSencryption
{
private:
    float bm;

public:
    encrypt_image(std::string input_path, bool remove_noise = false, int noise_level = 3);

    encrypt_image(cv::Mat input, bool remove_noise = false, int noise_level = 3);

    encrypt_image() {}

    void get_mat(cv::Mat& dest);

    cv::Mat get_mat();

    cv::Mat get_sampled_mat();

    cv::Mat get_sampled_mask();

    void encrypt(float pixel_p = 0.3, int seed = 1);

    void encrypt(std::vector<int> ri_x_g, std::vector<int> ri_y_g);

    void writeEncryptedImageToDisk(std::string output_path);

    cv::Size get_size(); 

private:
};

#endif