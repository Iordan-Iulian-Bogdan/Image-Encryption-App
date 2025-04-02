#ifndef ENCRYPT_IMAGE_HPP
#define ENCRYPT_IMAGE_HPP

#include "CS_encryption.hpp"

class encrypt_image : CSencryption
{
private:
    float bm = 0.0f;

public:
    encrypt_image(std::string input_path);

    encrypt_image(const cv::Mat& input);

    encrypt_image() {}

    void get_mat(cv::Mat& dest);

    cv::Mat get_mat();

    void encrypt(const float& pixel_p, const std::string& password);

    void encrypt(const std::vector<int>& ri_x_g, const std::vector<int>& ri_y_g);

    void writeEncryptedImageToDisk(const std::string& output_path);

    cv::Size get_size(); 

private:
};

#endif