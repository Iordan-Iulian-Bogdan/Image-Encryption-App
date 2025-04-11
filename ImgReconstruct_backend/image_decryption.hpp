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

    float get_compression_ratio();

    cv::Size get_org_size();
};

void decrypt_tiles(int num_threads, std::vector<std::vector<cv::Mat>>& mats_in, std::vector<std::vector<indices>> indices,
    std::vector<std::vector<cv::Mat>>& mats_out, std::vector<std::string> processing_order, int iterations, cv::Size tile_size, float coef);
/** @brief decrypts a given image and writes the result to disk
@param input_path:  path to input image
@param output_path:  path to output image
@param password:  password to be used for encryption and decryption
@param parameters_type: use AUTO_PARAM for automatically choosing parameters or MANUAL_PARAM for fine tuning
@param num_tiles : number of tiles to be processed
@param overlap : how many pixels should the tiles overlap, bigger number may result in better quality
@param iterations : number of iterations, higher value may result in better quality
@param nun_threads : number of CPU threads
@param coef : coeficient used for the solver, higher value should be used for a more compressed image
*/
int decrypt_image_tiled(const std::string& input_path, const std::string& output_path, const std::string& password, int parameters_type = AUTO_PARAM, int num_tiles = 24, int overlap = 48, int iterations = 20, int nun_threads = 8, float coef = 0.01f);

#endif