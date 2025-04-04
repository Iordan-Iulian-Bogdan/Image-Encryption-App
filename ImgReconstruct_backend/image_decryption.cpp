#include "image_decryption.hpp"
#include "image_encryption.hpp"

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
    
    for (int i = 11; i < ri_x.size() + 11 - 32; i += 32) {
        // I think this helps with cache hits
        for (int j = 0; j < 32; ++j) {
           sampled_mat.at<cv::Vec3b>(ri_x[i - 11 + j], ri_y[i - 11 + j]) = encrypted_img.at<cv::Vec3b>(i + j);
           masked_mat.at<cv::Vec3b>(ri_x[i - 11 + j], ri_y[i - 11 + j]) = cv::Vec3b(1, 1, 1);
        }
    }
    
}

cv::Size decrypt_image::get_org_size() {
    return org_size;
}

//decrypts tiles in parallel
void decrypt_tiles(int num_threads, std::vector<std::vector<cv::Mat>>& mats_in, std::vector<std::vector<indices>> indices,
    std::vector<std::vector<cv::Mat>>& mats_out, std::vector<std::string> processing_order, int iterations, cv::Size tile_size, float coef) {

    // we use a reference image as the initial solution
    // this helps speed up convergence
    const std::vector<cv::Mat> ref = createRefSolutions(tile_size.width, tile_size.height);

#pragma omp parallel for num_threads(num_threads) schedule(dynamic)
    for (int k = 0; k < mats_in[0].size() * mats_in[0].size(); k++) {
        std::vector<std::string> splitText = splitString(processing_order[k], '_');
        int i = std::stoi(splitText[0]);
        int j = std::stoi(splitText[1]);

        // it's faster to make a copy of the original reference rather than create one every time
        cv::Mat copied[3];

        for (int i = 0; i < ref.size(); ++i) {
            copied[i] = ref[i].clone();
        }

        decrypt_image dimgs = decrypt_image(mats_in[i][j]);
        dimgs.decrypt(copied, indices[i][j].ri_x_g, indices[i][j].ri_y_g, iterations, coef, mats_out[i][j]);
    }

}

void decrypt_image_tiled(const int& tiles, const int& overlap, const int& iterations, const int& nun_threads, const float& coef, const std::string& input_path, const std::string& output_path, const std::string& password) {

    cv::Mat encrypted_img_g = cv::imread(input_path, cv::IMREAD_COLOR);

    cv::Mat sampled_mat;
    cv::Mat masked_mat;
    decrypt_image dimgs = decrypt_image(encrypted_img_g);
    cv::Size org_size = dimgs.get_org_size();
    int num_samples = 0;

    // extracting all the sampled pixels (sampled_mat) and thier coordinates (masked_mat)
    dimgs.get_sampled_mat(password, sampled_mat, masked_mat);

    std::string windowName = "ImageWindow";

    std::vector<std::vector<cv::Mat>> reconfigured_cropped_out;
    int N_reconfigured = tiles;
    reconfigured_cropped_out.resize(N_reconfigured, std::vector<cv::Mat>(N_reconfigured));
    std::vector<std::vector<cv::Mat>> reconfigured_cropped_mats_in;

    std::vector<std::vector<indices>> indices_reconfigured(N_reconfigured, std::vector<indices>(N_reconfigured));
    std::vector<std::vector<TileCoord>> coordinates;

    // we split the sampled image into tiles
    // every tile overlaps with other neighboring tiles  
    // this is done because otherwise the titles wont quite match with eachother along the borders
    // this becomes more obvious as the numer of samples goes down, aka more compression 
    splitImageIntoTiles(sampled_mat, reconfigured_cropped_mats_in, coordinates, N_reconfigured, overlap);

    std::vector<std::vector<std::string>> matrix(N_reconfigured, std::vector<std::string>(N_reconfigured));

    int k = 0;
    for (int i = 0; i < N_reconfigured; i++) {
        for (int j = 0; j < N_reconfigured; j++) {
            matrix[i][j] = std::to_string(i) + "_" + std::to_string(j);
        }
    }

    // we will decrypt the tiles in a spiral order from the middle
    // this is done just beacuse it looks "better" this way
    const std::vector<std::string> result = spiralOrder(matrix);

    cv::Size tile_size;
    int estimated_number_of_samples = 0;

    for (int i = 0; i < N_reconfigured; i++) {
        for (int j = 0; j < N_reconfigured; j++) {

            std::vector<int> ri_x_g, ri_y_g;

            // reserving space to avoid realocations
            ri_x_g.reserve(estimated_number_of_samples);
            ri_y_g.reserve(estimated_number_of_samples);

            int base_row = i * (reconfigured_cropped_mats_in[i][j].rows - overlap);
            int base_col = j * (reconfigured_cropped_mats_in[i][j].cols - overlap);

            for (int q = 0; q < reconfigured_cropped_mats_in[i][j].rows; q++) {
                for (int k = 0; k < reconfigured_cropped_mats_in[i][j].cols; k++) {
                    if (masked_mat.at<cv::Vec3b>(base_row + q, base_col + k) == cv::Vec3b(1, 1, 1)) {
                        ri_x_g.push_back(q);
                        ri_y_g.push_back(k);
                    }
                }
            }

            // we use the the number of previous samples as estimations, 
            // all tiles are going to have roughly the same number of samples
            estimated_number_of_samples = ri_x_g.size();

            reconfigured_cropped_out[i][j] = cv::Mat::zeros(reconfigured_cropped_mats_in[i][j].rows, reconfigured_cropped_mats_in[i][j].cols, CV_8UC3);
            encrypt_image img(reconfigured_cropped_mats_in[i][j]);

            indices_reconfigured[i][j] = { ri_x_g, ri_y_g };
            img.encrypt(ri_x_g, ri_y_g);
            reconfigured_cropped_mats_in[i][j] = img.get_mat();
            tile_size = img.get_size();
        }
    }

    cv::Mat reconstructed = cv::Mat::zeros(sampled_mat.rows, sampled_mat.cols, CV_8UC3);

    // this updates and displays the image as it is being decrypted
    display disp;
    disp.display_image(windowName, reconstructed, coordinates, reconfigured_cropped_out);

    std::thread decrypt_tiles_thread;
    decrypt_tiles_thread = std::thread(decrypt_tiles, nun_threads, std::ref(reconfigured_cropped_mats_in), std::ref(indices_reconfigured),
        std::ref(reconfigured_cropped_out), std::ref(result), iterations, tile_size, coef);
    decrypt_tiles_thread.join();

    disp.stop_display();

    reconstructed = reconstructImage(reconfigured_cropped_out, coordinates);

    // blending the overlapping tiles together for better quality
    cv::Mat blended = blendTilesWithImage(reconfigured_cropped_out, coordinates, reconstructed, 0.5f);
    cv::resize(blended, blended, org_size);
    cv::imwrite(output_path, blended);
}