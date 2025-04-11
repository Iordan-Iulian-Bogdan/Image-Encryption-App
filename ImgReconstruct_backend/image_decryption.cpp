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

float decrypt_image::get_compression_ratio() {
    return float(m) / (rows * cols);
}

cv::Size decrypt_image::get_org_size() {
    return org_size;
}

// decrypts tiles in parallel
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

int decrypt_image_tiled(
    const std::string& input_path,
    const std::string& output_path,
    const std::string& password,
    int parameters_type,
    int num_tiles,
    int overlap, 
    int iterations, 
    int nun_threads, 
    float coef
) {
    try {
        if (overlap < 24 || overlap > 96) {
            throw std::runtime_error("Overlap is outside the acceptable range of (24, 96)");
        }

        if (num_tiles < 24) {
            throw std::runtime_error("Number of tiles is less than 24");
        }

        if (coef < 0.01f || coef > 0.05f) {
            throw std::runtime_error("Coef is outside of the acceptable range of (0.01, 0.05)");
        }

        if (nun_threads < 1) {
            throw std::runtime_error("Number of threads is less than 1");
        }

        if (parameters_type != AUTO_PARAM && parameters_type != MANUAL_PARAM) {
            throw std::runtime_error("Parameters type is incorrect");
        }
    }
    catch (const std::runtime_error& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }

    //int N_reconfigured = tiles;
    std::string windowName = input_path;
    std::vector<std::vector<cv::Mat>> encrypted_image_tiles;
    std::vector<std::vector<cv::Mat>> decrypted_image_tiles;
    decrypted_image_tiles.resize(num_tiles, std::vector<cv::Mat>(num_tiles));
    std::vector<std::vector<indices>> indices_reconfigured(num_tiles, std::vector<indices>(num_tiles));
    std::vector<std::vector<TileCoord>> coordinates;
    cv::Mat sampled_mat;
    cv::Mat masked_mat;
    cv::Mat encrypted_img_g = cv::imread(input_path, cv::IMREAD_COLOR);
    decrypt_image dimgs = decrypt_image(encrypted_img_g);
    cv::Size org_size = dimgs.get_org_size();
    if (parameters_type == AUTO_PARAM) {
        iterations = (1.0f / dimgs.get_compression_ratio()) * 10;
       
        if (dimgs.get_compression_ratio() < 0.5f) {
            coef = 0.05f;
        }
        else
        {
            coef = (1.0f / dimgs.get_compression_ratio()) * 0.01875f;
        }
        num_tiles = 48;
        overlap = 48;
        nun_threads = omp_get_max_threads();
    }
    // extracting all the sampled pixels (sampled_mat) and thier coordinates (masked_mat)
    dimgs.get_sampled_mat(password, sampled_mat, masked_mat);

    // we split the sampled image into tiles
    // every tile overlaps with other neighboring tiles  
    // this is done because otherwise the titles wont quite match with eachother along the borders
    // this becomes more obvious as the numer of samples goes down, aka more compression 
    splitImageIntoTiles(sampled_mat, encrypted_image_tiles, coordinates, num_tiles, overlap);

    // we will decrypt the tiles in a spiral order from the middle
    // this is done just beacuse it looks "better" this way
    const std::vector<std::string> processing_order = spiralOrder(num_tiles);

    cv::Size tile_size;
    int estimated_number_of_samples = 0;

    for (int i = 0; i < num_tiles; i++) {
        for (int j = 0; j < num_tiles; j++) {

            std::vector<int> ri_x_g, ri_y_g;

            // reserving space to avoid realocations
            ri_x_g.reserve(estimated_number_of_samples);
            ri_y_g.reserve(estimated_number_of_samples);

            int base_row = i * (encrypted_image_tiles[i][j].rows - overlap);
            int base_col = j * (encrypted_image_tiles[i][j].cols - overlap);

            for (int q = 0; q < encrypted_image_tiles[i][j].rows; q++) {
                for (int k = 0; k < encrypted_image_tiles[i][j].cols; k++) {
                    if (masked_mat.at<cv::Vec3b>(base_row + q, base_col + k) == cv::Vec3b(1, 1, 1)) {
                        ri_x_g.push_back(q);
                        ri_y_g.push_back(k);
                    }
                }
            }

            // we use the the number of previous samples as estimations, 
            // all tiles are going to have roughly the same number of samples
            estimated_number_of_samples = ri_x_g.size();

            decrypted_image_tiles[i][j] = cv::Mat::zeros(encrypted_image_tiles[i][j].rows, encrypted_image_tiles[i][j].cols, CV_8UC3);
            encrypt_image img(encrypted_image_tiles[i][j]);

            indices_reconfigured[i][j] = { ri_x_g, ri_y_g };
            img.encrypt(ri_x_g, ri_y_g);
            encrypted_image_tiles[i][j] = img.get_mat();
            tile_size = img.get_size();
        }
    }

    cv::Mat reconstructed = cv::Mat::zeros(sampled_mat.rows, sampled_mat.cols, CV_8UC3);

    // this updates and displays the image as it is being decrypted
    display disp;
    disp.display_image(windowName, reconstructed, coordinates, decrypted_image_tiles);

    std::thread decrypt_tiles_thread;
    decrypt_tiles_thread = std::thread(decrypt_tiles, nun_threads, std::ref(encrypted_image_tiles), std::ref(indices_reconfigured),
        std::ref(decrypted_image_tiles), std::ref(processing_order), iterations, tile_size, coef);
    decrypt_tiles_thread.join();

    disp.stop_display();

    reconstructed = reconstructImage(decrypted_image_tiles, coordinates);

    // blending the overlapping tiles together for better quality
    cv::Mat blended = blendTilesWithImage(decrypted_image_tiles, coordinates, reconstructed, 0.5f);
    cv::resize(blended, blended, org_size);
    cv::imwrite(output_path, blended);

    return 0;
}