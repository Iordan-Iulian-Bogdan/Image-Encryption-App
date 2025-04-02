#include "lbfgs.hpp"
#include "image_encryption.hpp"
#include "image_decryption.hpp"

//decrypts tiles in parallel

static void decrypt_tiles(int num_threads, std::vector<std::vector<cv::Mat>>& mats_in, std::vector<std::vector<indices>> indices,
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

static void decrypt_image_tiled(const cv::Mat& encrypted_img_g, const int& tiles, const int& overlap, const int& iterations, const int& nun_threads, const float& coef, std::string location, const std::string& password) {
    
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
    cv::imwrite(location, blended);
}

static cv::Mat encrypt_image_tiled(const cv::Mat& input_image, const float& compression_ratio, const std::string& password) {

    encrypt_image encrypt_img(input_image);
    encrypt_img.encrypt(compression_ratio, password);
    return encrypt_img.get_mat();
}

int main(int argc, char** argv)
{
    const std::string password = "5v48d254h33432";

    std::vector<const char*> inputs(4);
    inputs[0] = "IMG_3690.png";
    inputs[1] = "IMG_1297.png";
    inputs[2] = "IMG_0007.png";
    inputs[3] = "IMG_9321.png";
    //cv::Mat img = cv::imread(inputs[0], cv::IMREAD_COLOR);
    //cv::Mat encrypted_img_g = encrypt_image_tiled(img, 0.5f, password);
    //cv::imwrite("encrypted_img_g.png", encrypted_img_g);
    cv::Mat encrypted_img_g = cv::imread("encrypted_img_g.png", cv::IMREAD_COLOR);
    decrypt_image_tiled(encrypted_img_g, 24, 48, 20, 24, 0.035f, "decrypted_image.png", password);
    
    return 0;
}