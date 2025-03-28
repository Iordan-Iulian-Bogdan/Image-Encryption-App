#include "lbfgs.hpp"
#include "image_encryption.hpp"
#include "image_decryption.hpp"

std::vector<std::vector<cv::Mat>> reconfigured_cropped_out;

std::mutex mtx;
std::mutex mtx_decryption;
bool g_dsp = true;
std::vector<int> passwords;

void process_1(int num_threads, int pass, std::vector<std::vector<cv::Mat>>& mats_in, std::vector<std::vector<indices>> indices,
    std::vector<std::vector<cv::Mat>>& mats_out, std::vector<std::string> processing_order, int iterations, cv::Size tile_size) {
    
    std::vector<cv::Mat> ref = createRefDCT(tile_size.height, tile_size.width);

    #pragma omp parallel for num_threads(num_threads) schedule(dynamic)
    for (int k = 0; k < mats_in[0].size() * mats_in[0].size(); k++) {
        std::vector<std::string> splitText = splitString(processing_order[k], '_');
        int i = std::stoi(splitText[0]);
        int j = std::stoi(splitText[1]);

        std::vector<cv::Mat> copied(3);

        for (int i = 0; i < ref.size(); ++i) {
            copied[i] = ref[i].clone();
        }

        decrypt_image* dimgs = new decrypt_image;
        *dimgs = decrypt_image(mats_in[i][j]);
        dimgs->decrypt(copied, indices[i][j].ri_x_g, indices[i][j].ri_y_g, iterations, 0.05, false);
        dimgs->get_mat(mats_out[i][j]);
        delete(dimgs);

        mats_in[i][j].deallocate();
        indices[i][j].ri_x_g.resize(0);
        indices[i][j].ri_y_g.resize(0);
    }

}

void display_output(std::string windowName, cv::Mat& temp, std::vector<std::vector<TileCoord>>& coordinates) {
    while (g_dsp) {
        cv::waitKey(10);
        temp = reconstructImage(reconfigured_cropped_out, coordinates);
        updateImage(windowName, temp);
        cv::waitKey(1);
    }
}

void decrypt_image_tiled(cv::Mat encrypted_img_g, int tiles, int overlap, int iterations, int nun_threads) {
    cv::Mat sampled_mat;
    decrypt_image dimgs = decrypt_image(encrypted_img_g);
    cv::Size org_size = dimgs.get_org_size();

    sampled_mat = dimgs.get_sampled_mat(1);

    std::string windowName = "ImageWindow";


    int N_reconfigured = tiles;
    reconfigured_cropped_out.resize(N_reconfigured, std::vector<cv::Mat>(N_reconfigured));
    std::vector<std::vector<cv::Mat>> reconfigured_cropped_mats_in;

    std::vector<std::vector<indices>> indices_reconfigured(N_reconfigured, std::vector<indices>(N_reconfigured));
    std::vector<std::vector<TileCoord>> coordinates;

    splitImageIntoTiles(sampled_mat, reconfigured_cropped_mats_in, coordinates, N_reconfigured, overlap);

    sampled_mat.deallocate();
    std::vector<std::vector<std::string>> matrix(N_reconfigured, std::vector<std::string>(N_reconfigured));
    passwords.resize(N_reconfigured * N_reconfigured);

    for (int i = 0; i < N_reconfigured * N_reconfigured; i++) {
        passwords[i] = i;
    }

    int k = 0;
    for (int i = 0; i < N_reconfigured; i++) {
        for (int j = 0; j < N_reconfigured; j++) {
            matrix[i][j] = std::to_string(i) + "_" + std::to_string(j);
        }
    }

    std::vector<std::string> result = spiralOrder(matrix);
    cv::Size tile_size;

    #pragma omp parallel for num_threads(nun_threads) schedule(dynamic)
    for (int i = 0; i < N_reconfigured; i++) {
        for (int j = 0; j < N_reconfigured; j++) {
            
            std::vector<int> ri_x_g, ri_y_g;

            for (int q = 0; q < reconfigured_cropped_mats_in[i][j].rows; q++) {
                for (int k = 0; k < reconfigured_cropped_mats_in[i][j].cols; k++) {
                    if (reconfigured_cropped_mats_in[i][j].at<cv::Vec3b>(q, k) != cv::Vec3b(255, 255, 255)) {
                        ri_x_g.push_back(q);
                        ri_y_g.push_back(k);
                    }
                }
            }
            reconfigured_cropped_out[i][j] = cv::Mat::zeros(reconfigured_cropped_mats_in[i][j].rows, reconfigured_cropped_mats_in[i][j].cols, CV_8UC3);
            encrypt_image img(reconfigured_cropped_mats_in[i][j], false);

            indices_reconfigured[i][j] = { ri_x_g, ri_y_g };
            img.encrypt(ri_x_g, ri_y_g);
            img.get_mat(reconfigured_cropped_mats_in[i][j]);
            tile_size = img.get_size();
        }
    }

    cv::Mat reconstructed = cv::Mat::zeros(sampled_mat.rows, sampled_mat.cols, CV_8UC3);
    std::thread CPU_display_output;
    CPU_display_output = std::thread(display_output, windowName, std::ref(reconstructed), std::ref(coordinates));
    
    std::thread CPUProcessing1;
    CPUProcessing1 = std::thread(process_1, nun_threads, 1, std::ref(reconfigured_cropped_mats_in), std::ref(indices_reconfigured),
        std::ref(reconfigured_cropped_out), std::ref(result), iterations, tile_size);
    CPUProcessing1.join();

    g_dsp = false;
    CPU_display_output.join();
    cv::Mat blended = blendTilesWithImage(reconfigured_cropped_out, coordinates, reconstructed, 0.5f);
    cv::resize(blended, blended, org_size);
    cv::imwrite("blended.png", blended);
}

cv::Mat encrypt_image_tiled(cv::Mat input_image, float compression_ratio = 0.5f) {

    encrypt_image encrypt_img(input_image, false);
    encrypt_img.encrypt(compression_ratio, 1);
    return encrypt_img.get_mat();
}

int main(int argc, char* argv)
{
    
    std::vector<const char*> inputs(4);
    inputs[0] = "IMG_3690.png";
    inputs[1] = "IMG_1297.png";
    inputs[2] = "IMG_0007.png";
    inputs[3] = "IMG_9321.png";

    //cv::Mat img = cv::imread(inputs[0], cv::IMREAD_COLOR);
    //cv::Mat encrypted_img_g = encrypt_image_tiled(img, 0.33f);
    //cv::imwrite("encrypted_img_g.png", encrypted_img_g);

    cv::Mat encrypted_img_g = cv::imread("encrypted_img_g.png", cv::IMREAD_COLOR);
    decrypt_image_tiled(encrypted_img_g, 24, 48, 25, 24);

    return 0;
}