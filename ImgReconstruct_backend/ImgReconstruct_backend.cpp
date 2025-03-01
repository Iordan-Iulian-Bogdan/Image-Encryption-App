#include <stdio.h>
#include "lbfgs.h"
#include "lbfgs.h"
#include <mutex>
#include <vector>
#include <string>
#include "sparseImageEncryption.h"
#include <chrono>
#include <stdlib.h>

std::mutex mtx;
bool g_dsp = true;
std::vector<cv::Mat> mats_out;
// Function to update the image
void updateImage(const std::string& windowName, const cv::Mat& newImage) {
    cv::Mat aux = newImage.clone();
    cv::resize(aux, aux, cv::Size(newImage.cols / 3, newImage.rows / 3));
    cv::imshow(windowName, aux);
}

void updateImage2(const std::string& windowName, const cv::Mat& newImage) {
    cv::Mat aux = newImage.clone();
    cv::resize(aux, aux, cv::Size(newImage.cols * 2, newImage.rows * 2));
    cv::imshow(windowName, aux);
}

std::vector<cv::Mat> splitMat(cv::Mat& image, int M, int N)
{
    int width = image.cols / M;
    int height = image.rows / N;
    int width_last_column = width + (image.cols % width);
    int height_last_row = height + (image.rows % height);

    std::vector<cv::Mat> result;

    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < M; ++j)
        {
            cv::Rect roi(width * j,
                height * i,
                (j == (M - 1)) ? width_last_column : width,
                (i == (N - 1)) ? height_last_row : height);

            result.push_back(image(roi));
        }
    }

    return result;
}

cv::Mat stitchMats(const std::vector<cv::Mat>& mats, int N, int M) {
    // Check if the vector is empty
    if (mats.empty()) {
        std::cout << "Error: The vector of mats is empty." << std::endl;
        return cv::Mat();
    }

    // Get the size of each sub-matrix
    int subRows = mats[0].rows;
    int subCols = mats[0].cols;

    // Create the output matrix with the appropriate size
    cv::Mat stitchedMat = cv::Mat::zeros(N * subRows, M * subCols, mats[0].type());

    // Iterate through the vector and copy each sub-matrix to the correct position
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < M; ++j) {
            int index = i * M + j;
            if (index >= mats.size()) {
                std::cout << "Error: Not enough mats to fill the output matrix." << std::endl;
                return cv::Mat();
            }
            // Define the region of interest in the output matrix
            cv::Rect roi(j * subCols, i * subRows, subCols, subRows);
            // Copy the sub-matrix to the region of interest
            mats[index].copyTo(stitchedMat(roi));
        }
    }

    return stitchedMat;
}

cv::Mat stitchMats_alt(const std::vector<cv::Mat>& mats, int N, int M) {
    // Check if the vector is empty
    if (mats.empty()) {
        std::cout << "Error: The vector of mats is empty." << std::endl;
        return cv::Mat();
    }

    // Calculate the total number of images
    int numImages = mats.size();

    // Calculate the number of rows and columns in the grid
    int rows = std::ceil(static_cast<float>(numImages) / M);
    int cols = std::min<int>(numImages, M);

    // Create the output matrix with the appropriate size
    int totalRows = 0, totalCols = 0;
    for (const auto& mat : mats) {
        totalRows = std::max<int>(totalRows, mat.rows);
        totalCols = std::max<int>(totalCols, mat.cols);
    }
    cv::Mat stitchedMat = cv::Mat::zeros(rows * totalRows, cols * totalCols, mats[0].type());

    // Iterate through the vector and copy each sub-matrix to the correct position
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            int index = i * cols + j;
            if (index >= mats.size()) {
                std::cout << "Error: Not enough mats to fill the output matrix." << std::endl;
                return cv::Mat();
            }
            // Define the region of interest in the output matrix
            cv::Rect roi(j * totalCols, i * totalRows, mats[index].cols, mats[index].rows);
            // Copy the sub-matrix to the region of interest
            mats[index].copyTo(stitchedMat(roi));
        }
    }

    return stitchedMat;
}


class CSencryption {
protected:
    cv::Mat decrypted_img, encrypted_img, input_img;
    int rows, cols, m, n;
    std::vector<int> ri_x, ri_y;

    void returnRandomIndices(std::vector<int>& ri_x, std::vector<int>& ri_y, int xm, int ym, int numOfIndices, int seed = 1) {
        std::random_device e;
        std::default_random_engine generator(e());
        generator.seed(seed);
        std::uniform_int_distribution<> disx(0, xm - 1);
        std::uniform_int_distribution<> disy(0, ym - 1);

        cv::Mat xy(xm, ym, CV_8U, cv::Scalar(0));

        int k = 0;

        while (k < numOfIndices) {
            int x = disx(generator);
            int y = disy(generator);

            if (xy.at<unsigned char>(x, y) == 1) {
                continue;
            }
            else {
                ri_x[k] = x;
                ri_y[k] = y;
                xy.at<unsigned char>(x, y) = 1;
                k++;
            }
        }
    }
};

inline void updateAxb2AndComputeFx(float* x_copy, int* ri_x, int* ri_y,
    float* Axb2_vec, const float* b, int cols, float& fx, int n) {
    fx = 0.0f;

    // Process elements
    for (int i = 0; i < n; ++i) {
        int idx = ri_x[i] * cols + ri_y[i];
        float diff = x_copy[idx] - b[i];
        fx += diff * diff;
        Axb2_vec[idx] = diff;
    }
}


inline void eval_g(float* Axb2, float* g, int n) {
    // Process multiples of 16
    __m512 scalar = _mm512_set1_ps(2.0f);
    int i = 0;
    for (; i <= n - 16; i += 16) {
        __m512 vecData = _mm512_loadu_ps(&Axb2[i]);  // load 16 floats
        _mm512_storeu_ps(&g[i], _mm512_mul_ps(vecData, scalar));  // store 16 floats
    }

    // Process remaining elements
    for (; i < n; ++i) {
        g[i] = Axb2[i] * 2.0f;
    }
}

inline void copy_x(float* x_copy, float* x, float* Axb2_vec, int n) {
    __m512 factor = _mm512_set1_ps(0.0f);
    // Process multiples of 16
    int i = 0;
    for (; i <= n - 16; i += 16) {
        __m512 vecData = _mm512_loadu_ps(&x[i]);  // load 16 floats from vector
        _mm512_storeu_ps(&x_copy[i], vecData);  // store 16 floats to array
        _mm512_storeu_ps(&Axb2_vec[i], factor);
    }

    // Process remaining elements
    for (; i < n; ++i) {
        x_copy[i] = x[i];
        Axb2_vec[i] = 0.0f;
    }
}


void write_imgout(std::vector<cv::Mat>& mats_out, int tile_index, int color_c, cv::Mat Ax) {
    std::vector<cv::Mat> dst_channels;
    mtx.lock();
    cv::split(mats_out[tile_index].clone(), dst_channels);
    int channel_to_copy = color_c;
    cv::Mat aux_c = Ax;
    aux_c = aux_c * 255;
    aux_c.convertTo(aux_c, CV_8U);
    dst_channels[channel_to_copy] = aux_c;
    cv::merge(dst_channels, mats_out[tile_index]);
    mtx.unlock();
}

float evaluate(
    void* instance,
    const float* x,
    eval_data data,
    float* g,
    const int n,
    const float step
)
{
    float fx = 0;
    copy_x(data.x_copy, (float*)x, data.Axb2, n);
    cv::Mat Ax(data.rows, data.cols, CV_32F, data.x_copy);
    dct(Ax, Ax, cv::DCT_INVERSE);
    updateAxb2AndComputeFx(data.x_copy, data.ri_x, data.ri_y, data.Axb2, data.b, data.cols, fx, data.m);
    cv::Mat Axb2(data.rows, data.cols, CV_32F, data.Axb2);
    dct(Axb2, Axb2);
    eval_g(data.Axb2, g, n);

    return fx;
}

int progress(
    void* instance,
    const float* x,
    const float* g,
    const float fx,
    const float xnorm,
    const float gnorm,
    const float step,
    int n,
    int k,
    int ls
)
{
    printf("Iteration %d:\n", k);
    printf("  fx = %f, x[0] = %f, x[1] = %f\n", fx, x[0], x[1]);
    printf("  xnorm = %f, gnorm = %f, step = %f\n", xnorm, gnorm, step);
    printf("\n");

    return 0;
}

std::vector<cv::Mat> createRefDCT(int rows, int cols) {
    cv::Mat ref = cv::imread("ref.png", cv::IMREAD_COLOR);
    cv::resize(ref, ref, cv::Size(rows, cols));
    std::vector<cv::Mat> c;
    cv::split(ref, c);

    for (int i = 0; i < 3; i++) {
        c[i].convertTo(c[i], CV_32F);
        c[i] = c[i] / 255.0;
        cv::dct(c[i], c[i], 0);
        c[i] = c[i] / 10.0;
    }

    return c;
}

std::vector<cv::Mat> createRefDCT(int rows, int cols, cv::Mat ref) {
    cv::resize(ref, ref, cv::Size(rows, cols));
    std::vector<cv::Mat> c;
    cv::split(ref, c);

    for (int i = 0; i < 3; i++) {
        c[i].convertTo(c[i], CV_32F);
        c[i] = c[i] / 255.0;
        cv::dct(c[i], c[i], 0);
        c[i] = c[i];
    }

    return c;
}

void reconstruct_color_chanel(cv::Mat& out, cv::Mat& measurement, int k, float param_c, float optimal_value, int rows, int cols, std::vector<int>& ri_x, std::vector<int>& ri_y, int iterations, std::vector<cv::Mat> ref, bool opt, int tile_index, bool copy_result = true) {

    int n = rows * cols; // size of solution (size of vectorized image)

    float fx;
    // initializing the solution using the DCT of a natural image helps speed up convergence
    // since the first element has the largest magnitude we assign the avreage value of the three original DCT[0] values, this also helps speed up convergence
    if (opt) {
        //std::cout << std::endl << ref[k].at<float>(0) << " " << optimal_value << std::endl;
        ref[k].at<float>(0) = optimal_value;
    }


    /* Initialize the parameters for the optimization. */
    lbfgs_parameter_t param;
    lbfgs_parameter_init(&param);
    param.orthantwise_c = (float)param_c; // this tells lbfgs to do OWL-QN
    param.linesearch = LBFGS_LINESEARCH_BACKTRACKING;
    param.max_iterations = iterations;
    int lbfgs_ret;
    std::vector<float> b;
    //auto update_progress = progress;
    lbfgs_progress_t update_progress = NULL;
    eval_data data;
    std::vector<float> Axb2(n);
    std::vector<float> x_copy(n);

    for (int i = 11; i < ri_x.size() + 11; ++i) {
        b.push_back(measurement.at<cv::Vec3b>(i)[k] / 255.0);
    }

    data.b = b.data();
    data.Axb2 = Axb2.data();
    data.x_copy = x_copy.data();
    data.ri_x = ri_x.data();
    data.ri_x = ri_x.data();
    data.m = ri_x.size();
    data.ri_x = ri_x.data();
    data.ri_y = ri_y.data();
    data.rows = rows;
    data.cols = cols;

    lbfgs_ret = lbfgs(n, (float*)ref[k].data, data, &fx, evaluate, update_progress, NULL, &param);

    if (copy_result) {
        cv::Mat AtAxb2(rows, cols, CV_32F, (float*)ref[k].data);
        dct(AtAxb2, AtAxb2, cv::DCT_INVERSE);
        AtAxb2 = AtAxb2 * 255.0;
        out = AtAxb2.clone();
    }
}

std::vector<std::string> splitString(const std::string& str, char delimiter) {
    std::vector<std::string> result;
    std::string temp;
    for (char c : str) {
        if (c == delimiter) {
            if (!temp.empty()) {
                result.push_back(temp);
                temp.clear();
            }
        }
        else {
            temp.push_back(c);
        }
    }
    // Add the last substring if there is any
    if (!temp.empty()) {
        result.push_back(temp);
    }
    return result;
}

std::string removeCharacter(const std::string& str, char ch) {
    std::string result;
    for (char c : str) {
        if (c != ch) {
            result.push_back(c);
        }
    }
    return result;
}

void storeStringInColorMat(const std::string& text, cv::Mat& colorMat) {
    // Ensure the colorMat is large enough to hold the string
    int rows = (text.size() / 3) + 1;
    int cols = 1;
    colorMat = cv::Mat::zeros(rows, cols, CV_8UC3);

    // Encode the string into the Mat
    for (int i = 0; i < text.size(); ++i) {
        int row = i / 3;
        int channel = i % 3;
        colorMat.at<cv::Vec3b>(row, 0)[channel] = static_cast<uchar>(text[i]);
    }
}

std::string retrieveStringFromColorMat(const cv::Mat& colorMat) {
    std::string text;

    // Decode the Mat back into a string
    for (int i = 0; i < colorMat.rows; ++i) {
        for (int channel = 0; channel < 3; ++channel) {
            uchar value = colorMat.at<cv::Vec3b>(i, 0)[channel];
            if (value != 0) {
                text.push_back(static_cast<char>(value));
            }
        }
    }

    return text;
}

class encrypt_image : CSencryption 
{
private:
    float bm;

public:
    encrypt_image(std::string input_path, bool remove_noise = false, int noise_level = 3) {
        input_img = cv::imread(input_path, cv::IMREAD_COLOR);
        if (remove_noise) {
            cv::fastNlMeansDenoisingColored(input_img, input_img, noise_level);
        }
        rows = input_img.rows;
        cols = input_img.cols;
    }

    encrypt_image(cv::Mat input, bool remove_noise = false, int noise_level = 3) {
        input.copyTo(input_img);
        if (remove_noise) {
            cv::fastNlMeansDenoisingColored(input_img, input_img, noise_level);
        }
        rows = input_img.rows;
        cols = input_img.cols;
    }

    void get_mat(cv::Mat& dest) {
        encrypted_img.copyTo(dest);
    }

    void encrypt(float pixel_p = 0.3, int seed = 1) {
        bm = pixel_p;
        m = rows * cols * bm;
        int n = rows * cols;
        long optim = (long)analyze(input_img);
        std::vector<int>rand_ind_x(m), rand_ind_y(n);
        returnRandomIndices(rand_ind_x, rand_ind_y, rows, cols, m, seed);
        encrypted_img = cv::Mat(1, (int)(sqrt(m + 6) + 1) * (int)(sqrt(m + 6) + 1), CV_8UC3);

        std::string text = std::to_string(m) + "|" + std::to_string(rows) + "|" + std::to_string(cols) + "|" + std::to_string(optim); // Example string 
        std::string padding;

        int paddingSize = 32 - text.size();
        for (int i = 0; i < paddingSize; i++) {
            padding += "-";
        }
        text = padding + text;
        cv::Mat colorMat;
        storeStringInColorMat(text, colorMat); 
        std::string retrievedText = retrieveStringFromColorMat(colorMat);

        int i = 0;

        for (; i < colorMat.total(); i++) {
            encrypted_img.at<cv::Vec3b>(i) = colorMat.at<cv::Vec3b>(i);
        }

        int k = 0;

        for (; (i < encrypted_img.rows * encrypted_img.cols) && k < m - 1; i++) {
            encrypted_img.at<cv::Vec3b>(k + colorMat.total()) = input_img.at<cv::Vec3b>(rand_ind_x[k], rand_ind_y[k]);
            k++;
        }

        encrypted_img = encrypted_img.reshape(0, (int)sqrt(encrypted_img.total()));
    }

    void writeEncryptedImageToDisk(std::string output_path) {
        cv::imwrite(output_path, encrypted_img);
    }

private:

    float analyze(cv::Mat in) {
        cv::Mat p = in.clone();

        std::vector<cv::Mat> c;
        cv::split(p, c);

        #pragma omp parallel for num_threads(3) schedule(dynamic)
        for (int i = 0; i < 3; i++) {
            c[i].convertTo(c[i], CV_32F);
            c[i] = c[i] / 255.0;
            cv::dct(c[i], c[i], 0);
        }

        return (c[0].at<float>(0, 0) + c[1].at<float>(0, 0) + c[2].at<float>(0, 0)) / 3.0;
    }
};

class decrypt_image : CSencryption {
private:
    cv::Mat c[3];
    float optimal_value;

public:
    decrypt_image(std::string input_path) {
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
        optimal_value = (float)std::stoi(splitText[3]);
    }

    decrypt_image() {}

    decrypt_image(cv::Mat input) {
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
        optimal_value = (float)std::stoi(splitText[3]);
    }

    void decrypt(std::vector<cv::Mat> ref, int seed, int num_iterations, float coef, bool opt, int tile_index = -1) {
        ri_x.resize(m);
        ri_y.resize(m);
        returnRandomIndices(ri_x, ri_y, rows, cols, m, seed);
        //std::vector<cv::Mat> ref = createRefDCT(rows, cols);
        int n = rows * cols;
        std::vector<std::thread> CPUProcessing(3);

        for (int i = 0; i < 3; i++) {
            //reconstruct_color_chanel(c[i], encrypted_img, i, coef, optimal_value, rows, cols, ri_x, ri_y, num_iterations, ref, opt, tile_index);
        }

        for (int i = 0; i < 3; i++) {
            CPUProcessing[i] = std::thread(reconstruct_color_chanel, std::ref(c[i]), std::ref(encrypted_img), i, coef, optimal_value, rows, cols, std::ref(ri_x), std::ref(ri_y), num_iterations, std::ref(ref), opt, tile_index, true);
        }

        for (int i = 0; i < 3; i++) {
            CPUProcessing[i].join();
        }

        cv::merge(c, 3, decrypted_img);
        decrypted_img.convertTo(decrypted_img, CV_8UC3);
    }

    void decrypt_cont(std::vector<cv::Mat>& ref, int seed, int num_iterations, float coef, bool opt, int tile_ind = - 1) {
         ref = createRefDCT(decrypted_img.rows, decrypted_img.cols, decrypted_img);
        int n = rows * cols;
        std::vector<std::thread> CPUProcessing(3);



        for (int i = 0; i < 3; i++) {
            //reconstruct_color_chanel(c[i], encrypted_img, i, coef, optimal_value, rows, cols, ri_x, ri_y, num_iterations, ref, opt, tile_ind, false);
            CPUProcessing[i] = std::thread(reconstruct_color_chanel, std::ref(c[i]), std::ref(encrypted_img), i, coef,
                optimal_value, rows, cols, std::ref(ri_x), std::ref(ri_y), num_iterations, ref, opt, tile_ind, false);
        }

        for (int i = 0; i < 3; i++) {
            CPUProcessing[i].join();
        }

        //cv::merge(c, 3, decrypted_img);
        //decrypted_img.convertTo(decrypted_img, CV_8UC3);
    }

    void get_mat(cv::Mat& dest) {
        decrypted_img.copyTo(dest);
    }

    void writeDecryptedImageToDisk(std::string output_path, bool remove_noise = false, bool noise_level = 3) {

        cv::imwrite(output_path, decrypted_img);

        if (remove_noise) {
            cv::Mat decrypted_img_noisless = cv::imread(output_path, cv::IMREAD_COLOR);

            cv::fastNlMeansDenoisingColored(decrypted_img_noisless, decrypted_img_noisless, noise_level);
            cv::imwrite(output_path, decrypted_img_noisless);
        }
    }
};

int nextClosestDivisible(int x, int y) {
    // Ensure y is not zero to avoid division by zero error
    if (y == 0) {
        throw std::invalid_argument("y must not be zero");
    }

    // Find the next multiple of y greater than x
    int nextMultiple = ((x + y - 1) / y) * y;

    return nextMultiple;
}

std::vector<int> spiralOrder(std::vector<std::vector<int>>& matrix) {
    std::vector<int> result;
    int m = matrix.size();
    if (m == 0) return result;
    int n = matrix[0].size();

    int startRow = m / 2, startCol = n / 2; // start from the middle
    int dir = 0; // 0 = up, 1 = left, 2 = down, 3 = right
    int steps = 1, stepCount = 0;

    int row = startRow, col = startCol;
    result.push_back(matrix[row][col]);

    while (result.size() < m * n) {
        for (int i = 0; i < 2; ++i) {
            for (int j = 0; j < steps; ++j) {
                if (dir == 0) --row;
                else if (dir == 1) --col;
                else if (dir == 2) ++row;
                else ++col;

                if (row >= 0 && row < m && col >= 0 && col < n) {
                    result.push_back(matrix[row][col]);
                }
            }
            dir = (dir + 1) % 4;
        }
        ++steps;
    }

    return result;
}



void copyTilesToImage(const std::vector<cv::Mat>& tiles, cv::Mat& image, int tile_width, int tile_height, int rows, int cols) {
    // Check if the tiles vector has the expected number of tiles
    if (tiles.size() != rows * cols) {
        std::cerr << "Error: Number of tiles does not match rows*cols." << std::endl;
        return;
    }

    // Iterate over each tile and copy it to the corresponding place in the image
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            int tile_index = i * cols + j;
            cv::Rect roi(j * tile_width, i * tile_height, tile_width, tile_height);
            tiles[tile_index].copyTo(image(roi));
        }
    }
}

std::vector<cv::Mat> splitImageIntoTiles(const cv::Mat& image, int tile_width, int tile_height, int rows, int cols) {
    std::vector<cv::Mat> tiles;

    // Iterate over each tile position and extract the tile from the image
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            cv::Rect roi(j * tile_width, i * tile_height, tile_width, tile_height);
            tiles.push_back(image(roi).clone());
        }
    }

    return tiles;
}

void process_1(std::vector<float>& stddevs, int scaled_rows, int scaled_cols, 
    std::vector<int>& result, int num_threads, int pass, std::vector<cv::Mat>& mats_in,
    std::vector<cv::Mat>& mats_out, std::vector<decrypt_image>& dimgs, int N, std::vector<std::vector<cv::Mat>>& refs) {

    #pragma omp parallel for num_threads(num_threads) schedule(dynamic)
    for (int i = 0; i < result.size(); i++) {
        std::vector<cv::Mat> ref = createRefDCT(scaled_rows, scaled_cols);
        dimgs[i] = decrypt_image(mats_in[result[i]]);
        //dimgs[i].decrypt(ref, pass, 10, (0.1 * stddevs[result[i]] <= 0.01) ? 0.01 : 0.1 * stddevs[result[i]], true);
        dimgs[i].decrypt(ref, pass, 10, 0.05, true);
        mtx.lock();
        dimgs[i].get_mat(mats_out[result[i]]);
        mtx.unlock();
    }
}

void process_2(std::vector<float>& stddevs, int scaled_rows, int scaled_cols,
    std::vector<int>& result, int num_threads, int pass, std::vector<cv::Mat>& mats_in,
    std::vector<cv::Mat>& mats_out, std::vector<decrypt_image>& dimgs, int N, std::vector<std::vector<cv::Mat>>& refs) {

    #pragma omp parallel for num_threads(num_threads) schedule(dynamic)
    for (int i = 0; i < result.size(); i++) {
        std::vector<cv::Mat> ref = createRefDCT(scaled_rows, scaled_cols, mats_out[result[i]].clone());
        dimgs[i] = decrypt_image(mats_in[result[i]]);
        dimgs[i].decrypt(ref, pass, 50, 0.05, true);
        mtx.lock();
        dimgs[i].get_mat(mats_out[result[i]]);
        mtx.unlock();
    }
}

void display_output(std::string windowName, cv::Mat& temp, int N) {
    while(g_dsp) {
        cv::waitKey(100);
        copyTilesToImage(mats_out, temp, mats_out[0].cols, mats_out[0].rows, N, N);
        updateImage(windowName, temp);
        cv::waitKey(1);
    }
}

int main(int argc, char* argv)
{
    std::vector<const char*> inputs(4);
    inputs[0] = "IMG_3690.png";
    inputs[1] = "IMG_1297.png";
    inputs[2] = "IMG_0007.png";
    inputs[3] = "IMG_9321.png";

    std::vector<const char*> inputs_encrypted(4);
    inputs_encrypted[0] = "IMG_3690_encrypted.png";
    inputs_encrypted[1] = "IMG_1297_encrypted.png";
    inputs_encrypted[2] = "IMG_0007_encrypted.png";
    inputs_encrypted[3] = "IMG_9321_encrypted.png";

    std::vector<const char*> inputs_decrypted(4);
    inputs_decrypted[0] = "IMG_3690_decrypted.png";
    inputs_decrypted[1] = "IMG_1297_decrypted.png";
    inputs_decrypted[2] = "IMG_0007_decrypted.png";
    inputs_decrypted[3] = "IMG_9321_decrypted.png";

    int N = 8, M = 8;
    int num_threads = 12;

    std::vector<cv::Mat> mats_in(N * M);
    mats_out.resize(N * N);
    cv::Mat img = cv::imread(inputs[0], cv::IMREAD_COLOR);
    cv::resize(img, img, cv::Size(nextClosestDivisible(img.cols, N), nextClosestDivisible(img.rows, M)));
    mats_in = splitMat(img, N , M);
    img.deallocate();

    if (mats_in[0].rows % 2 != 0) {
        for (int i = 0; i < N * M; i++) {
            cv::resize(mats_in[i], mats_in[i], cv::Size(mats_in[i].cols, mats_in[i].rows + 1));
        }
    }

    if (mats_in[0].cols % 2 != 0) {
        for (int i = 0; i < N * M; i++) {
            cv::resize(mats_in[i], mats_in[i], cv::Size(mats_in[i].cols + 1, mats_in[i].rows));
        }
    }

    int scaled_rows = mats_in[0].rows, scaled_cols = mats_in[0].cols;

    std::vector<std::vector<int>> matrix(N, std::vector<int>(M));

    int k = 0;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            matrix[i][j] = k++;
        }
    }

    std::vector<int> result = spiralOrder(matrix);

    std::vector<float> stddevs;

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            cv::Scalar mean, stddev;
            cv::meanStdDev(mats_in[i * N + j], mean, stddev);
            stddevs.push_back(stddev.val[0]);
        }
    }

    #pragma omp parallel for num_threads(num_threads) schedule(dynamic)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            stddevs[i * N + j] = stddevs[i * N + j] / *std::max_element(stddevs.begin(), stddevs.end());// *0.8;

            if (stddevs[i * N + j] < 0.125) {
                stddevs[i * N + j] = 0.125;
            }

            if (stddevs[i * N + j] > 0.5) {
                stddevs[i * N + j] = 0.5;
            }

            encrypt_image img(mats_in[i * N + j], false);
            img.encrypt(stddevs[i * N + j], 1);
            img.get_mat(mats_in[i * N + j]);
        }
    }

    cv::Mat encrypted_img_g = stitchMats_alt(mats_in, N, M);

    cv::imwrite("encrypted_img_g.png", encrypted_img_g);

    #pragma omp parallel for num_threads(num_threads) schedule(dynamic)
    for (int i = 0; i < result.size(); i++) {
        mats_out[i] = cv::Mat::zeros(scaled_rows, scaled_cols, CV_8UC3);
        mats_out[i].setTo(cv::Scalar(255, 255, 255));
    }

    std::vector<decrypt_image> dimgs(N * N);
    int pass = 1;
    std::string windowName = "ImageWindow";
    

    cv::Mat temp = cv::Mat::zeros(scaled_rows * N, scaled_cols * N, CV_8UC3);

    volatile bool* dsp = new bool;
    *dsp = true;
    std::thread CPU_display_output;
    CPU_display_output = std::thread(display_output, windowName, std::ref(temp), N);
    
    std::vector<std::vector<cv::Mat>> ref(N * N);

    std::thread CPUProcessing;

    CPUProcessing = std::thread(process_1, std::ref(stddevs), scaled_rows, scaled_cols, std::ref(result), std::ref(num_threads), 
       pass, std::ref(mats_in), std::ref(mats_out), std::ref(dimgs), N, std::ref(ref));

    CPUProcessing.join();

    copyTilesToImage(mats_out, temp, mats_out[0].cols, mats_out[0].rows, N, N);
    updateImage(windowName, temp);
    cv::waitKey(1);

    std::thread CPUProcessing2;

    CPUProcessing2 = std::thread(process_2, std::ref(stddevs), scaled_rows, scaled_cols, std::ref(result), std::ref(num_threads),
        pass, std::ref(mats_in), std::ref(mats_out), std::ref(dimgs), N, std::ref(ref));


    CPUProcessing2.join();
    g_dsp = false;
    CPU_display_output.join();
    cv::Mat out = cv::Mat::zeros(scaled_rows * N, scaled_cols * N, CV_8UC3);
    copyTilesToImage(mats_out, out, mats_out[0].cols, mats_out[0].rows, N, N);
    cv::fastNlMeansDenoising(out, out, 3);
    cv::imwrite("result.png", out);
    return 0;
}




