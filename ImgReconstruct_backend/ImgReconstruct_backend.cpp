#include <stdio.h>
#include "lbfgs.h"
#include "lbfgs.h"
#include <mutex>
#include <vector>
#include <string>
#include <chrono>
#include <stdlib.h>
#include <random>

int nextClosestDivisible(int x, int y) {
    // Ensure y is not zero to avoid division by zero error
    if (y == 0) {
        throw std::invalid_argument("y must not be zero");
    }

    // Find the next multiple of y greater than x
    int nextMultiple = ((x + y - 1) / y) * y;

    return nextMultiple;
}

struct TileCoord {
    int x;
    int y;
};

cv::Mat reconstructImage(const std::vector<std::vector<cv::Mat>>& tiles,
    const std::vector<std::vector<TileCoord>>& coordinates) {
    if (tiles.empty() || coordinates.empty() ||
        tiles.size() != coordinates.size() ||
        tiles[0].size() != coordinates[0].size()) {
        return cv::Mat();
    }

    int tileCountN = tiles.size();

    // Calculate output image size
    int maxX = 0, maxY = 0;
    for (int i = 0; i < tileCountN; i++) {
        for (int j = 0; j < tileCountN; j++) {
            int rightEdge = coordinates[i][j].x + tiles[i][j].cols;
            int bottomEdge = coordinates[i][j].y + tiles[i][j].rows;
            maxX = std::max(maxX, rightEdge);
            maxY = std::max(maxY, bottomEdge);
        }
    }

    // Create output image
    cv::Mat output(maxY, maxX, tiles[0][0].type(), cv::Scalar(0));

    // Copy tiles to their original positions
    for (int i = 0; i < tileCountN; i++) {
        for (int j = 0; j < tileCountN; j++) {
            if (!tiles[i][j].empty()) {
                cv::Rect roi(coordinates[i][j].x,
                    coordinates[i][j].y,
                    tiles[i][j].cols,
                    tiles[i][j].rows);
                tiles[i][j].copyTo(output(roi));
            }
        }
    }

    return output;
}

struct indices {
    std::vector<int> ri_x_g, ri_y_g;
};

std::vector<std::vector<cv::Mat>> reconfigured_cropped_out;

std::mutex mtx;
std::mutex mtx_decryption;
bool g_dsp = true;
std::vector<int> passwords;

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

class CSencryption {
protected:
    cv::Mat encrypted_img, input_img;
    cv::Mat decrypted_img;
    cv::Size org_size;
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
    }

    return c;
}

void reconstruct_color_chanel(cv::Mat& out, cv::Mat& measurement, int k, float param_c, float optimal_value, int rows, int cols, std::vector<int>& ri_x, std::vector<int>& ri_y, int iterations, std::vector<cv::Mat> ref, bool opt, int tile_index, bool copy_result = true) {

    int n = rows * cols; // size of solution (size of vectorized image)
    float fx;
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

    cv::Mat AtAxb2(rows, cols, CV_32F, (float*)ref[k].data);
    dct(AtAxb2, AtAxb2, cv::DCT_INVERSE);
    AtAxb2 = AtAxb2 * 255.0;
    out = AtAxb2;// .clone();
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
        org_size.width = cols;
        org_size.height = rows;
        cv::resize(input_img, input_img, cv::Size(nextClosestDivisible(input_img.cols, 24), nextClosestDivisible(input_img.rows, 24)));
    }

    encrypt_image(cv::Mat input, bool remove_noise = false, int noise_level = 3) {
        input.copyTo(input_img);
        if (remove_noise) {
            cv::fastNlMeansDenoisingColored(input_img, input_img, noise_level);
        }
        rows = input_img.rows;
        cols = input_img.cols;
        org_size.width = cols;
        org_size.height = rows;
    }

    encrypt_image() {}

    void get_mat(cv::Mat& dest) {
        encrypted_img.copyTo(dest);
    }

    cv::Mat get_mat() {
        return encrypted_img;// .clone();
    }

    cv::Mat get_sampled_mat() {
        cv::Mat sampled_mat = cv::Mat::zeros(rows, cols, CV_8UC3);

        for (int i = 0; i < ri_x.size(); i++) {
            sampled_mat.at<cv::Vec3b>(ri_x[i], ri_y[i]) = input_img.at<cv::Vec3b>(ri_x[i], ri_y[i]);
        }

        return sampled_mat.clone();
    }

    cv::Mat get_sampled_mask() {
        cv::Mat sampled_mat = cv::Mat::zeros(rows, cols, CV_8UC3);
        sampled_mat.setTo(cv::Scalar(0, 0, 0));

        for (int i = 0; i < ri_x.size(); i++) {
            sampled_mat.at<cv::Vec3b>(ri_x[i], ri_y[i]) = cv::Vec3b(1, 1, 1);
        }

        return sampled_mat.clone();
    }

    void encrypt(float pixel_p = 0.3, int seed = 1) {
        bm = pixel_p;
        m = rows * cols * bm;
        int n = rows * cols;
        ri_x.resize(m);
        ri_y.resize(m);
        returnRandomIndices(ri_x, ri_y, rows, cols, m, seed);
        encrypted_img = cv::Mat(1, (int)(sqrt(m + 6) + 1) * (int)(sqrt(m + 6) + 1), CV_8UC3);

        std::string text = std::to_string(m) + "|" + std::to_string(rows) + "|" + std::to_string(cols) + "|" + std::to_string(org_size.height) + "|" + std::to_string(org_size.width); // Example string 
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
            encrypted_img.at<cv::Vec3b>(k + colorMat.total()) = input_img.at<cv::Vec3b>(ri_x[k], ri_y[k]);
            k++;
        }

        encrypted_img = encrypted_img.reshape(0, (int)sqrt(encrypted_img.total()));
    }

    void encrypt(std::vector<int> ri_x_g, std::vector<int> ri_y_g) {
        m = ri_x_g.size();
        int n = rows * cols;
        encrypted_img = cv::Mat(1, (int)(sqrt(m + 6) + 1) * (int)(sqrt(m + 6) + 1), CV_8UC3);

        std::string text = std::to_string(m) + "|" + std::to_string(rows) + "|" + std::to_string(cols) + "|" + std::to_string(org_size.height) + "|" + std::to_string(org_size.width); // Example string 
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
            encrypted_img.at<cv::Vec3b>(k + colorMat.total()) = input_img.at<cv::Vec3b>(ri_x_g[k], ri_y_g[k]);
            k++;
        }

        encrypted_img = encrypted_img.reshape(0, (int)sqrt(encrypted_img.total()));
    }

    void writeEncryptedImageToDisk(std::string output_path) {
        cv::imwrite(output_path, encrypted_img);
    }

private:

    std::vector<float> analyze(cv::Mat in) {
        cv::Mat p = in.clone();

        std::vector<cv::Mat> c;
        std::vector<float> res(3);
        cv::split(p, c);


#pragma omp parallel for num_threads(3) schedule(dynamic)
        for (int i = 0; i < 3; i++) {
            c[i].convertTo(c[i], CV_32F);
            c[i] = c[i] / 255.0;
            cv::dct(c[i], c[i], 0);
        }
        res[0] = c[0].at<float>(0, 0);
        res[1] = c[1].at<float>(0, 0);
        res[2] = c[2].at<float>(0, 0);
        return res;
    }
};

class decrypt_image : CSencryption {
private:
    cv::Mat c[3];
    std::vector<float> optimal_values;

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
        optimal_values.resize(3);
        org_size.height = (float)std::stoi(splitText[3]);
        org_size.width = (float)std::stoi(splitText[4]);
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
        optimal_values.resize(3);
        org_size.height = (float)std::stoi(splitText[3]);
        org_size.width = (float)std::stoi(splitText[4]);
    }

    void decrypt(int seed, int num_iterations, float coef, bool opt, int tile_index = -1) {
        ri_x.resize(m);
        ri_y.resize(m);
        returnRandomIndices(ri_x, ri_y, rows, cols, m, seed);
        std::vector<cv::Mat> ref = createRefDCT(rows, cols);

        int n = rows * cols;
        std::vector<std::thread> CPUProcessing(3);

        for (int i = 0; i < 3; i++) {
            reconstruct_color_chanel(c[i], encrypted_img, i, coef, optimal_values[0], rows, cols, ri_x, ri_y, num_iterations, ref, opt, tile_index, true);
        }
        /*
        for (int i = 0; i < 3; i++) {
            CPUProcessing[i] = std::thread(reconstruct_color_chanel, std::ref(c[i]), std::ref(encrypted_img), i, coef,
                optimal_values[0], rows, cols, std::ref(ri_x), std::ref(ri_y),
                num_iterations, std::ref(ref), opt, tile_index, true);
        }

        for (int i = 0; i < 3; i++) {
            CPUProcessing[i].join();
        }
        */

        cv::merge(c, 3, decrypted_img);
        decrypted_img.convertTo(decrypted_img, CV_8UC3);
    }

    void decrypt(cv::Mat ref_img, std::vector<int> ri_x_g, std::vector<int> ri_y_g, int num_iterations, float coef, bool opt, int tile_index = -1) {
        ri_x = ri_x_g;
        ri_y = ri_y_g;
        std::vector<cv::Mat> ref = createRefDCT(rows, cols, ref_img);
        int n = rows * cols;
        std::vector<std::thread> CPUProcessing(3);

        for (int i = 0; i < 3; i++) {
            reconstruct_color_chanel(c[i], encrypted_img, i, coef, optimal_values[0], rows, cols, ri_x, ri_y, num_iterations, ref, opt, tile_index, true);
        }

        /*
        for (int i = 0; i < 3; i++) {
            CPUProcessing[i] = std::thread(reconstruct_color_chanel, std::ref(c[i]), std::ref(encrypted_img), i, coef,
                optimal_values[0], rows, cols, std::ref(ri_x), std::ref(ri_y),
                num_iterations, std::ref(ref), opt, tile_index, true);
        }

        for (int i = 0; i < 3; i++) {
            CPUProcessing[i].join();
        }
        */

        cv::merge(c, 3, decrypted_img);
        decrypted_img.convertTo(decrypted_img, CV_8UC3);
    }

    void decrypt(std::vector<int> ri_x_g, std::vector<int> ri_y_g, int num_iterations, float coef, bool opt, int tile_index = -1) {
        ri_x = ri_x_g;
        ri_y = ri_y_g;
        std::vector<cv::Mat> ref = createRefDCT(rows, cols);
        int n = rows * cols;
        std::vector<std::thread> CPUProcessing(3);

        for (int i = 0; i < 3; i++) {
            reconstruct_color_chanel(c[i], encrypted_img, i, coef, optimal_values[0], rows, cols, ri_x, ri_y, num_iterations, ref, opt, tile_index, true);
        }
        /*
        for (int i = 0; i < 3; i++) {
            CPUProcessing[i] = std::thread(reconstruct_color_chanel, std::ref(c[i]), std::ref(encrypted_img), i, coef,
                optimal_values[0], rows, cols, std::ref(ri_x), std::ref(ri_y),
                num_iterations, std::ref(ref), opt, tile_index, true);
        }

        for (int i = 0; i < 3; i++) {
            CPUProcessing[i].join();
        }
        */
        cv::merge(c, 3, decrypted_img);
        decrypted_img.convertTo(decrypted_img, CV_8UC3);
    }

    void decrypt_cont(std::vector<int> ri_x_g, std::vector<int> ri_y_g, int num_iterations, float coef, bool opt, int tile_ind = -1) {

    }

    void get_mat(cv::Mat& dest) {
        decrypted_img.copyTo(dest);
    }

    cv::Mat get_mat() {
        return decrypted_img;// .clone();
    }

    void writeDecryptedImageToDisk(std::string output_path, bool remove_noise = false, bool noise_level = 3) {

        cv::imwrite(output_path, decrypted_img);

        if (remove_noise) {
            cv::Mat decrypted_img_noisless = cv::imread(output_path, cv::IMREAD_COLOR);

            cv::fastNlMeansDenoisingColored(decrypted_img_noisless, decrypted_img_noisless, noise_level);
            cv::imwrite(output_path, decrypted_img_noisless);
        }
    }


    cv::Mat get_sampled_mat(int seed) {
        cv::Mat sampled_mat = cv::Mat::zeros(rows, cols, CV_8UC3);
        sampled_mat.setTo(cv::Scalar(255, 255, 255));
        ri_x.resize(m);
        ri_y.resize(m);
        returnRandomIndices(ri_x, ri_y, rows, cols, m, seed);

        for (int i = 11; i < ri_x.size() + 11; ++i) {
            sampled_mat.at<cv::Vec3b>(ri_x[i - 11], ri_y[i - 11]) = encrypted_img.at<cv::Vec3b>(i);
        }

        return sampled_mat;
    }

    cv::Mat get_sampled_mask(int seed) {
        cv::Mat sampled_mask = cv::Mat::zeros(rows, cols, CV_8UC3);

        //ri_x.resize(m);
        //ri_y.resize(m);
        //returnRandomIndices(ri_x, ri_y, rows, cols, m, seed);
        
        for (int i = 0; i < ri_x.size(); i++) {
            sampled_mask.at<cv::Vec3b>(ri_x[i], ri_y[i]) = cv::Vec3b(1, 1, 1);
        }

        return sampled_mask.clone();
    }

    void get_sampled_mask_mats(int seed, cv::Mat& sampled_mat, cv::Mat& sampled_mask) {
        sampled_mask = cv::Mat::zeros(rows, cols, CV_8UC3);
        sampled_mat = cv::Mat::zeros(rows, cols, CV_8UC3);
        ri_x.resize(m);
        ri_y.resize(m);
        returnRandomIndices(ri_x, ri_y, rows, cols, m, seed);

        for (int i = 11, j = 0; i < ri_x.size() + 11, j < ri_x.size(); i++, j++) {
            sampled_mat.at<cv::Vec3b>(ri_x[i - 11], ri_y[i - 11]) = encrypted_img.at<cv::Vec3b>(i);
            sampled_mask.at<cv::Vec3b>(ri_x[j], ri_y[j]) = cv::Vec3b(1, 1, 1);
        }
    }

    cv::Size get_org_size() {
        return org_size;
    }
};

std::vector<std::string> spiralOrder(std::vector<std::vector<std::string>>& matrix) {
    std::vector<std::string> result;
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

void process_1(int num_threads, int pass, std::vector<std::vector<cv::Mat>>& mats_in, std::vector<std::vector<indices>> indices,
    std::vector<std::vector<cv::Mat>>& mats_out, std::vector<std::string> processing_order, int iterations) {

    #pragma omp parallel for num_threads(num_threads) schedule(dynamic)
    for (int k = 0; k < mats_in[0].size() * mats_in[0].size(); k++) {
        std::vector<std::string> splitText = splitString(processing_order[k], '_');
        int i = std::stoi(splitText[0]);
        int j = std::stoi(splitText[1]);

        decrypt_image* dimgs = new decrypt_image;
        *dimgs = decrypt_image(mats_in[i][j]);
        dimgs->decrypt(indices[i][j].ri_x_g, indices[i][j].ri_y_g, iterations, 0.05, false);
        //mtx_decryption.lock();
        dimgs->get_mat(mats_out[i][j]);
        //mtx_decryption.unlock();
        delete(dimgs);
        mats_in[i][j].deallocate();
        indices[i][j].ri_x_g.resize(0);
        indices[i][j].ri_y_g.resize(0);
    }

}

void process_2(int num_threads, int pass, std::vector<std::vector<cv::Mat>>& mats_in, std::vector<std::vector<indices>> indices,
    std::vector<std::vector<cv::Mat>>& mats_out, std::vector<std::vector<decrypt_image>>& dimgs, std::vector<std::string> processing_order) {

    #pragma omp parallel for num_threads(num_threads) schedule(dynamic)
    for (int i = 0; i < mats_in[0].size(); i++) {
        for (int j = 0; j < mats_in[0].size(); j++) {
            dimgs[i][j].decrypt_cont(indices[i][j].ri_x_g, indices[i][j].ri_y_g, 15, 0.05, false);
            dimgs[i][j].get_mat(mats_out[i][j]);
        }
    }
}

void display_output(std::string windowName, cv::Mat& temp, std::vector<std::vector<TileCoord>>& coordinates) {
    while (g_dsp) {
        //mtx.lock();
        cv::waitKey(10);
        temp = reconstructImage(reconfigured_cropped_out, coordinates);
        updateImage(windowName, temp);
        cv::waitKey(1);
        //mtx.unlock();
    }
}

void splitImageIntoTiles(const cv::Mat& inputImage,
    std::vector<std::vector<cv::Mat>>& tiles,
    std::vector<std::vector<TileCoord>>& coordinates,
    int tileCountN,
    int overlap) {
    // Input validation
    if (inputImage.empty() || tileCountN <= 0 || overlap < 0) {
        return;
    }

    int height = inputImage.rows;
    int width = inputImage.cols;

    // Calculate tile dimensions considering overlap
    int tileWidth = (width + (tileCountN - 1) * overlap) / tileCountN;
    int tileHeight = (height + (tileCountN - 1) * overlap) / tileCountN;

    // Resize vectors to N x N
    tiles.resize(tileCountN, std::vector<cv::Mat>(tileCountN));
    coordinates.resize(tileCountN, std::vector<TileCoord>(tileCountN));

    // Split image into tiles
#pragma omp parallel for num_threads(8) schedule(dynamic)
    for (int i = 0; i < tileCountN; i++) {
        for (int j = 0; j < tileCountN; j++) {
            // Calculate tile position
            int x = j * (tileWidth - overlap);
            int y = i * (tileHeight - overlap);

            // Adjust for edges
            int currentWidth = tileWidth;
            int currentHeight = tileHeight;

            if (x + tileWidth > width) {
                currentWidth = width - x;
            }
            if (y + tileHeight > height) {
                currentHeight = height - y;
            }

            // Ensure valid coordinates
            if (x < 0 || y < 0 || x >= width || y >= height) {
                continue;
            }

            // Extract tile
            cv::Rect roi(x, y, currentWidth, currentHeight);
            tiles[i][j] = inputImage(roi).clone();

            // Store coordinates
            coordinates[i][j] = { x, y };
        }
    }
}


cv::Mat blendTilesWithImage(const std::vector<std::vector<cv::Mat>>& tiles,
    const std::vector<std::vector<TileCoord>>& coordinates,
    const cv::Mat& targetImage,
    float alpha = 0.5f) {
    // Input validation
    if (tiles.empty() || coordinates.empty() ||
        tiles.size() != coordinates.size() ||
        tiles[0].size() != coordinates[0].size() ||
        targetImage.empty()) {
        return cv::Mat();
    }

    // Check if target image has valid dimensions
    int tileCountN = tiles.size();
    int maxX = targetImage.cols;
    int maxY = targetImage.rows;

    // Create a copy of the target image as base
    cv::Mat output = targetImage.clone();

    // Validate alpha value
    alpha = std::max(0.0f, std::min(1.0f, alpha));  // Clamp between 0 and 1

    // Blend each tile with the target image
    #pragma omp parallel for num_threads(8) schedule(dynamic)
    for (int i = 0; i < tileCountN; i++) {
        for (int j = 0; j < tileCountN; j++) {
            if (!tiles[i][j].empty()) {
                // Get tile dimensions and position
                int tileWidth = tiles[i][j].cols;
                int tileHeight = tiles[i][j].rows;
                int x = coordinates[i][j].x;
                int y = coordinates[i][j].y;

                // Ensure tile fits within output image
                if (x < 0 || y < 0 || x + tileWidth > maxX || y + tileHeight > maxY) {
                    continue;
                }

                // Define ROI in output image
                cv::Rect roi(x, y, tileWidth, tileHeight);
                cv::Mat outputROI = output(roi);

                // Ensure compatible types
                if (tiles[i][j].type() != outputROI.type()) {
                    continue;
                }

                // Perform alpha blending
                // outputROI = alpha * tile + (1 - alpha) * outputROI
                addWeighted(tiles[i][j], alpha, outputROI, 1.0 - alpha, 0.0, outputROI);
            }
        }
    }

    return output;
}

void decrypt_image_tiled(cv::Mat encrypted_img_g, int tiles, int overlap, int iterations, int nun_threads) {
    cv::Mat sampled_mat;
    decrypt_image dimgs = decrypt_image(encrypted_img_g);
    cv::Size org_size = dimgs.get_org_size();

    sampled_mat = dimgs.get_sampled_mat(1);
    //cv::Mat masked_mat = cv::Mat::zeros(sampled_mat.rows, sampled_mat.cols, CV_32FC1);
    //masked_mat = masked_mat - sampled_mat;
    //masked_mat = dimgs.get_sampled_mask(1);
    //dimgs.get_sampled_mask_mats(1, sampled_mat, masked_mat);

    for (int i = 0; i < 2; i++) {
        if (i == 0) {
            //sampled_mat = dimgs.get_sampled_mat(1);
        }
        if (i == 1) {
            //masked_mat = dimgs.get_sampled_mask(1);
        }
    }

    //cv::imwrite("sampled_mat.png", sampled_mat);
    //cv::imwrite("sampled_mask.png", masked_mat);
    std::string windowName = "ImageWindow";


    int N_reconfigured = tiles;
    reconfigured_cropped_out.resize(N_reconfigured, std::vector<cv::Mat>(N_reconfigured));
    std::vector<std::vector<cv::Mat>> reconfigured_cropped_mats_in;
    //std::vector<std::vector<cv::Mat>>  reconfigured_cropped_masks;
    //std::vector<std::vector<decrypt_image>> reconfigured_dimgs(N_reconfigured, std::vector<decrypt_image>(N_reconfigured));
    std::vector<std::vector<indices>> indices_reconfigured(N_reconfigured, std::vector<indices>(N_reconfigured));
    std::vector<std::vector<TileCoord>> coordinates;

    splitImageIntoTiles(sampled_mat, reconfigured_cropped_mats_in, coordinates, N_reconfigured, overlap);
    //splitImageIntoTiles(masked_mat, reconfigured_cropped_masks, coordinates, N_reconfigured, overlap);
    //masked_mat.deallocate();
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
            //reconfigured_cropped_masks[i][j].deallocate();
        }
    }

    cv::Mat reconstructed = cv::Mat::zeros(sampled_mat.rows, sampled_mat.cols, CV_8UC3);
    std::thread CPU_display_output;
    CPU_display_output = std::thread(display_output, windowName, std::ref(reconstructed), std::ref(coordinates));

    std::thread CPUProcessing1;
    CPUProcessing1 = std::thread(process_1, nun_threads, 1, std::ref(reconfigured_cropped_mats_in), std::ref(indices_reconfigured),
        std::ref(reconfigured_cropped_out), std::ref(result), iterations);
    CPUProcessing1.join();

    reconstructed = reconstructImage(reconfigured_cropped_out, coordinates);
    cv::Mat blended = blendTilesWithImage(reconfigured_cropped_out, coordinates, reconstructed, 0.5f);
    cv::resize(blended, blended, org_size);
    cv::imwrite("blended.png", blended);
    g_dsp = false;
    CPU_display_output.join();
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