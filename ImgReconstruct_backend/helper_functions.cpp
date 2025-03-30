#include "helper_functions.hpp"

int nextClosestDivisible(int x, int y) {
    // Ensure y is not zero to avoid division by zero error
    if (y == 0) {
        throw std::invalid_argument("y must not be zero");
    }

    // Find the next multiple of y greater than x
    int nextMultiple = ((x + y - 1) / y) * y;

    return nextMultiple;
}

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

// Function to update the image
void updateImage(const std::string& windowName, const cv::Mat& newImage) {
    cv::Mat aux = newImage.clone();
    cv::resize(aux, aux, cv::Size(newImage.cols / 3, newImage.rows / 3));
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

inline void updateAxb2AndComputeFx(float* x_copy, int* ri_x, int* ri_y,
    float* Axb2_vec, const float* b, int cols, float& fx, int n) {
    __m256 fx_vec = _mm256_setzero_ps();  // Accumulator for fx

    // Process 8 elements at a time
    int i = 0;
    for (; i <= n - 8; i += 8) {
        // Gather indices
        int idx[8];
        for (int k = 0; k < 8; k++) {
            idx[k] = ri_x[i + k] * cols + ri_y[i + k];
        }

        // Load x_copy values using gather
        __m256 x_val = _mm256_i32gather_ps(x_copy, _mm256_load_si256((__m256i*) & idx[0]), 4);

        // Load b values
        __m256 b_val = _mm256_load_ps(&b[i]);

        // Compute differences
        __m256 diff = _mm256_sub_ps(x_val, b_val);

        // Accumulate fx (diff * diff)
        fx_vec = _mm256_fmadd_ps(diff, diff, fx_vec);

        // Store differences to Axb2_vec
        alignas(32) float temp[8];
        _mm256_store_ps(temp, diff);
        for (int k = 0; k < 8; k++) {
            Axb2_vec[idx[k]] = temp[k];
        }
    }

    // Handle remaining elements
    float fx_temp = 0.0f;
    for (; i < n; ++i) {
        int idx = ri_x[i] * cols + ri_y[i];
        float diff = x_copy[idx] - b[i];
        fx_temp += diff * diff;
        Axb2_vec[idx] = diff;
    }

    // Reduce fx_vec to scalar
    __m128 hi = _mm256_extractf128_ps(fx_vec, 1);
    __m128 lo = _mm256_castps256_ps128(fx_vec);
    __m128 sum = _mm_add_ps(hi, lo);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    fx = _mm_cvtss_f32(sum) + fx_temp;
}

inline void eval_g(float* Axb2, float* g, int n) {
    __m256 scalar = _mm256_set1_ps(2.0f); // Set scalar to 2.0f
    int i = 0;
    // Process multiples of 8 (AVX2 processes 8 floats at a time)
    for (; i <= n - 8; i += 8) {
        __m256 vecData = _mm256_load_ps(&Axb2[i]);  // Load 8 floats
        _mm256_store_ps(&g[i], _mm256_mul_ps(vecData, scalar));  // Multiply and store
    }

    // Process remaining elements
    for (; i < n; ++i) {
        g[i] = Axb2[i] * 2.0f;
    }
}

inline void copy_x(float* x_copy, float* x, float* Axb2_vec, int n) {
    __m256 factor = _mm256_set1_ps(0.0f); // Set factor to 0.0f
    int i = 0;
    // Process multiples of 8
    for (; i <= n - 8; i += 8) {
        __m256 vecData = _mm256_load_ps(&x[i]);  // Load 8 floats from input
        _mm256_store_ps(&x_copy[i], vecData);  // Copy to x_copy
        _mm256_store_ps(&Axb2_vec[i], factor); // Set Axb2_vec to 0
    }

    // Process remaining elements
    for (; i < n; ++i) {
        x_copy[i] = x[i];
        Axb2_vec[i] = 0.0f;
    }
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
        c[i] = c[i] / 255.0f;
        cv::dct(c[i], c[i], 0);
        c[i] = c[i] / 10.0;
    }

    return c;
}

void reconstruct_color_chanel(cv::Mat& out, cv::Mat& measurement, int k, float param_c, int rows, int cols, std::vector<int>& ri_x, std::vector<int>& ri_y, int iterations, std::vector<cv::Mat> ref) {

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

    for (int i = 11; i < ri_x.size() + 11; i++) {

        // Because of the way the sampled tiles are constructed 
        // it's possible to go out of bounds so we do an aditional check here
        if (i < measurement.total()) {
            b.push_back(measurement.at<cv::Vec3b>(i)[k] / 255.0f);
        }
        else
        {
            b.push_back(0.0f);
        }
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
    AtAxb2 = AtAxb2 * 255.0f;
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
    float alpha) {
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