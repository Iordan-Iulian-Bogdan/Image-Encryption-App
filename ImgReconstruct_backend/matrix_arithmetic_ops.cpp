#include "matrix_arithmetic_ops.h"
#include <omp.h>
#include <immintrin.h>
#include <tuple>

void multiplyMatricesAVX512(const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& C, size_t numRowsA, size_t numColsA, size_t numColsB) {

    const int64_t avx512RegisterSize = 16;

    int numThreads = omp_get_max_threads();
    #pragma omp parallel for num_threads(numThreads) schedule(dynamic)
    for (int64_t i = 0; i < numRowsA; ++i) {
        int64_t j;
        for (j = 0; j <= numColsB - avx512RegisterSize; j += avx512RegisterSize) {
            __m512 sum = _mm512_setzero_ps();
            for (int64_t k = 0; k < numColsA; ++k) {
                __m512 a = _mm512_set1_ps(A[i * numColsA + k]);
                __m512 b = _mm512_loadu_ps(&B[k * numColsB + j]);
                sum = _mm512_fmadd_ps(a, b, sum);
            }
            _mm512_storeu_ps(&C[i * numColsB + j], sum);
        }

        // Handle remaining columns using scalar or smaller vectorization
        for (; j < numColsB; ++j) {
            float sum = 0.0f;
            for (int64_t k = 0; k < numColsA; ++k) {
                sum += A[i * numColsA + k] * B[k * numColsB + j];
            }
            C[i * numColsB + j] = sum;
        }
    }
}

void multiplyMatrixVectorAVX512(const std::vector<float>& matrix, const std::vector<float>& vector, std::vector<float>& result, int numRows, int numCols) {

    const int avx512RegisterSize = 16;
    int numThreads = omp_get_max_threads();
    #pragma omp parallel for num_threads(numThreads) schedule(dynamic)
    for (int i = 0; i < numRows; ++i) {
        __m512 sum = _mm512_setzero_ps();

        int j;
        for (j = 0; j <= numCols - avx512RegisterSize; j += avx512RegisterSize) {
            __m512 matRow = _mm512_loadu_ps(&matrix[i * numCols + j]);
            __m512 vecPart = _mm512_loadu_ps(&vector[j]);
            sum = _mm512_fmadd_ps(matRow, vecPart, sum);
        }

        // Handle the remaining elements using scalar instructions
        float remainingSum = 0.0f;
        for (; j < numCols; ++j) {
            remainingSum += matrix[i * numCols + j] * vector[j];
        }

        // Horizontal sum of the AVX-512 register
        __m256 sum256 = _mm256_add_ps(_mm512_extractf32x8_ps(sum, 0), _mm512_extractf32x8_ps(sum, 1));
        sum256 = _mm256_hadd_ps(sum256, sum256);
        sum256 = _mm256_hadd_ps(sum256, sum256);

        // Add the remaining sum and store the result
        result[i] = _mm256_cvtss_f32(sum256) + remainingSum;
    }
}

void multiplyVectorScalar(std::vector<float>& vec, const float scalar) {
    int numThreads = omp_get_max_threads();
    #pragma omp parallel for num_threads(numThreads) schedule(dynamic)
    for (int64_t i = 0; i < vec.size(); i++){
        vec[i] *= scalar;
    }
}

void multiplyVectorScalarAVX512(std::vector<float>& vec, const float scalar) {
    const size_t vecSize = vec.size();
    const size_t remainder = vecSize % 16;  // 16 floats per AVX-512 register

    // Process the majority of the vector using AVX-512
    int numThreads = omp_get_max_threads();
    #pragma omp parallel for num_threads(numThreads) schedule(dynamic)
    for (int64_t i = 0; i < vecSize - remainder; i += 16) {
        __m512 v1 = _mm512_loadu_ps(&vec[i]);
        __m512 scalarVec = _mm512_set1_ps(scalar);
        __m512 result = _mm512_mul_ps(v1, scalarVec);
        _mm512_storeu_ps(&vec[i], result);
    }

    // Process the remaining elements using scalar multiplication
    for (int64_t i = vecSize - remainder; i < vecSize; ++i) {
        vec[i] *= scalar;
    }
}

void multiplyMatrices(const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& result,
    size_t rows_A, size_t cols_A, size_t cols_B) {

    // Use scalar instructions for the remaining elements
    int numThreads = omp_get_max_threads();
    #pragma omp parallel for num_threads(numThreads) schedule(dynamic)
    for (int64_t i = 0; i < rows_A; ++i) {
        for (int64_t j = 0; j < cols_B; ++j) {
            float sum = 0.0f;
            for (int64_t k = 0; k < cols_A; ++k) {
                sum += A[i * cols_A + k] * B[k * cols_B + j];
            }
            result[i * cols_B + j] += sum;
        }
    }
}

void mat_vec_mul_gpu(std::tuple <cl::Context, cl::CommandQueue, cl::Program> context, cl::Buffer& buffer_mat, cl::Buffer& buffer_vec, cl::Buffer& buffer_res, int rows, int cols)
{
    cl::Kernel matrixVectorMultiply;
    matrixVectorMultiply = cl::Kernel(std::get<2>(context), "mat_vec_mul_gpu_fp32");
    matrixVectorMultiply.setArg(0, buffer_mat);
    matrixVectorMultiply.setArg(1, buffer_vec);
    matrixVectorMultiply.setArg(2, buffer_res);
    matrixVectorMultiply.setArg(3, (int)rows);
    matrixVectorMultiply.setArg(4, (int)cols);
    std::get<1>(context).enqueueNDRangeKernel(matrixVectorMultiply, cl::NullRange, cl::NDRange((int)(rows)));
    std::get<1>(context).finish();
}

void mat_mat_mul_gpu(std::tuple <cl::Context, cl::CommandQueue, cl::Program> context, cl::Buffer& buffer_mat1, cl::Buffer& buffer_mat2, cl::Buffer& buffer_res, int rows, int cols)
{
    const int TS = 32;
    const int WPT = 8;

    cl::Kernel kernel_mat_mat_mul_gpu;
    kernel_mat_mat_mul_gpu = cl::Kernel(std::get<2>(context), "mat_mat_mul_gpu_sp");
    kernel_mat_mat_mul_gpu.setArg(0, (int)rows);
    kernel_mat_mat_mul_gpu.setArg(1, (int)cols);
    kernel_mat_mat_mul_gpu.setArg(2, buffer_mat1);
    kernel_mat_mat_mul_gpu.setArg(3, buffer_mat2);
    kernel_mat_mat_mul_gpu.setArg(4, buffer_res);
    std::get<1>(context).enqueueNDRangeKernel(kernel_mat_mat_mul_gpu, cl::NullRange, cl::NDRange((int)rows, (int)rows / WPT), cl::NDRange(TS, TS / WPT));
    std::get<1>(context).finish();
}

void mat_mat_mul_gpu2(std::tuple <cl::Context, cl::CommandQueue, cl::Program> context, cl::Buffer& buffer_mat1, cl::Buffer& buffer_mat2, cl::Buffer& buffer_res, int rows_mat1, int cols_mat1, int cols_mat2)
{
    cl::Kernel kernel_mat_mat_mul_gpu;
    kernel_mat_mat_mul_gpu = cl::Kernel(std::get<2>(context), "matrixMultiplication1");
    kernel_mat_mat_mul_gpu.setArg(0, buffer_mat1);
    kernel_mat_mat_mul_gpu.setArg(1, buffer_mat2);
    kernel_mat_mat_mul_gpu.setArg(2, buffer_res);
    kernel_mat_mat_mul_gpu.setArg(3, rows_mat1);
    kernel_mat_mat_mul_gpu.setArg(4, cols_mat1);
    kernel_mat_mat_mul_gpu.setArg(5, cols_mat2);

    cl::NDRange globalSize(cols_mat2, rows_mat1);
    std::get<1>(context).enqueueNDRangeKernel(kernel_mat_mat_mul_gpu, cl::NullRange, globalSize);
    std::get<1>(context).finish();
}

std::tuple <cl::Context, cl::CommandQueue, cl::Program> creat_opencl_context()
{
    std::vector<cl::Platform> all_platforms;
    cl::Platform::get(&all_platforms);
    cl::Platform default_platform = all_platforms[0];
    std::vector<cl::Device> all_devices;
    default_platform.getDevices(CL_DEVICE_TYPE_GPU, &all_devices);
    cl::Device default_device = all_devices[0];
    cl::Context context(default_device);
    std::ifstream src("gpu_kernels.cl");
    std::string str((std::istreambuf_iterator<char>(src)), std::istreambuf_iterator<char>());
    cl::Program::Sources sources;
    sources.push_back({ str.c_str(),str.length() });
    cl::Program program(context, sources);
    program.build({ default_device });
    cl::CommandQueue queue(context, default_device);
    //(context, default_device);

    std::tuple <cl::Context, cl::CommandQueue, cl::Program> cl_context(context, queue, program);

    return { context, queue, program };
}