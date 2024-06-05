#include "matrix_arithmetic_ops.h"
#include <omp.h>
#include <immintrin.h>
#include <tuple>
#include <fstream>
#include <sstream>

void transpose_matrix_avx512(const std::vector<float>& input, std::vector<float>& output, size_t rows, size_t cols) {

}

float horizontal_add(__m512 vec) {
    // Perform horizontal addition by reducing the elements
    __m256 low = _mm512_castps512_ps256(vec);
    __m256 high = _mm512_extractf32x8_ps(vec, 1);
    __m256 sum = _mm256_add_ps(low, high);

    __m128 low128 = _mm256_castps256_ps128(sum);
    __m128 high128 = _mm256_extractf128_ps(sum, 1);
    __m128 sum128 = _mm_add_ps(low128, high128);

    __m128 shuf = _mm_movehdup_ps(sum128);
    __m128 sums = _mm_add_ps(sum128, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);

    return _mm_cvtss_f32(sums);
}

void matrix_vector_mult_avx512(const std::vector<float>& matrix, const std::vector<float>& vector, std::vector<float>& result, size_t rows, size_t cols) {
    result.resize(rows, 0.0f);
    int numThreads = omp_get_max_threads();
    #pragma omp parallel for num_threads(numThreads / 2) schedule(static)
    for (int64_t i = 0; i < rows; ++i) {
        // Initialize the result for the current row to zero
        __m512 sum = _mm512_setzero_ps();

        for (int64_t j = 0; j < cols; j += 16) {
            // Load 16 elements from the current row of the matrix
            __m512 mat_row = _mm512_loadu_ps(matrix.data() + i * cols + j);

            // Load 16 elements from the vector
            __m512 vec = _mm512_loadu_ps(vector.data() + j);

            // Multiply the matrix row by the vector and accumulate using FMA
            sum = _mm512_fmadd_ps(mat_row, vec, sum);
        }

        // Perform horizontal addition to get the final result for the current row
        result[i] = horizontal_add(sum);
    }
}

void matrix_matrix_mult_avx512(const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& C, size_t rowsA, size_t colsA, size_t colsB) {
    int numThreads = omp_get_max_threads();
    //#pragma omp parallel for num_threads(numThreads) schedule(dynamic)
    for (int64_t i = 0; i < rowsA; ++i) {
        for (int64_t k = 0; k < colsA; ++k) {
            // Load one element from A
            __m512 a_elem = _mm512_set1_ps(A[i * colsA + k]);

            for (int64_t j = 0; j < colsB; j += 16) {
                // Load 16 elements from the current row of B
                __m512 b_row = _mm512_loadu_ps(B.data() + k * colsB + j);

                // Load 16 elements from the current row of C
                __m512 c_row = _mm512_loadu_ps(C.data() + i * colsB + j);

                // Multiply and accumulate using FMA
                c_row = _mm512_fmadd_ps(a_elem, b_row, c_row);

                // Store the result back into C
                _mm512_storeu_ps(C.data() + i * colsB + j, c_row);
            }
        }
    }
}

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

void display_opencl_info()
{
    // Get the number of platforms
    cl_uint numPlatforms = 0;
    clGetPlatformIDs(0, NULL, &numPlatforms);

    // Get platform IDs
    std::vector<cl_platform_id> platforms(numPlatforms);
    clGetPlatformIDs(numPlatforms, platforms.data(), NULL);

    std::cout << "Number of OpenCL Platforms: " << numPlatforms << std::endl;
    std::cout << "-----------------------------------" << std::endl;

    // Iterate over platforms
    for (cl_uint i = 0; i < numPlatforms; ++i) {
        // Display platform information
        char platformName[1024];
        clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, sizeof(platformName), platformName, NULL);
        std::cout << "Platform " << i << " Name: " << platformName << std::endl;

        clGetPlatformInfo(platforms[i], CL_PLATFORM_VENDOR, sizeof(platformName), platformName, NULL);
        std::cout << "Platform " << i << " Vendor: " << platformName << std::endl;

        // Get the number of devices for this platform
        cl_uint numDevices = 0;
        clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, NULL, &numDevices);

        // Get device IDs
        std::vector<cl_device_id> devices(numDevices);
        clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, numDevices, devices.data(), NULL);

        std::cout << "Number of Devices: " << numDevices << std::endl;

        // Iterate over devices
        for (cl_uint j = 0; j < numDevices; ++j) {
            // Display device information
            char deviceName[1024];
            clGetDeviceInfo(devices[j], CL_DEVICE_NAME, sizeof(deviceName), deviceName, NULL);
            std::cout << "  Device " << j << " Name: " << deviceName << std::endl;

            clGetDeviceInfo(devices[j], CL_DEVICE_VENDOR, sizeof(deviceName), deviceName, NULL);
            std::cout << "  Device " << j << " Vendor: " << deviceName << std::endl;

            cl_device_type deviceType;
            clGetDeviceInfo(devices[j], CL_DEVICE_TYPE, sizeof(deviceType), &deviceType, NULL);
            if (deviceType & CL_DEVICE_TYPE_CPU)
                std::cout << "  Device " << j << " Type: CPU" << std::endl;
            if (deviceType & CL_DEVICE_TYPE_GPU)
                std::cout << "  Device " << j << " Type: GPU" << std::endl;
            if (deviceType & CL_DEVICE_TYPE_ACCELERATOR)
                std::cout << "  Device " << j << " Type: Accelerator" << std::endl;

            cl_uint maxComputeUnits;
            clGetDeviceInfo(devices[j], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(maxComputeUnits), &maxComputeUnits, NULL);
            std::cout << "  Device " << j << " Max Compute Units: " << maxComputeUnits << std::endl;

            cl_uint maxWorkGroupSize;
            clGetDeviceInfo(devices[j], CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(maxWorkGroupSize), &maxWorkGroupSize, NULL);
            std::cout << "  Device " << j << " Max Work Group Size: " << maxWorkGroupSize << std::endl;

            cl_ulong globalMemSize;
            clGetDeviceInfo(devices[j], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(globalMemSize), &globalMemSize, NULL);
            std::cout << "  Device " << j << " Global Memory Size: " << globalMemSize / (1024 * 1024) << " MB" << std::endl;


            cl_ulong onDeviceQueue;
            clGetDeviceInfo(devices[j], CL_DEVICE_MAX_ON_DEVICE_EVENTS, sizeof(onDeviceQueue), &onDeviceQueue, NULL);
            std::cout << "  Device " << j << " CL_DEVICE_MAX_ON_DEVICE_EVENTS: " << onDeviceQueue << std::endl;

            std::cout << std::endl;
        }
        std::cout << "-----------------------------------" << std::endl;
    }
}

void runVectorAddKernel(cl_context context, cl_command_queue queue, cl_kernel kernel,
    cl_mem bufferA, cl_mem bufferB, cl_mem bufferC, int length) {
    // Set kernel arguments
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufferA);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufferB);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufferC);
    clSetKernelArg(kernel, 3, sizeof(int), &length);

    // Execute the kernel
    size_t globalWorkSize = length;
    clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalWorkSize, NULL, 0, NULL, NULL);
}