#include "matrix_arithmetic_ops.h"
#include <omp.h>
#include <immintrin.h>

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
    for (size_t i = 0; i < rows_A; ++i) {
        for (size_t j = 0; j < cols_B; ++j) {
            float sum = 0.0f;
            for (size_t k = 0; k < cols_A; ++k) {
                sum += A[i * cols_A + k] * B[k * cols_B + j];
            }
            result[i * cols_B + j] += sum;
        }
    }
}