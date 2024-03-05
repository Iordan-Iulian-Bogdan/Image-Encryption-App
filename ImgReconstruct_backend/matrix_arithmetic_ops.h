#pragma once
#include <vector>

//function to multiply matrices using AVX512 and multithreading 
void multiplyMatricesAVX512(const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& C, size_t numRowsA, size_t numColsA, size_t numColsB);
void multiplyMatrixVectorAVX512(const std::vector<float>& matrix, const std::vector<float>& vector, std::vector<float>& result, int numRows, int numCols);
void multiplyMatrices(const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& result, size_t rows_A, size_t cols_A, size_t cols_B);
void multiplyVectorScalarAVX512(std::vector<float>& vec, float scalar);
void multiplyVectorScalar(std::vector<float>& vec, float scalar);