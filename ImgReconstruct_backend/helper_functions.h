#pragma once
#include <CL/cl.h>
#include <vector>
#include <tuple>
#include <iostream>
#include <fstream>

//function to multiply matrices using AVX512 and multithreading 
//void multiplyMatricesAVX512(const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& C, size_t numRowsA, size_t numColsA, size_t numColsB);
//void multiplyMatrixVectorAVX512(const std::vector<float>& matrix, const std::vector<float>& vector, std::vector<float>& result, int numRows, int numCols);
//void multiplyMatrices(const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& result, size_t rows_A, size_t cols_A, size_t cols_B);
//void multiplyVectorScalarAVX512(std::vector<float>& vec, float scalar);
//void multiplyVectorScalar(std::vector<float>& vec, float scalar);
void matrix_vector_mult_avx512(const std::vector<float>& matrix, const std::vector<float>& vector, std::vector<float>& result, size_t rows, size_t cols);
//void matrix_matrix_mult_avx512(const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& C, size_t rowsA, size_t colsA, size_t colsB);
float eigen_aprox_polynomial(uint32_t x);
int retAvailableTile(std::vector<int>& array_of_images);

void runVectorAddKernel(cl_context context, cl_command_queue queue, cl_kernel kernel,
    cl_mem bufferA, cl_mem bufferB, cl_mem bufferC, int length);

void display_opencl_info();

//	struct used to store the encrypted data and how it was formated
struct encryptionImage {
	int TILE_SIZE, original_width, original_height, processed_width, processed_height;
	std::vector<float> data_array;
};