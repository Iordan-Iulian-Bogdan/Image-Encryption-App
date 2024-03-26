#pragma once
#define CL_USE_DEPRECATED_OPENCL_2_0_APIS
#include <CL/cl.hpp>
#include <vector>
#include <tuple>
#include <iostream>
#include <fstream>

//function to multiply matrices using AVX512 and multithreading 
void multiplyMatricesAVX512(const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& C, size_t numRowsA, size_t numColsA, size_t numColsB);
void multiplyMatrixVectorAVX512(const std::vector<float>& matrix, const std::vector<float>& vector, std::vector<float>& result, int numRows, int numCols);
void multiplyMatrices(const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& result, size_t rows_A, size_t cols_A, size_t cols_B);
void multiplyVectorScalarAVX512(std::vector<float>& vec, float scalar);
void multiplyVectorScalar(std::vector<float>& vec, float scalar);

//opencl functions
std::tuple <cl::Context, cl::CommandQueue, cl::Program> creat_opencl_context();
void mat_vec_mul_gpu(std::tuple <cl::Context, 
						cl::CommandQueue, 
						cl::Program> context,
						cl::Buffer& buffer_mat, 
						cl::Buffer& buffer_vec, 
						cl::Buffer& buffer_res, 
						int rows, 
						int cols);

void mat_mat_mul_gpu(std::tuple <cl::Context,
	cl::CommandQueue,
	cl::Program> context,
	cl::Buffer& buffer_mat1,
	cl::Buffer& buffer_mat2,
	cl::Buffer& buffer_res,
	int rows,
	int cols);

void mat_mat_mul_gpu2(std::tuple <cl::Context,
	cl::CommandQueue,
	cl::Program> context,
	cl::Buffer& buffer_mat1,
	cl::Buffer& buffer_mat2,
	cl::Buffer& buffer_res,
	int rows_mat1,
	int cols_mat1,
	int cols_mat2);