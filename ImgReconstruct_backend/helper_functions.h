#pragma once
#include <CL/cl.hpp>
#include <vector>
#include <tuple>
#include <iostream>
#include <fstream>
#include <string>
#include <map>
#include <string>
#include <map>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/photo.hpp>

//	struct used to store the encrypted data and how it was formated
struct encryptionImage {
	int TILE_SIZE, num_tiles, original_width, original_height, processed_width, processed_height;
	std::vector<float> data_array;
};

struct tile_range {
	int first, last;
};

//	struct used to store openCL context
struct openCLContext {
	cl::Platform platform;
	cl::Device device;
	cl::Context context;
	cl::CommandQueue queue;
	cl::Program program;
	cl_int err;
};

typedef void (*StatusCallback)(const char*);

void matrix_vector_mult_avx512(const std::vector<float>& matrix, const std::vector<float>& vector, std::vector<float>& result, size_t rows, size_t cols);
void matrix_mult_avx512(const std::vector<float>& matrixA, const std::vector<float>& matrixB, std::vector<float>& result, size_t rowsA, size_t colsA, size_t colsB);
float eigen_aprox_polynomial(uint32_t x);
int retAvailableTile(std::vector<int>& array_of_images);
void runVectorAddKernel(cl_context context, cl_command_queue queue, cl_kernel kernel, cl_mem bufferA, cl_mem bufferB, cl_mem bufferC, int length);
void writeToFile(const std::string& filename, const encryptionImage& img);
void readFromFile(const std::string& filename, encryptionImage& img);
unsigned long get_free_core(std::string source);
const char* replaceSubstring(const char* input, const char* oldSubstring, const char* newSubstring);

inline int mat_vec_mul_GPU(cl::Buffer& buffer_mat, cl::Buffer& buffer_vec, cl::Buffer& buffer_res, int n, int m, openCLContext cl_data, std::map<std::string, cl::Kernel> kernels) {
	kernels["mat_vec_mul_gpu_fp32"].setArg(0, buffer_mat);
	kernels["mat_vec_mul_gpu_fp32"].setArg(1, buffer_vec);
	kernels["mat_vec_mul_gpu_fp32"].setArg(2, buffer_res);
	kernels["mat_vec_mul_gpu_fp32"].setArg(3, m);
	kernels["mat_vec_mul_gpu_fp32"].setArg(4, n);

	return cl_data.queue.enqueueNDRangeKernel(kernels["mat_vec_mul_gpu_fp32"], cl::NullRange, cl::NDRange(m, 1));
}
inline int vec_sub_GPU(cl::Buffer buffer_res, cl::Buffer buffer_vec, int m, openCLContext cl_data, std::map<std::string, cl::Kernel> kernels) {
	kernels["vec_sub_gpu_sp"].setArg(0, buffer_res);
	kernels["vec_sub_gpu_sp"].setArg(1, buffer_vec);
	
	return cl_data.queue.enqueueNDRangeKernel(kernels["vec_sub_gpu_sp"], cl::NullRange, cl::NDRange(m));
}
inline int vec_scalar_GPU(cl::Buffer buffer_res, float scalar, int m, openCLContext cl_data, std::map<std::string, cl::Kernel> kernels) {
	kernels["vec_scalar_gpu_sp"].setArg(0, buffer_res);
	kernels["vec_scalar_gpu_sp"].setArg(1, scalar);

	return cl_data.queue.enqueueNDRangeKernel(kernels["vec_scalar_gpu_sp"], cl::NullRange, cl::NDRange(m));
}
inline int shrink_GPU(cl::Buffer buffer_res, float scalar, int m, openCLContext cl_data, std::map<std::string, cl::Kernel> kernels) {

	kernels["shrink_gpu_sp"].setArg(0, buffer_res);
	kernels["shrink_gpu_sp"].setArg(1, scalar);

	return cl_data.queue.enqueueNDRangeKernel(kernels["shrink_gpu_sp"], cl::NullRange, cl::NDRange(m));
}