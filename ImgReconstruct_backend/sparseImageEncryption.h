#pragma once
#include <CL/cl.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/photo.hpp>
#include <tuple>
#include <set>
#include <map>
#include "Matrix.h"
#include "Matrix.cpp"

// is going to be the fastest option if the CPU is decently fast
#define HYBRID_ACCELERATION 1 
// possibly the fastest if the TILE_SIZE is large
#define GPU_ACCELERATION 2 
// could be the fastest is TILE_SIZE is small and the GPU is slow like an integrated one
// porbably the fastest to use for the encryption step
#define CPU_ONLY_ACCELERATION 3 

//	struct used to store the encrypted data and how it was formated
struct encryptionImage {
	int TILE_SIZE, original_width, original_height, processed_width, processed_height;
	std::vector<float> data_array;
};

struct openCLContext {
	cl_platform_id platform;
	cl_device_id device;
	cl_context context;
	cl_command_queue queue;
	cl_command_queue device_queue;
	cl_program program;
	cl_int err;
};

encryptionImage encryptImage(cv::Mat img, int TILE_SIZE, std::string passphrase, int acceleration, int threads = 1);
cv::Mat decryptImage(encryptionImage img, std::string passphrase, int acceleration, int threads = 1, int iterations = 300, bool removeNoise = false);
