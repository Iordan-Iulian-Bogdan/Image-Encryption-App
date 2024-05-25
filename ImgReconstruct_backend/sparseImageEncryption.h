#pragma once
#include <CL/cl.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/photo.hpp>
#include "sparse_representations_solver.h"
#include "sparse_representations_solver.cpp"
#include <tuple>
#include <set>
#include <map>

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

encryptionImage encryptImage(cv::Mat img, int TILE_SIZE, std::string passphrase);
cv::Mat decryptImage(encryptionImage img, std::string passphrase);
