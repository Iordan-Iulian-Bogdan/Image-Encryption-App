#pragma once
//#include <CL/cl.h>
#include <CL/cl.hpp>
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
#define CPU_ACCELERATION 3 

typedef void (*StatusCallback)(const char*);

encryptionImage encryptImage(StatusCallback callback, cv::Mat img, int TILE_SIZE, std::string passphrase, int acceleration, int threads = 1);
void decryptImage(StatusCallback callback, cv::Mat& img_out, encryptionImage img, std::string passphrase, int acceleration, int threads = 1, int iterations = 300, bool removeNoise = false, tile_range range = { 0, 0 });


void encryptAndWriteFile(StatusCallback callback, const char* input, const char* output, const char* passphrase, int TILE_SIZE, int acceleration, int threads, bool upscaling_enable);
void decryptAndWriteFile(StatusCallback callback, const char* input, const char* output, const char* passphrase, int acceleration, int threads, int iterations, bool removeNoise);
void deleteOriginalImage(StatusCallback callback, const char* input, bool scramble);
