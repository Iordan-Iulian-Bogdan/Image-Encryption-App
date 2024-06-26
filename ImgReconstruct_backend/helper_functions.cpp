#include "helper_functions.h"
#include <omp.h>
#include <immintrin.h>
#include <tuple>
#include <fstream>
#include <sstream>

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
    #pragma omp parallel for num_threads(numThreads / 2) schedule(dynamic)
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

float eigen_aprox_polynomial(uint32_t x) {
    //return 5.0139 * x * x - 4.9005 * x + 144.3337;
    return (0.0035 * x * x * x) + (4.0195 * x * x) + (78.5388 * x) - 1.7564e+03;
}

// this searches for the first tile that hasn't been processed yet and returns it's index
int retAvailableTile(std::vector<int>& array_of_images) {
    int index = 0;
    bool found = false;
    for (int i = 0; i < array_of_images.size(); i++) {
        if (array_of_images[i] == 1) {
            array_of_images[i] = 0;
            index = i;
            found = true;
            break;
        }
    }

    if (found) {
        return index;
    }
    else {
        return -1;
    }
}
