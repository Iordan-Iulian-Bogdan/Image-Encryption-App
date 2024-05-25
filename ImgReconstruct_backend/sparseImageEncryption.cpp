#include "sparseImageEncryption.h"
#include "AVX_functions.h"

std::vector <float> ADM_gpu(map<string, cl_mem>& buffers,
	int A_cols, int A_rows, float max_eig, float beta, float tau, int iterations,
	cl_context context, cl_command_queue queue, map<string, cl_kernel> kernels, cl_device_id device, cl_program program, int index1, int index2)
{
	cl_queue_properties properties[] = { CL_QUEUE_PROPERTIES, (cl_command_queue_properties)(CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE | CL_QUEUE_ON_DEVICE | CL_QUEUE_ON_DEVICE_DEFAULT), 0 };

	int n = A_cols;
	int m = A_rows;

	int err = 0;
	size_t globalSize_m[2] = { m, 1 };
	size_t globalSize_n[2] = { n, 1 };

	float gamma = 1.99f - (tau * max_eig);
	float f = 0.0f;

	std::vector<float> x1(n);

	buffers["buffer_res" + std::to_string(index1) + "_" + std::to_string(index2)] = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * n, NULL, NULL);
	buffers["buffer_y" + std::to_string(index1) + "_" + std::to_string(index2)] = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * m, NULL, NULL);
	buffers["buffer_x" + std::to_string(index1) + "_" + std::to_string(index2)] = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * n, NULL, NULL);
	buffers["buffer_res_aux" + std::to_string(index1) + "_" + std::to_string(index2)] = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * n, NULL, NULL);
	buffers["buffer_count" + std::to_string(index1) + "_" + std::to_string(index2)] = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int), NULL, NULL);
	clFinish(queue);


	size_t globalSize_g[1] = { 1 };
	cl_event event1;

	for (int i = 0; i < iterations; i++)
	{

		clSetKernelArg(kernels["mat_vec_mul_gpu_fp32"], 0, sizeof(cl_mem), &buffers["buffer_A"]);
		clSetKernelArg(kernels["mat_vec_mul_gpu_fp32"], 1, sizeof(cl_mem), &buffers["buffer_x" + std::to_string(index1) + "_" + std::to_string(index2)]);
		clSetKernelArg(kernels["mat_vec_mul_gpu_fp32"], 2, sizeof(cl_mem), &buffers["buffer_res" + std::to_string(index1) + "_" + std::to_string(index2)]);
		clSetKernelArg(kernels["mat_vec_mul_gpu_fp32"], 3, sizeof(int), &m);
		clSetKernelArg(kernels["mat_vec_mul_gpu_fp32"], 4, sizeof(int), &n);
		err = clEnqueueNDRangeKernel(queue, kernels["mat_vec_mul_gpu_fp32"], 1, NULL, globalSize_m, NULL, 0, NULL, NULL);
		clFinish(queue);

		clSetKernelArg(kernels["vec_sub_gpu_sp"], 0, sizeof(cl_mem), &buffers["buffer_res" + std::to_string(index1) + "_" + std::to_string(index2)]);
		clSetKernelArg(kernels["vec_sub_gpu_sp"], 1, sizeof(cl_mem), &buffers["buffer_b" + std::to_string(index1) + "_" + std::to_string(index2)]);
		err = clEnqueueNDRangeKernel(queue, kernels["vec_sub_gpu_sp"], 1, NULL, globalSize_m, NULL, 0, NULL, NULL);
		clFinish(queue);

		f = (1 / beta);
		clSetKernelArg(kernels["vec_scalar_gpu_sp"], 0, sizeof(cl_mem), &buffers["buffer_y" + std::to_string(index1) + "_" + std::to_string(index2)]);
		clSetKernelArg(kernels["vec_scalar_gpu_sp"], 1, sizeof(float), &f);
		err = clEnqueueNDRangeKernel(queue, kernels["vec_scalar_gpu_sp"], 1, NULL, globalSize_m, NULL, 0, NULL, NULL);
		clFinish(queue);

		clSetKernelArg(kernels["vec_sub_gpu_sp"], 0, sizeof(cl_mem), &buffers["buffer_res" + std::to_string(index1) + "_" + std::to_string(index2)]);
		clSetKernelArg(kernels["vec_sub_gpu_sp"], 1, sizeof(cl_mem), &buffers["buffer_y" + std::to_string(index1) + "_" + std::to_string(index2)]);
		err = clEnqueueNDRangeKernel(queue, kernels["vec_sub_gpu_sp"], 1, NULL, globalSize_m, NULL, 0, NULL, NULL);
		clFinish(queue);

		clSetKernelArg(kernels["mat_vec_mul_gpu_fp32"], 0, sizeof(cl_mem), &buffers["buffer_A_t"]);
		clSetKernelArg(kernels["mat_vec_mul_gpu_fp32"], 1, sizeof(cl_mem), &buffers["buffer_res" + std::to_string(index1) + "_" + std::to_string(index2)]);
		clSetKernelArg(kernels["mat_vec_mul_gpu_fp32"], 2, sizeof(cl_mem), &buffers["buffer_res_aux" + std::to_string(index1) + "_" + std::to_string(index2)]);
		clSetKernelArg(kernels["mat_vec_mul_gpu_fp32"], 3, sizeof(int), &n);
		clSetKernelArg(kernels["mat_vec_mul_gpu_fp32"], 4, sizeof(int), &m);
		err = clEnqueueNDRangeKernel(queue, kernels["mat_vec_mul_gpu_fp32"], 1, NULL, globalSize_n, NULL, 0, NULL, NULL);
		clFinish(queue);

		clSetKernelArg(kernels["vec_sub_gpu_sp"], 0, sizeof(cl_mem), &buffers["buffer_x" + std::to_string(index1) + "_" + std::to_string(index2)]);
		clSetKernelArg(kernels["vec_sub_gpu_sp"], 1, sizeof(cl_mem), &buffers["buffer_res_aux" + std::to_string(index1) + "_" + std::to_string(index2)]);
		err = clEnqueueNDRangeKernel(queue, kernels["vec_sub_gpu_sp"], 1, NULL, globalSize_n, NULL, 0, NULL, NULL);
		clFinish(queue);

		f = (tau / beta);
		clSetKernelArg(kernels["shrink_gpu_sp"], 0, sizeof(cl_mem), &buffers["buffer_x" + std::to_string(index1) + "_" + std::to_string(index2)]);
		clSetKernelArg(kernels["shrink_gpu_sp"], 1, sizeof(float), &f);
		err = clEnqueueNDRangeKernel(queue, kernels["shrink_gpu_sp"], 1, NULL, globalSize_n, NULL, 0, NULL, NULL);

		clSetKernelArg(kernels["mat_vec_mul_gpu_fp32"], 0, sizeof(cl_mem), &buffers["buffer_A"]);
		clSetKernelArg(kernels["mat_vec_mul_gpu_fp32"], 1, sizeof(cl_mem), &buffers["buffer_x" + std::to_string(index1) + "_" + std::to_string(index2)]);
		clSetKernelArg(kernels["mat_vec_mul_gpu_fp32"], 2, sizeof(cl_mem), &buffers["buffer_res_aux" + std::to_string(index1) + "_" + std::to_string(index2)]);
		clSetKernelArg(kernels["mat_vec_mul_gpu_fp32"], 3, sizeof(int), &m);
		clSetKernelArg(kernels["mat_vec_mul_gpu_fp32"], 4, sizeof(int), &n);
		err = clEnqueueNDRangeKernel(queue, kernels["mat_vec_mul_gpu_fp32"], 1, NULL, globalSize_m, NULL, 0, NULL, NULL);
		clFinish(queue);

		clSetKernelArg(kernels["vec_sub_gpu_sp"], 0, sizeof(cl_mem), &buffers["buffer_res_aux" + std::to_string(index1) + "_" + std::to_string(index2)]);
		clSetKernelArg(kernels["vec_sub_gpu_sp"], 1, sizeof(cl_mem), &buffers["buffer_b" + std::to_string(index1) + "_" + std::to_string(index2)]);
		err = clEnqueueNDRangeKernel(queue, kernels["vec_sub_gpu_sp"], 1, NULL, globalSize_m, NULL, 0, NULL, NULL);
		clFinish(queue);

		f = (gamma * beta);
		clSetKernelArg(kernels["vec_scalar_gpu_sp"], 0, sizeof(cl_mem), &buffers["buffer_res_aux" + std::to_string(index1) + "_" + std::to_string(index2)]);
		clSetKernelArg(kernels["vec_scalar_gpu_sp"], 1, sizeof(float), &f);
		err = clEnqueueNDRangeKernel(queue, kernels["vec_scalar_gpu_sp"], 1, NULL, globalSize_m, NULL, 0, NULL, NULL);
		clFinish(queue);

		f = (1 / (1 / beta));
		clSetKernelArg(kernels["vec_scalar_gpu_sp"], 0, sizeof(cl_mem), &buffers["buffer_y" + std::to_string(index1) + "_" + std::to_string(index2)]);
		clSetKernelArg(kernels["vec_scalar_gpu_sp"], 1, sizeof(float), &f);
		err = clEnqueueNDRangeKernel(queue, kernels["vec_scalar_gpu_sp"], 1, NULL, globalSize_m, NULL, 0, NULL, NULL);
		clFinish(queue);

		clSetKernelArg(kernels["vec_sub_gpu_sp"], 0, sizeof(cl_mem), &buffers["buffer_y" + std::to_string(index1) + "_" + std::to_string(index2)]);
		clSetKernelArg(kernels["vec_sub_gpu_sp"], 1, sizeof(cl_mem), &buffers["buffer_res_aux" + std::to_string(index1) + "_" + std::to_string(index2)]);
		err = clEnqueueNDRangeKernel(queue, kernels["vec_sub_gpu_sp"], 1, NULL, globalSize_m, NULL, 0, NULL, NULL);
		clFinish(queue);
	}

	clEnqueueReadBuffer(queue, buffers["buffer_x" + std::to_string(index1) + "_" + std::to_string(index2)], CL_TRUE, 0, sizeof(float) * x1.size(), x1.data(), 0, NULL, NULL);
	
	return x1;
}

void decrypt_data(cv::Mat& out, map<string, cl_mem>& buffers, cl_context context, cl_command_queue queue, map<string, cl_kernel> kernels, uint32_t tile_size, cl_device_id device, cl_program program, float max_eig, int index1, int index2, float seed = 1) {
	cl_int err;

	cv::Mat img = cv::Mat::zeros(cv::Size(tile_size, tile_size), CV_8U);

	int width = img.size[1];
	int height = img.size[0];
	size_t n = width * height;
	size_t m = n;
	int T = 1;
	int k = 0;
	vector<float> res(m);
	vector<float> ek(n);
	vector<float> psi(n);
	vector<float> x1(n);
	vector<float> s1(n);

	cv::Mat reconstructedImg;
	out.convertTo(reconstructedImg, CV_32FC1);

	float beta = 0.000001f;
	float tau = 0.000001f;
	int iterations = 1000;

	vector<float>sol_alt = ADM_gpu(buffers, n, m, max_eig, beta, tau, iterations, context, queue, kernels, device, program, index1, index2);

	s1 = sol_alt;
	float o = s1[n - 1];
	vec_fill(x1, 0.0f);
	vec_fill(ek, 0.0f);
	vec_fill(psi, 0.0f);

	for (int i = 0; i < n; i++)
	{
		vec_fill(ek, 0.0f);
		ek[i] = 1.0f;
		cv::idct(ek, psi, 0);
		vec_scalar_avx(psi, s1[i], T);
		vec_add_avx(x1, psi, T);
	}

	k = 0;

	for (int i = 0; i < img.rows; i++)
	{
		for (int j = 0; j < img.cols; j++)
		{
			reconstructedImg.at<float>(j, i) = x1[k++];
		}
	}
	k = 0;

	reconstructedImg.convertTo(reconstructedImg, CV_8U);
	out = reconstructedImg;
}

void decrypt_image(cv::Mat& out, map<string, cl_mem>& buffers, cl_context context, cl_command_queue queue, map<string, cl_kernel> kernels, uint32_t TILE_SIZE, cl_device_id device, cl_program program, float max_eig, int index1) {

	cv::Mat channels[3];
	cv::split(out, channels);

	for (int i = 0; i < 3; i++) {
		decrypt_data(channels[i], buffers, context, queue, kernels, TILE_SIZE, device, program, max_eig, index1, i);
	}

	cv::merge(channels, 3, out);
}

void decrypt_image_alt(cv::Mat& out, map<string, cl_mem>& buffers, uint32_t TILE_SIZE, float max_eig, int index) {

	cl_platform_id platform;
	cl_device_id device;
	cl_context context;
	cl_command_queue queue;
	cl_program program;
	cl_int err;

	clGetPlatformIDs(1, &platform, NULL);
	clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
	context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
	queue = clCreateCommandQueueWithProperties(context, device, 0, &err);

	std::ifstream kernelFile("gpu_kernels.cl");

	if (!kernelFile.is_open()) {
		std::cerr << "Failed to open kernel file." << std::endl;
	}

	std::string kernelSource((std::istreambuf_iterator<char>(kernelFile)), std::istreambuf_iterator<char>());
	const char* sources = kernelSource.data();
	program = clCreateProgramWithSource(context, 1, &sources, NULL, &err);
	const char options[] = "-cl-std=CL2.0";
	clBuildProgram(program, 1, &device, options, NULL, NULL);

	map<string, cl_kernel> kernels;

	kernels["transpose"] = clCreateKernel(program, "transpose", &err);
	kernels["mat_mat_mul_gpu_sp"] = clCreateKernel(program, "mat_mat_mul_gpu_sp", &err);
	kernels["power_method"] = clCreateKernel(program, "power_method", &err);
	kernels["mat_vec_mul_gpu_fp32"] = clCreateKernel(program, "mat_vec_mul_gpu_fp32", &err);
	kernels["vec_scalar_gpu_sp"] = clCreateKernel(program, "vec_scalar_gpu_sp", &err);
	kernels["vec_sub_gpu_sp"] = clCreateKernel(program, "vec_sub_gpu_sp", &err);
	kernels["shrink_gpu_sp"] = clCreateKernel(program, "shrink_gpu_sp", &err);
	kernels["ADM"] = clCreateKernel(program, "ADM", &err);

	cv::Mat channels[3];
	cv::split(out, channels);

	for (int i = 0; i < 3; i++) {
		decrypt_data(channels[i], buffers, context, queue, kernels, TILE_SIZE, device, program, max_eig, index, i);
	}

	cv::merge(channels, 3, out);

	clReleaseCommandQueue(queue);
	clReleaseContext(context);
	clReleaseDevice(device);
	clReleaseProgram(program);

	for (auto i = kernels.begin(); i != kernels.end(); i++)
		clReleaseKernel(i->second);
}

void generateIDCT(vector<float>& IDCT, int n)
{
	vector<float> ek(n);
	vector<float> psi(n);

	int numThreads = omp_get_max_threads();
	//#pragma omp parallel for num_threads(numThreads) schedule(dynamic)
	for (int i = 0; i < n; i++)
	{
		vec_fill(ek, 0.0f);
		ek[i] = 1.0f;
		cv::idct(ek, psi, 0);

		for (int k = 0; k < n; k = k + 16)
		{
			_mm512_storeu_ps(&IDCT[(i * n) + k], _mm512_loadu_ps(&psi[k]));
		}
	}
}

float generate_max_eig(cl_context context, cl_command_queue queue,
	map<string, cl_mem> buffers, map<string, cl_kernel> kernels, int A_rows, int A_cols)
{
	std::vector<float> max_eig(1);

	int err = 0;

	int m = A_rows;
	int n = A_cols;
	const int TS = 32;
	const int WPT = 8;


	buffers["buffer_X"] = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * m * m, NULL, NULL);
	buffers["buffer_b_k"] = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * m, NULL, NULL);
	buffers["buffer_b_k_res"] = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * m, NULL, NULL);
	buffers["buffer_b_k1"] = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * m, NULL, NULL);
	buffers["buffer_aux"] = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * m, NULL, NULL);
	buffers["buffer_max_eig"] = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float), NULL, NULL);
	clFinish(queue);

	clSetKernelArg(kernels["mat_mat_mul_gpu_sp"], 0, sizeof(int), &m);
	clSetKernelArg(kernels["mat_mat_mul_gpu_sp"], 1, sizeof(int), &n);
	clSetKernelArg(kernels["mat_mat_mul_gpu_sp"], 2, sizeof(cl_mem), &buffers["buffer_A_t"]);
	clSetKernelArg(kernels["mat_mat_mul_gpu_sp"], 3, sizeof(cl_mem), &buffers["buffer_A"]);
	clSetKernelArg(kernels["mat_mat_mul_gpu_sp"], 4, sizeof(cl_mem), &buffers["buffer_X"]);



	size_t globalSize[2] = { (int)m, (int)m / WPT };
	size_t localSize[2] = { TS, TS / WPT };
	err = clEnqueueNDRangeKernel(queue, kernels["mat_mat_mul_gpu_sp"], 2, NULL, globalSize, localSize, 0, NULL, NULL);
	clFinish(queue);

	clSetKernelArg(kernels["power_method"], 0, sizeof(cl_mem), &buffers["buffer_X"]);
	clSetKernelArg(kernels["power_method"], 1, sizeof(cl_mem), &buffers["buffer_b_k"]);
	clSetKernelArg(kernels["power_method"], 2, sizeof(cl_mem), &buffers["buffer_b_k_res"]);
	clSetKernelArg(kernels["power_method"], 3, sizeof(cl_mem), &buffers["buffer_b_k1"]);
	clSetKernelArg(kernels["power_method"], 4, sizeof(cl_mem), &buffers["buffer_aux"]);
	clSetKernelArg(kernels["power_method"], 5, sizeof(cl_mem), &buffers["buffer_max_eig"]);
	clSetKernelArg(kernels["power_method"], 6, sizeof(int), &m);
	clSetKernelArg(kernels["power_method"], 7, sizeof(int), &m);

	size_t globalWorkSize = 1;
	err = clEnqueueNDRangeKernel(queue, kernels["power_method"], 1, NULL, &globalWorkSize, NULL, 0, NULL, NULL);

	clEnqueueReadBuffer(queue, buffers["buffer_max_eig"], CL_TRUE, 0, sizeof(float), max_eig.data(), 0, NULL, NULL);
	clFinish(queue);

	clReleaseMemObject(buffers["buffer_X"]);
	clReleaseMemObject(buffers["buffer_b_k"]);
	clReleaseMemObject(buffers["buffer_b_k_res"]);
	clReleaseMemObject(buffers["buffer_b_k1"]);
	clReleaseMemObject(buffers["buffer_aux"]);
	clReleaseMemObject(buffers["buffer_max_eig"]);

	return max_eig[0];
}

float generate_dictionary(map<string, cl_mem>& buffers, cl_context context, cl_command_queue queue, map<string, cl_kernel> kernels, uint32_t size, cl_device_id device, cl_program program, float seed = 1) {
	cl_int err;

	int width = size;
	int height = size;
	size_t n = width * height;
	size_t m = n;
	int T = 12;
	int k = 0;

	buffers["buffer_A"] = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * n * m, NULL, NULL);

	k = 0;
	float sum = 0.0f;

	vector<float> IDCT_alt(n * n, 0.0f);
	generateIDCT(IDCT_alt, n);
	Matrix<float> transpose(n, n);
	transpose.data = IDCT_alt;
	transpose.data = transpose.getTransposedMatrix().data;
	buffers["buffer_IDCT"] = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * IDCT_alt.size(), NULL, NULL);
	clEnqueueWriteBuffer(queue, buffers["buffer_IDCT"], CL_TRUE, 0, sizeof(float) * IDCT_alt.size(), transpose.data.data(), 0, NULL, NULL);
	IDCT_alt = std::vector<float>();
	transpose.data = std::vector<float>();
	clFinish(queue);

	size_t globalSize[1] = { m };


	const int TS = 32;
	const int WPT = 8;

	clSetKernelArg(kernels["mat_mat_mul_gpu_sp"], 0, sizeof(int), &m);
	clSetKernelArg(kernels["mat_mat_mul_gpu_sp"], 1, sizeof(int), &n);
	clSetKernelArg(kernels["mat_mat_mul_gpu_sp"], 2, sizeof(cl_mem), &buffers["buffer_IDCT"]);
	clSetKernelArg(kernels["mat_mat_mul_gpu_sp"], 3, sizeof(cl_mem), &buffers["buffer_phi"]);
	clSetKernelArg(kernels["mat_mat_mul_gpu_sp"], 4, sizeof(cl_mem), &buffers["buffer_A"]);

	size_t globalSize1[2] = { (int)n, (int)n / WPT };
	size_t localSize[2] = { TS, TS / WPT };


	err = clEnqueueNDRangeKernel(queue, kernels["mat_mat_mul_gpu_sp"], 2, NULL, globalSize1, localSize, 0, NULL, NULL);
	clFinish(queue);

	buffers["buffer_A_t"] = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * n * m, NULL, NULL);

	clSetKernelArg(kernels["transpose"], 0, sizeof(cl_mem), &buffers["buffer_A"]);
	clSetKernelArg(kernels["transpose"], 1, sizeof(cl_mem), &buffers["buffer_A_t"]);
	clSetKernelArg(kernels["transpose"], 2, sizeof(int), &n);
	clSetKernelArg(kernels["transpose"], 3, sizeof(int), &m);
	size_t globalWorkSize[2] = { n, m };
	size_t localWorkSize[2] = { 16, 16 };
	err = clEnqueueNDRangeKernel(queue, kernels["transpose"], 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
	clFinish(queue);

	float max_eig = generate_max_eig(context, queue, buffers, kernels, m, n);

	float f = 0.000001f;
	size_t globalSize_scalar_mul = n * m;
	clSetKernelArg(kernels["vec_scalar_gpu_sp"], 0, sizeof(cl_mem), &buffers["buffer_A_t"]);
	clSetKernelArg(kernels["vec_scalar_gpu_sp"], 1, sizeof(float), &f);
	err = clEnqueueNDRangeKernel(queue, kernels["vec_scalar_gpu_sp"], 1, NULL, &globalSize_scalar_mul, NULL, 0, NULL, NULL);
	clFinish(queue);

	return max_eig;
}

void encrypt_data(cv::Mat& img, map<string, cl_mem>& buffers, map<string, std::vector<float>>& measurments, cl_context context, cl_command_queue queue, map<string, cl_kernel> kernels, uint32_t tile_size, int index1, int index2)
{
	cl_int err;

	int width = img.size[1];
	int height = img.size[0];
	size_t n = width * height;
	size_t m = n;
	int k = 0;
	vector<float> res(m);
	vector<float> x(n);
	vector<float> x_aux(n);

	buffers["buffer_vec_decrypt" + std::to_string(index1) + "_" + std::to_string(index2)] = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * n, NULL, NULL);
	buffers["buffer_res_decrypt" + std::to_string(index1) + "_" + std::to_string(index2)] = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * m, NULL, NULL);

	cv::Mat floatImg;
	img.convertTo(floatImg, CV_32FC1);

	k = 0;

	for (int i = 0; i < floatImg.rows; i++)
	{
		for (int j = 0; j < floatImg.cols; j++)
		{
			x[k++] = floatImg.at<float>(j, i);
		}
	}

	x_aux = x;

	float sum = 0.0f;

	buffers["buffer_b" + std::to_string(index1) + "_" + std::to_string(index2)] = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * m, NULL, NULL);
	clFinish(queue);

	size_t globalSize[1] = { m };

	clEnqueueWriteBuffer(queue, buffers["buffer_vec_decrypt" + std::to_string(index1) + "_" + std::to_string(index2)], CL_TRUE, 0, sizeof(float) * x_aux.size(), x_aux.data(), 0, NULL, NULL);

	clFinish(queue);
	clSetKernelArg(kernels["mat_vec_mul_gpu_fp32"], 0, sizeof(cl_mem), &buffers["buffer_phi"]);
	clSetKernelArg(kernels["mat_vec_mul_gpu_fp32"], 1, sizeof(cl_mem), &buffers["buffer_vec_decrypt" + std::to_string(index1) + "_" + std::to_string(index2)]);
	clSetKernelArg(kernels["mat_vec_mul_gpu_fp32"], 2, sizeof(cl_mem), &buffers["buffer_res_decrypt" + std::to_string(index1) + "_" + std::to_string(index2)]);
	clSetKernelArg(kernels["mat_vec_mul_gpu_fp32"], 3, sizeof(int), &m);
	clSetKernelArg(kernels["mat_vec_mul_gpu_fp32"], 4, sizeof(int), &n);

	err = clEnqueueNDRangeKernel(queue, kernels["mat_vec_mul_gpu_fp32"], 1, NULL, globalSize, NULL, 0, NULL, NULL);

	clEnqueueReadBuffer(queue, buffers["buffer_res_decrypt" + std::to_string(index1) + "_" + std::to_string(index2)], CL_TRUE, 0, sizeof(float) * res.size(), res.data(), 0, NULL, NULL);
	buffers["buffer_b" + std::to_string(index1) + "_" + std::to_string(index2)] = buffers["buffer_res_decrypt" + std::to_string(index1) + "_" + std::to_string(index2)];

	measurments["buffer_b" + std::to_string(index1) + "_" + std::to_string(index2)] = res;
}

void encrypt_image_alt(cv::Mat& img, map<string, cl_mem>& buffers, map<string, std::vector<float>>& measurments, uint32_t TILE_SIZE, int index) {

	cl_platform_id platform;
	cl_device_id device;
	cl_context context;
	cl_command_queue queue;
	cl_program program;
	cl_int err;

	clGetPlatformIDs(1, &platform, NULL);
	clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
	context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
	cl_queue_properties properties[] = { CL_QUEUE_PROPERTIES, (cl_command_queue_properties)(CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE | CL_QUEUE_ON_DEVICE | CL_QUEUE_ON_DEVICE_DEFAULT), 0 };

	queue = clCreateCommandQueueWithProperties(context, device, 0, &err);

	cl_command_queue device_queue = clCreateCommandQueueWithProperties(context, device, properties, &err);

	std::ifstream kernelFile("gpu_kernels.cl");

	if (!kernelFile.is_open()) {
		std::cerr << "Failed to open kernel file." << std::endl;
	}

	std::string kernelSource((std::istreambuf_iterator<char>(kernelFile)), std::istreambuf_iterator<char>());
	const char* sources = kernelSource.data();
	program = clCreateProgramWithSource(context, 1, &sources, NULL, &err);
	const char options[] = "-cl-std=CL2.0";
	clBuildProgram(program, 1, &device, options, NULL, NULL);

	map<string, cl_kernel> kernels;

	kernels["transpose"] = clCreateKernel(program, "transpose", &err);
	kernels["mat_mat_mul_gpu_sp"] = clCreateKernel(program, "mat_mat_mul_gpu_sp", &err);
	kernels["power_method"] = clCreateKernel(program, "power_method", &err);
	kernels["mat_vec_mul_gpu_fp32"] = clCreateKernel(program, "mat_vec_mul_gpu_fp32", &err);
	kernels["vec_scalar_gpu_sp"] = clCreateKernel(program, "vec_scalar_gpu_sp", &err);
	kernels["vec_sub_gpu_sp"] = clCreateKernel(program, "vec_sub_gpu_sp", &err);
	kernels["shrink_gpu_sp"] = clCreateKernel(program, "shrink_gpu_sp", &err);
	kernels["ADM"] = clCreateKernel(program, "ADM", &err);

	cv::Mat channels[3];
	cv::split(img, channels);

	for (int i = 0; i < 3; i++) {
		encrypt_data(channels[i], buffers, measurments, context, queue, kernels, TILE_SIZE, index, i);
	}

	clReleaseCommandQueue(queue);
	clReleaseCommandQueue(device_queue);
	clReleaseContext(context);
	clReleaseDevice(device);
	clReleaseProgram(program);

	for (auto i = kernels.begin(); i != kernels.end(); i++)
		clReleaseKernel(i->second);
}

void generate_decryption_matrix(map<string, cl_mem>& buffers, cl_context context, cl_command_queue queue, map<string, cl_kernel> kernels, uint32_t size, cl_device_id device, cl_program program, vector<unsigned int> seeds) {
	cl_int err;

	int width = size;
	int height = size;
	size_t n = width * height;
	size_t m = n;

	vector<float> Phi_alt(m * n);

	int chunk_size = (m * n) / seeds.size();

	#pragma omp parallel for num_threads(8) schedule(dynamic)
	for (int i = 0; i < seeds.size(); i++) {
		vec_rand(Phi_alt, 1, seeds[i], chunk_size * i, chunk_size * i + chunk_size);
	}

	buffers["buffer_phi"] = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * Phi_alt.size(), NULL, NULL);
	err = clEnqueueWriteBuffer(queue, buffers["buffer_phi"], CL_TRUE, 0, sizeof(float) * Phi_alt.size(), Phi_alt.data(), 0, NULL, NULL);

	Phi_alt = vector<float>();

	clFinish(queue);
}

std::vector<cv::Mat> splitImage(cv::Mat& image, int M, int N)
{
	int width = image.cols / M;
	int height = image.rows / N;
	int width_last_column = width + (image.cols % width);
	int height_last_row = height + (image.rows % height);

	std::vector<cv::Mat> result;

	for (int i = 0; i < N; ++i)
	{
		for (int j = 0; j < M; ++j)
		{
			cv::Rect roi(width * j,
				height * i,
				(j == (M - 1)) ? width_last_column : width,
				(i == (N - 1)) ? height_last_row : height);

			result.push_back(image(roi));
		}
	}

	return result;
}

unsigned int buffToInteger(char* buffer)
{
	return *reinterpret_cast<unsigned int*>(buffer);
}

// converts the passphrase into a series of seeds that are used to generate the encryption matrix
vector<unsigned int> passord_to_seeds(string& password) {

	char seed[4];

	vector<unsigned int> seeds(password.size() - 3);
	int q = 0;
	int password_size = password.size();
	//int password_size_adjusted = password_size - password_size % 4;

	for (int i = 0; i < password_size - 3; i++) {
		seed[0] = password[i];
		seed[1] = password[i + 1];
		seed[2] = password[i + 2];
		seed[3] = password[i + 3];

		seeds[q++] = buffToInteger(seed);
	}

	return seeds;
}

void createOpenCLcontext(openCLContext& cl_data) {
	clGetPlatformIDs(1, &cl_data.platform, NULL);
	clGetDeviceIDs(cl_data.platform, CL_DEVICE_TYPE_GPU, 1, &cl_data.device, NULL);
	cl_data.context = clCreateContext(NULL, 1, &cl_data.device, NULL, NULL, &cl_data.err);
	cl_queue_properties properties[] = { CL_QUEUE_PROPERTIES, (cl_command_queue_properties)(CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE | CL_QUEUE_ON_DEVICE | CL_QUEUE_ON_DEVICE_DEFAULT), 0 };

	cl_data.queue = clCreateCommandQueueWithProperties(cl_data.context, cl_data.device, 0, &cl_data.err);

	cl_data.device_queue = clCreateCommandQueueWithProperties(cl_data.context, cl_data.device, properties, &cl_data.err);

	std::ifstream kernelFile("gpu_kernels.cl");

	if (!kernelFile.is_open()) {
		std::cerr << "Failed to open kernel file." << std::endl;
	}

	std::string kernelSource((std::istreambuf_iterator<char>(kernelFile)), std::istreambuf_iterator<char>());
	const char* sources = kernelSource.data();
	cl_data.program = clCreateProgramWithSource(cl_data.context, 1, &sources, NULL, &cl_data.err);
	const char options[] = "-cl-std=CL2.0";
	clBuildProgram(cl_data.program, 1, &cl_data.device, options, NULL, NULL);
}

encryptionImage encryptImage(cv::Mat img, /*image to be encrypted*/
							int TILE_SIZE, /*size of tiles in which the image is broken up and processed, larger tiles may provide better quality at the cost of memory and speed*/
							string passphrase /*passphare used to generate the encryption matrix*/) {

	try {
		bool isImgEmpty = img.empty();
		if (isImgEmpty) {
			throw isImgEmpty;
		}
	}
	catch (...) {
		std::cout << "Image empty";
	}

	int original_width = img.size[1];
	int original_height = img.size[0];

	int processed_width = 0;
	int processed_height = 0;

	/* 
	for simplcity we resize the image such that it's perfectly divisible by the size of tiles
	later when the image is decrypted it will be resized to it's original size
	*/
	if (original_width % TILE_SIZE != 0 || original_height % TILE_SIZE != 0) {

		processed_width = original_width - original_width % TILE_SIZE + TILE_SIZE;
		processed_height = original_height - original_height % TILE_SIZE + TILE_SIZE;

		cv::Mat resized_img;
		cv::resize(img, resized_img, cv::Size(processed_width, processed_height));
		img = resized_img;
	}
	else {
		processed_width = original_width;
		processed_height = original_height;
	}

	int N = processed_width / TILE_SIZE;
	int M = processed_height / TILE_SIZE;

	openCLContext cl_data{NULL, NULL, NULL, NULL, NULL, NULL, 0};
	createOpenCLcontext(cl_data);

	int k = 0;

	map<string, cl_kernel> kernels;
	map<string, cl_mem> buffers;
	map<string, std::vector<float>> measurments;

	kernels["transpose"] = clCreateKernel(cl_data.program, "transpose", &cl_data.err);
	kernels["mat_mat_mul_gpu_sp"] = clCreateKernel(cl_data.program, "mat_mat_mul_gpu_sp", &cl_data.err);
	kernels["power_method"] = clCreateKernel(cl_data.program, "power_method", &cl_data.err);
	kernels["mat_vec_mul_gpu_fp32"] = clCreateKernel(cl_data.program, "mat_vec_mul_gpu_fp32", &cl_data.err);
	kernels["vec_scalar_gpu_sp"] = clCreateKernel(cl_data.program, "vec_scalar_gpu_sp", &cl_data.err);
	kernels["vec_sub_gpu_sp"] = clCreateKernel(cl_data.program, "vec_sub_gpu_sp", &cl_data.err);
	kernels["shrink_gpu_sp"] = clCreateKernel(cl_data.program, "shrink_gpu_sp", &cl_data.err);
	kernels["ADM"] = clCreateKernel(cl_data.program, "ADM", &cl_data.err);

	cv::Mat channels[3];
	cv::Mat out_channels[3];
	cv::split(img, channels);
	cv::Mat out = img;
	cv::split(img, out_channels);
	cv::Mat reconstructed_img;
	cv::Mat original_img;

	std::vector<cv::Mat> array_of_images = splitImage(img, N, M);
	std::vector<cv::Mat> array_of_images_out = splitImage(img, N, M);

	char seed[4];

	std::vector<unsigned int> seeds = passord_to_seeds(passphrase);

	generate_decryption_matrix(buffers, cl_data.context, cl_data.queue, kernels, TILE_SIZE, cl_data.device, cl_data.program, seeds);

	#pragma omp parallel for num_threads(4) schedule(dynamic)
	for (int i = 0; i < array_of_images.size(); i++) {
		encrypt_image_alt(array_of_images[i], buffers, measurments, TILE_SIZE, i);
	}

	std::vector<float> data_array;

	for (int i = 0; i < array_of_images.size(); i++) {
		for (int j = 0; j < 3; j++) {
			data_array.insert(data_array.end(), measurments["buffer_b" + std::to_string(i) + "_" + std::to_string(j)].begin(), measurments["buffer_b" + std::to_string(i) + "_" + std::to_string(j)].end());
		}
	}

	encryptionImage img_encrypted{ TILE_SIZE, original_width, original_height, processed_width, processed_height , data_array };

	return img_encrypted;
}

cv::Mat decryptImage(encryptionImage img, string passphrase) {

	cv::Mat outputImg(cv::Size(img.original_width, img.original_height), CV_8UC3);

	int original_width = img.original_width;
	int original_height = img.original_height;

	int processed_width = 0;
	int processed_height = 0;

	if (original_width % img.TILE_SIZE != 0 || original_height % img.TILE_SIZE != 0) {

		processed_width = original_width - original_width % img.TILE_SIZE + img.TILE_SIZE;
		processed_height = original_height - original_height % img.TILE_SIZE + img.TILE_SIZE;

		cv::Mat resized_img;
		cv::resize(outputImg, resized_img, cv::Size(processed_width, processed_height));
		outputImg = resized_img;
	}
	else {
		processed_width = original_width;
		processed_height = original_height;
	}

	int N = processed_width / img.TILE_SIZE;
	int M = processed_height / img.TILE_SIZE;

	openCLContext cl_data{ NULL, NULL, NULL, NULL, NULL, NULL, 0 };
	createOpenCLcontext(cl_data);

	int k = 0;

	map<string, cl_kernel> kernels;
	map<string, cl_mem> buffers;
	map<string, std::vector<float>> measurments;

	kernels["transpose"] = clCreateKernel(cl_data.program, "transpose", &cl_data.err);
	kernels["mat_mat_mul_gpu_sp"] = clCreateKernel(cl_data.program, "mat_mat_mul_gpu_sp", &cl_data.err);
	kernels["power_method"] = clCreateKernel(cl_data.program, "power_method", &cl_data.err);
	kernels["mat_vec_mul_gpu_fp32"] = clCreateKernel(cl_data.program, "mat_vec_mul_gpu_fp32", &cl_data.err);
	kernels["vec_scalar_gpu_sp"] = clCreateKernel(cl_data.program, "vec_scalar_gpu_sp", &cl_data.err);
	kernels["vec_sub_gpu_sp"] = clCreateKernel(cl_data.program, "vec_sub_gpu_sp", &cl_data.err);
	kernels["shrink_gpu_sp"] = clCreateKernel(cl_data.program, "shrink_gpu_sp", &cl_data.err);
	kernels["ADM"] = clCreateKernel(cl_data.program, "ADM", &cl_data.err);


	char seed[4];

	std::vector<unsigned int> seeds = passord_to_seeds(passphrase);

	generate_decryption_matrix(buffers, cl_data.context, cl_data.queue, kernels, img.TILE_SIZE, cl_data.device, cl_data.program, seeds);
	float max_eig = generate_dictionary(buffers, cl_data.context, cl_data.queue, kernels, img.TILE_SIZE, cl_data.device, cl_data.program);

	int index_step = 0;

	for (int i = 0; i < N * M; i++) {
		for (int j = 0; j < 3; j++) {
			int firstIndex = index_step;
			int lastIndex = index_step + (img.TILE_SIZE * img.TILE_SIZE);
			index_step = index_step + (img.TILE_SIZE * img.TILE_SIZE);
			vector<float>::const_iterator first = img.data_array.begin() + firstIndex;
			vector<float>::const_iterator last = img.data_array.begin() + lastIndex;
			vector<float> new_vec(first, last);
			buffers["buffer_b" + std::to_string(i) + "_" + std::to_string(j)] = clCreateBuffer(cl_data.context, CL_MEM_READ_WRITE, sizeof(float) * img.TILE_SIZE * img.TILE_SIZE, NULL, NULL);
			clEnqueueWriteBuffer(cl_data.queue, buffers["buffer_b" + std::to_string(i) + "_" + std::to_string(j)], CL_TRUE, 0, sizeof(float) * new_vec.size(), new_vec.data(), 0, NULL, NULL);

			vector<float> aux(img.TILE_SIZE * img.TILE_SIZE);
			clEnqueueReadBuffer(cl_data.queue, buffers["buffer_b" + std::to_string(i) + "_" + std::to_string(j)], CL_TRUE, 0, sizeof(float) * aux.size(), aux.data(), 0, NULL, NULL);
			clFinish(cl_data.queue);
		}
	}

	std::vector<cv::Mat> array_of_images_out = splitImage(outputImg, N, M);


	#pragma omp parallel for num_threads(4) schedule(dynamic)
	for (int i = 0; i < array_of_images_out.size(); i++) {
		array_of_images_out[i].setTo(0);
		decrypt_image_alt(array_of_images_out[i], buffers, img.TILE_SIZE, max_eig, i);
	}

	bool OK1 = false;
	bool OK2 = false;
	cv::Mat final_image2;

	for (int i = 0; i < M; ++i) {

		cv::Mat final_image1 = array_of_images_out[i * N];
		bool OK1 = false;

		for (int j = 0; j < N; ++j) {
			if (OK1) {
				cv::hconcat(final_image1, array_of_images_out[i * N + j], final_image1);
			}
			OK1 = true;
		}

		if (OK2) {
			cv::vconcat(final_image2, final_image1, final_image2);
		}

		if (!OK2) {
			final_image2 = final_image1;
			OK2 = true;
		}
	}

	cv::resize(final_image2, final_image2, cv::Size(original_width, original_height));

	return final_image2;
}