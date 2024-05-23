#include "sparseImageEncryption.h"
#include "AVX_functions.h"

std::vector <float> ADM_gpu(map<string, cl_mem>& buffers,
	int A_cols, int A_rows, float max_eig, float beta, float tau, int iterations,
	cl_context context, cl_command_queue queue, map<string, cl_kernel> kernels, cl_device_id device, cl_program program, int index1, int index2)
{
	cl_queue_properties properties[] = { CL_QUEUE_PROPERTIES, (cl_command_queue_properties)(CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE | CL_QUEUE_ON_DEVICE | CL_QUEUE_ON_DEVICE_DEFAULT), 0 };

	int n = A_cols;
	int m = A_rows;
	//Matrix<float> A_t;
	//A_t = A;
	//A_t.transposeMatrix();

	int err = 0;
	size_t globalSize_m[2] = { m, 1 };
	size_t globalSize_n[2] = { n, 1 };

	float gamma = 1.99f - (tau * max_eig);
	float f = 0.0f;

	//std::vector<float> aux1(n), aux2(n), aux3(n), aux_b(m), aux_x(n), b1(m), x1(n);
	std::vector<float> x1(n);
	//std::vector<float> y(m);
	//std::vector<float> r(m);
	//std::cout << A[0] << ' ' << A[n * m - 2] << std::endl;
	//std::cout << A_t[0] << ' ' << A_t[n * m - 2] << std::endl;
	//std::cout << b[0] << ' ' << b[m - 1] << std::endl;
	//clFinish(queue);
	//A_t = A_t * tau;
	//b1 = b.data;

	//opencl stuff

	//f = tau;
	//size_t globalSize = n * m;
	//clSetKernelArg(vec_scalar_gpu_sp, 0, sizeof(cl_mem), &buffer_A_t);
	//clSetKernelArg(vec_scalar_gpu_sp, 1, sizeof(float), &f);
	//err = clEnqueueNDRangeKernel(queue, vec_scalar_gpu_sp, 1, NULL, &globalSize, NULL, 0, NULL, NULL);
	//clFinish(queue);
	//queue = clCreateCommandQueueWithProperties(context, device, 0, &err);
	//std::vector<float> A_flat(n * m);
	//std::vector<float> A_t_flat(n * m);
	//std::vector<float> aux1_flat(n), res_flat(m);

	//A_flat = A.data;
	//A_t_flat = A_t.data;

	//float a1 = A.data[0];
	//float a2 = A.data[A.data.size() - 1];
	//float a3 = A_t.data[0];
	//float a4 = A_t.data[A_t.data.size() - 1];

	//A.data = std::vector<float>();
	//A_t.data = std::vector<float>();

	//cl_mem buffer_A = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * A_flat.size(), NULL, NULL);
	//cl_mem buffer_A_t = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * A_t_flat.size(), NULL, NULL);
	buffers["buffer_res" + std::to_string(index1) + "_" + std::to_string(index2)] = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * n, NULL, NULL);
	buffers["buffer_y" + std::to_string(index1) + "_" + std::to_string(index2)] = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * m, NULL, NULL);
	buffers["buffer_x" + std::to_string(index1) + "_" + std::to_string(index2)] = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * n, NULL, NULL);
	buffers["buffer_res_aux" + std::to_string(index1) + "_" + std::to_string(index2)] = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * n, NULL, NULL);
	buffers["buffer_count" + std::to_string(index1) + "_" + std::to_string(index2)] = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int), NULL, NULL);

	//cl_mem buffer_res = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * n, NULL, NULL);
	//cl_mem buffer_y = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * m, NULL, NULL);
	//cl_mem buffer_x = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * n, NULL, NULL);
	//cl_mem buffer_res2 = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * n, NULL, NULL);
	//cl_mem buffer_count = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int), NULL, NULL);
	//cl_mem buffer_b = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * m, NULL, NULL);

	//clEnqueueWriteBuffer(queue, buffer_A, CL_TRUE, 0, sizeof(float) * A_flat.size(), A_flat.data(), 0, NULL, NULL);
	//clEnqueueWriteBuffer(queue, buffer_A_t, CL_TRUE, 0, sizeof(float) * A_t_flat.size(), A_t_flat.data(), 0, NULL, NULL);
	//clEnqueueWriteBuffer(queue, buffer_b, CL_TRUE, 0, sizeof(float) * b1.size(), b1.data(), 0, NULL, NULL);

	float fill = 0.0f;
	//clEnqueueFillBuffer(queue, buffer_res, &fill, sizeof(float), 0, n, 0, NULL, NULL);
	//clEnqueueFillBuffer(queue, buffer_y, &fill, sizeof(float), 0, m, 0, NULL, NULL);
	//clEnqueueFillBuffer(queue, buffer_x, &fill, sizeof(float), 0, n, 0, NULL, NULL);
	//clEnqueueFillBuffer(queue, buffer_res2, &fill, sizeof(float), 0, n, 0, NULL, NULL);
	clFinish(queue);

	//A_flat = std::vector<float>();
	//A_t_flat = std::vector<float>();

	clFinish(queue);

	//clSetKernelArg(kernel_mat_vec_mul_gpu, 0, sizeof(cl_mem), &buffer_A);
	//clSetKernelArg(kernel_mat_vec_mul_gpu, 1, sizeof(cl_mem), &buffer_vec);
	//clSetKernelArg(kernel_mat_vec_mul_gpu, 2, sizeof(cl_mem), &buffer_res);
	//clSetKernelArg(kernel_mat_vec_mul_gpu, 3, sizeof(int), &m);

	//clSetKernelArg(vec_scalar_gpu_sp, 1, sizeof(float), &f);

	size_t globalSize_g[1] = { 1 };
	cl_event event1;

	//high_resolution_clock::time_point t1 = high_resolution_clock::now();

	for (int i = 0; i < iterations; i++)
	{
		//mat_vec_mul_gpu_fp32(buffer_A, buffer_x, buffer_res, m, n);
		//vec_sub_gpu_sp(buffer_res, buffer_b); 
		//vec_scalar_gpu_sp(buffer_y, (1 / beta));
		//vec_sub_gpu_sp(buffer_res, buffer_y); 
		//mat_vec_mul_gpu_fp32(buffer_A_t, buffer_res, buffer_res2, n, m); 
		//vec_sub_gpu_sp(buffer_x, buffer_res2); });
		//shrink_gpu_sp(buffer_x, (tau / beta)); });
		//mat_vec_mul_gpu_fp32(buffer_A, buffer_x, buffer_res2, m, n); 
		//vec_sub_gpu_sp(buffer_res2, buffer_b); });
		//vec_scalar_gpu_sp(buffer_res2, (gamma * beta)); 
		//vec_scalar_gpu_sp(buffer_y, (1 / (1 / beta))); 
		//vec_sub_gpu_sp(buffer_y, buffer_res2); 

		//mat_vec_mul_gpu_fp32(buffer_A, buffer_x, buffer_res, m, n);
		clSetKernelArg(kernels["mat_vec_mul_gpu_fp32"], 0, sizeof(cl_mem), &buffers["buffer_A"]);
		clSetKernelArg(kernels["mat_vec_mul_gpu_fp32"], 1, sizeof(cl_mem), &buffers["buffer_x" + std::to_string(index1) + "_" + std::to_string(index2)]);
		clSetKernelArg(kernels["mat_vec_mul_gpu_fp32"], 2, sizeof(cl_mem), &buffers["buffer_res" + std::to_string(index1) + "_" + std::to_string(index2)]);
		clSetKernelArg(kernels["mat_vec_mul_gpu_fp32"], 3, sizeof(int), &m);
		clSetKernelArg(kernels["mat_vec_mul_gpu_fp32"], 4, sizeof(int), &n);
		err = clEnqueueNDRangeKernel(queue, kernels["mat_vec_mul_gpu_fp32"], 1, NULL, globalSize_m, NULL, 0, NULL, NULL);
		//cl_mem new_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * n, NULL, NULL);
		//new_buffer = buffers["buffer_x" + std::to_string(index1) + "_" + std::to_string(index2)];
		//std::vector<float> auxt(n);
		//clEnqueueReadBuffer(queue, new_buffer, CL_TRUE, 0, sizeof(float) * auxt.size(), auxt.data(), 0, NULL, NULL);
		clFinish(queue);

		//vec_sub_gpu_sp(buffer_res, buffer_b); 
		clSetKernelArg(kernels["vec_sub_gpu_sp"], 0, sizeof(cl_mem), &buffers["buffer_res" + std::to_string(index1) + "_" + std::to_string(index2)]);
		clSetKernelArg(kernels["vec_sub_gpu_sp"], 1, sizeof(cl_mem), &buffers["buffer_b" + std::to_string(index1) + "_" + std::to_string(index2)]);
		err = clEnqueueNDRangeKernel(queue, kernels["vec_sub_gpu_sp"], 1, NULL, globalSize_m, NULL, 0, NULL, NULL);
		//std::vector<float> auxb(m);
		//clEnqueueReadBuffer(queue, buffer_res_x, CL_TRUE, 0, sizeof(float) * auxb.size(), auxb.data(), 0, NULL, NULL);
		clFinish(queue);

		//vec_scalar_gpu_sp(buffer_y, (1 / beta));
		f = (1 / beta);
		clSetKernelArg(kernels["vec_scalar_gpu_sp"], 0, sizeof(cl_mem), &buffers["buffer_y" + std::to_string(index1) + "_" + std::to_string(index2)]);
		clSetKernelArg(kernels["vec_scalar_gpu_sp"], 1, sizeof(float), &f);
		err = clEnqueueNDRangeKernel(queue, kernels["vec_scalar_gpu_sp"], 1, NULL, globalSize_m, NULL, 0, NULL, NULL);
		//clEnqueueReadBuffer(queue, buffer_aux_y, CL_TRUE, 0, sizeof(float) * auxb.size(), auxb.data(), 0, NULL, NULL);
		clFinish(queue);

		//vec_sub_gpu_sp(buffer_res, buffer_y); 
		clSetKernelArg(kernels["vec_sub_gpu_sp"], 0, sizeof(cl_mem), &buffers["buffer_res" + std::to_string(index1) + "_" + std::to_string(index2)]);
		clSetKernelArg(kernels["vec_sub_gpu_sp"], 1, sizeof(cl_mem), &buffers["buffer_y" + std::to_string(index1) + "_" + std::to_string(index2)]);
		err = clEnqueueNDRangeKernel(queue, kernels["vec_sub_gpu_sp"], 1, NULL, globalSize_m, NULL, 0, NULL, NULL);
		//std::vector<float> auxb(m);
		//clEnqueueReadBuffer(queue, buffer_res_x, CL_TRUE, 0, sizeof(float) * auxb.size(), auxb.data(), 0, NULL, NULL);
		clFinish(queue);

		//mat_vec_mul_gpu_fp32(buffer_A_t, buffer_res, buffer_res2, n, m); 
		clSetKernelArg(kernels["mat_vec_mul_gpu_fp32"], 0, sizeof(cl_mem), &buffers["buffer_A_t"]);
		clSetKernelArg(kernels["mat_vec_mul_gpu_fp32"], 1, sizeof(cl_mem), &buffers["buffer_res" + std::to_string(index1) + "_" + std::to_string(index2)]);
		clSetKernelArg(kernels["mat_vec_mul_gpu_fp32"], 2, sizeof(cl_mem), &buffers["buffer_res_aux" + std::to_string(index1) + "_" + std::to_string(index2)]);
		clSetKernelArg(kernels["mat_vec_mul_gpu_fp32"], 3, sizeof(int), &n);
		clSetKernelArg(kernels["mat_vec_mul_gpu_fp32"], 4, sizeof(int), &m);
		err = clEnqueueNDRangeKernel(queue, kernels["mat_vec_mul_gpu_fp32"], 1, NULL, globalSize_n, NULL, 0, NULL, NULL);
		//std::vector<float> auxt(n);
		//clEnqueueReadBuffer(queue, buffer_res_x, CL_TRUE, 0, sizeof(float) * auxt.size(), auxt.data(), 0, NULL, NULL);
		clFinish(queue);

		//vec_sub_gpu_sp(buffer_x, buffer_res2); });
		clSetKernelArg(kernels["vec_sub_gpu_sp"], 0, sizeof(cl_mem), &buffers["buffer_x" + std::to_string(index1) + "_" + std::to_string(index2)]);
		clSetKernelArg(kernels["vec_sub_gpu_sp"], 1, sizeof(cl_mem), &buffers["buffer_res_aux" + std::to_string(index1) + "_" + std::to_string(index2)]);
		err = clEnqueueNDRangeKernel(queue, kernels["vec_sub_gpu_sp"], 1, NULL, globalSize_n, NULL, 0, NULL, NULL);
		//std::vector<float> auxb(m);
		//clEnqueueReadBuffer(queue, buffer_res_x, CL_TRUE, 0, sizeof(float) * auxb.size(), auxb.data(), 0, NULL, NULL);
		clFinish(queue);

		//shrink_gpu_sp(buffer_x, (tau / beta)); });
		f = (tau / beta);
		clSetKernelArg(kernels["shrink_gpu_sp"], 0, sizeof(cl_mem), &buffers["buffer_x" + std::to_string(index1) + "_" + std::to_string(index2)]);
		clSetKernelArg(kernels["shrink_gpu_sp"], 1, sizeof(float), &f);
		err = clEnqueueNDRangeKernel(queue, kernels["shrink_gpu_sp"], 1, NULL, globalSize_n, NULL, 0, NULL, NULL);
		//clEnqueueReadBuffer(queue, buffer_x, CL_TRUE, 0, sizeof(float) * auxt.size(), auxt.data(), 0, NULL, NULL);
		//clFinish(queue);

		//mat_vec_mul_gpu_fp32(buffer_A, buffer_x, buffer_res2, m, n); 
		clSetKernelArg(kernels["mat_vec_mul_gpu_fp32"], 0, sizeof(cl_mem), &buffers["buffer_A"]);
		clSetKernelArg(kernels["mat_vec_mul_gpu_fp32"], 1, sizeof(cl_mem), &buffers["buffer_x" + std::to_string(index1) + "_" + std::to_string(index2)]);
		clSetKernelArg(kernels["mat_vec_mul_gpu_fp32"], 2, sizeof(cl_mem), &buffers["buffer_res_aux" + std::to_string(index1) + "_" + std::to_string(index2)]);
		clSetKernelArg(kernels["mat_vec_mul_gpu_fp32"], 3, sizeof(int), &m);
		clSetKernelArg(kernels["mat_vec_mul_gpu_fp32"], 4, sizeof(int), &n);
		err = clEnqueueNDRangeKernel(queue, kernels["mat_vec_mul_gpu_fp32"], 1, NULL, globalSize_m, NULL, 0, NULL, NULL);
		//std::vector<float> auxt(n);
		//clEnqueueReadBuffer(queue, buffer_res_x, CL_TRUE, 0, sizeof(float) * auxt.size(), auxt.data(), 0, NULL, NULL);
		clFinish(queue);

		//vec_sub_gpu_sp(buffer_res2, buffer_b); });
		clSetKernelArg(kernels["vec_sub_gpu_sp"], 0, sizeof(cl_mem), &buffers["buffer_res_aux" + std::to_string(index1) + "_" + std::to_string(index2)]);
		clSetKernelArg(kernels["vec_sub_gpu_sp"], 1, sizeof(cl_mem), &buffers["buffer_b" + std::to_string(index1) + "_" + std::to_string(index2)]);
		err = clEnqueueNDRangeKernel(queue, kernels["vec_sub_gpu_sp"], 1, NULL, globalSize_m, NULL, 0, NULL, NULL);
		//std::vector<float> auxb(m);
		//clEnqueueReadBuffer(queue, buffer_res_x, CL_TRUE, 0, sizeof(float) * auxb.size(), auxb.data(), 0, NULL, NULL);
		clFinish(queue);

		//vec_scalar_gpu_sp(buffer_res2, (gamma * beta)); 
		f = (gamma * beta);
		clSetKernelArg(kernels["vec_scalar_gpu_sp"], 0, sizeof(cl_mem), &buffers["buffer_res_aux" + std::to_string(index1) + "_" + std::to_string(index2)]);
		clSetKernelArg(kernels["vec_scalar_gpu_sp"], 1, sizeof(float), &f);
		err = clEnqueueNDRangeKernel(queue, kernels["vec_scalar_gpu_sp"], 1, NULL, globalSize_m, NULL, 0, NULL, NULL);
		//clEnqueueReadBuffer(queue, buffer_aux_y, CL_TRUE, 0, sizeof(float) * auxb.size(), auxb.data(), 0, NULL, NULL);
		clFinish(queue);

		//vec_scalar_gpu_sp(buffer_y, (1 / (1 / beta))); 
		f = (1 / (1 / beta));
		clSetKernelArg(kernels["vec_scalar_gpu_sp"], 0, sizeof(cl_mem), &buffers["buffer_y" + std::to_string(index1) + "_" + std::to_string(index2)]);
		clSetKernelArg(kernels["vec_scalar_gpu_sp"], 1, sizeof(float), &f);
		err = clEnqueueNDRangeKernel(queue, kernels["vec_scalar_gpu_sp"], 1, NULL, globalSize_m, NULL, 0, NULL, NULL);
		//clEnqueueReadBuffer(queue, buffer_aux_y, CL_TRUE, 0, sizeof(float) * auxb.size(), auxb.data(), 0, NULL, NULL);
		clFinish(queue);

		//vec_sub_gpu_sp(buffer_y, buffer_res2); 
		clSetKernelArg(kernels["vec_sub_gpu_sp"], 0, sizeof(cl_mem), &buffers["buffer_y" + std::to_string(index1) + "_" + std::to_string(index2)]);
		clSetKernelArg(kernels["vec_sub_gpu_sp"], 1, sizeof(cl_mem), &buffers["buffer_res_aux" + std::to_string(index1) + "_" + std::to_string(index2)]);
		err = clEnqueueNDRangeKernel(queue, kernels["vec_sub_gpu_sp"], 1, NULL, globalSize_m, NULL, 0, NULL, NULL);
		//std::vector<float> auxb(m);
		//clEnqueueReadBuffer(queue, buffer_res_x, CL_TRUE, 0, sizeof(float) * auxb.size(), auxb.data(), 0, NULL, NULL);
		clFinish(queue);
	}

	clEnqueueReadBuffer(queue, buffers["buffer_x" + std::to_string(index1) + "_" + std::to_string(index2)], CL_TRUE, 0, sizeof(float) * x1.size(), x1.data(), 0, NULL, NULL);
	//high_resolution_clock::time_point t2 = high_resolution_clock::now();
	//float solve_time = duration_cast<milliseconds>(t2 - t1).count();
	//std::cout << solve_time << endl;

	//t1 = high_resolution_clock::now();
	//queue = clCreateCommandQueueWithProperties(context, device, 0, &err);
	//t2 = high_resolution_clock::now();
	//solve_time = duration_cast<milliseconds>(t2 - t1).count();
	//std::cout << solve_time << endl;
	//clReleaseMemObject(buffer_b);
	//clReleaseMemObject(buffer_res);
	//clReleaseMemObject(buffer_y);
	//clReleaseMemObject(buffer_x);
	//clReleaseMemObject(buffer_res2);
	//clReleaseMemObject(buffer_count);

	//clReleaseKernel(mat_vec_mul_gpu_sp);
	//clReleaseKernel(vec_sub_gpu_sp);
	//clReleaseKernel(vec_scalar_gpu_sp);
	//clReleaseKernel(shrink_gpu_sp);

	//std::cout << x1[0] << ' ' << x1[n - 1] << std::endl;
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

	//buffers["buffer_A_t"] = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * n * m, NULL, NULL);

	//cv::Mat floatImg;
	cv::Mat reconstructedImg;
	//cv::Mat originalImg;
	//img.convertTo(floatImg, CV_32FC1);
	out.convertTo(reconstructedImg, CV_32FC1);
	//img.convertTo(originalImg, CV_32FC1);
	//img.setTo(0);

	clFinish(queue);

	//clEnqueueWriteBuffer(queue, buffers["buffer_b"], CL_TRUE, 0, sizeof(float) * encrypted_data.size(), encrypted_data.data(), 0, NULL, NULL);


	//clSetKernelArg(kernels["transpose"], 0, sizeof(cl_mem), &buffers["buffer_A"]);
	//clSetKernelArg(kernels["transpose"], 1, sizeof(cl_mem), &buffers["buffer_A_t"]);
	//clSetKernelArg(kernels["transpose"], 2, sizeof(int), &n);
	//clSetKernelArg(kernels["transpose"], 3, sizeof(int), &m);
	//size_t globalWorkSize[2] = { n, m };
	//size_t localWorkSize[2] = { 16, 16 }; // You can adjust this as needed
	//err = clEnqueueNDRangeKernel(queue, kernels["transpose"], 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
	//clFinish(queue);
	//std::vector <float> result(n * m);
	//clEnqueueReadBuffer(queue, buffers["buffer_A_t"], CL_TRUE, 0, sizeof(float) * result.size(), result.data(), 0, NULL, NULL);


	clFinish(queue);

	//static float max_eig = generate_max_eig(context, queue, buffers, kernels, m, n);

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

	//cv::Mat dst;
	reconstructedImg.convertTo(reconstructedImg, CV_8U);
	out = reconstructedImg;
}

void decrypt_image(cv::Mat& out, map<string, cl_mem>& buffers, cl_context context, cl_command_queue queue, map<string, cl_kernel> kernels, uint32_t TILE_SIZE, cl_device_id device, cl_program program, float max_eig, int index1) {

	cv::Mat channels[3];
	cv::split(out, channels);

	for (int i = 0; i < 3; i++) {
		//encrypt_data(channels[i], buffers, context, queue, kernels, TILE_SIZE, i);
		decrypt_data(channels[i], buffers, context, queue, kernels, TILE_SIZE, device, program, max_eig, index1, i);
		//recostruct(channels[i], out_channels[i], buffers, max_eig, context, queue, kernels, device, program);
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
	//cl_queue_properties properties[] = { CL_QUEUE_PROPERTIES, (cl_command_queue_properties)(CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE | CL_QUEUE_ON_DEVICE | CL_QUEUE_ON_DEVICE_DEFAULT), 0 };

	queue = clCreateCommandQueueWithProperties(context, device, 0, &err);

	//cl_command_queue device_queue = clCreateCommandQueueWithProperties(context, device, properties, &err);

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
	//cl_kernel kernel_mat_transpose = kernels["transpose"];
	//cl_kernel kernel_mat_mat_mul_gpu = kernels["mat_mat_mul_gpu_sp"];
	//cl_kernel power_method = kernels["power_method"];
	//cl_kernel kernel_mat_vec_mul_gpu = kernels["mat_vec_mul_gpu_fp32"];
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
		//encrypt_data(channels[i], buffers, context, queue, kernels, TILE_SIZE, i);
		decrypt_data(channels[i], buffers, context, queue, kernels, TILE_SIZE, device, program, max_eig, index, i);
		//recostruct(channels[i], out_channels[i], buffers, max_eig, context, queue, kernels, device, program);
	}

	cv::merge(channels, 3, out);

	clReleaseCommandQueue(queue);
	//clReleaseCommandQueue(device_queue);
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
			//IDCT[(i * n) + k] = psi[k];
		}
	}
}

float generate_max_eig(cl_context context, cl_command_queue queue,
	map<string, cl_mem> buffers, map<string, cl_kernel> kernels, int A_rows, int A_cols)
{
	std::vector<float> max_eig(1);

	int err = 0;

	int m = A_rows;  // Number of rows in matrix A
	int n = A_cols;  // Number of columns in matrix A and rows in matrix B
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
	size_t localWorkSize[2] = { 16, 16 }; // You can adjust this as needed
	err = clEnqueueNDRangeKernel(queue, kernels["transpose"], 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
	clFinish(queue);

	float max_eig = generate_max_eig(context, queue, buffers, kernels, m, n);

	float f = 0.000001f;
	size_t globalSize_scalar_mul = n * m;
	clSetKernelArg(kernels["vec_scalar_gpu_sp"], 0, sizeof(cl_mem), &buffers["buffer_A_t"]);
	clSetKernelArg(kernels["vec_scalar_gpu_sp"], 1, sizeof(float), &f);
	err = clEnqueueNDRangeKernel(queue, kernels["vec_scalar_gpu_sp"], 1, NULL, &globalSize_scalar_mul, NULL, 0, NULL, NULL);

	//clReleaseMemObject(buffers["buffer_IDCT"]);
	//clReleaseMemObject(buffers["buffer_phi"]);

	//buffers.erase("buffer_phi");
	//buffers.erase("buffer_IDCT");

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
	//vector<float> y(m);

	buffers["buffer_vec_decrypt" + std::to_string(index1) + "_" + std::to_string(index2)] = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * n, NULL, NULL);
	buffers["buffer_res_decrypt" + std::to_string(index1) + "_" + std::to_string(index2)] = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * m, NULL, NULL);

	cv::Mat floatImg;
	img.convertTo(floatImg, CV_32FC1);
	//img.setTo(0);

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

	//Matrix<float> y_alt(m);
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

	//clReleaseMemObject(buffers["buffer_vec_decrypt" + std::to_string(index1) + "_" + std::to_string(index2)]);
	//clReleaseMemObject(buffers["buffer_res_decrypt" + std::to_string(index1) + "_" + std::to_string(index2)]);

	//buffers.erase("buffer_vec_decrypt" + std::to_string(index1) + "_" + std::to_string(index2));
	//buffers.erase("buffer_res_decrypt" + std::to_string(index1) + "_" + std::to_string(index2));
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
	//cl_kernel kernel_mat_transpose = kernels["transpose"];
	//cl_kernel kernel_mat_mat_mul_gpu = kernels["mat_mat_mul_gpu_sp"];
	//cl_kernel power_method = kernels["power_method"];
	//cl_kernel kernel_mat_vec_mul_gpu = kernels["mat_vec_mul_gpu_fp32"];
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
		//decrypt_data(out_channels[i], buffers, context, queue, kernels, TILE_SIZE, device, program, max_eig, i, 1);
		//recostruct(channels[i], out_channels[i], buffers, max_eig, context, queue, kernels, device, program);
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


	//vec_rand(Phi_alt, 1, seed[0]);
	buffers["buffer_phi"] = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * Phi_alt.size(), NULL, NULL);
	err = clEnqueueWriteBuffer(queue, buffers["buffer_phi"], CL_TRUE, 0, sizeof(float) * Phi_alt.size(), Phi_alt.data(), 0, NULL, NULL);

	Phi_alt = vector<float>();

	clFinish(queue);
}

std::vector<cv::Mat> splitImage(cv::Mat& image, int M, int N)
{
	// All images should be the same size ...
	int width = image.cols / M;
	int height = image.rows / N;
	// ... except for the Mth column and the Nth row
	int width_last_column = width + (image.cols % width);
	int height_last_row = height + (image.rows % height);

	std::vector<cv::Mat> result;

	for (int i = 0; i < N; ++i)
	{
		for (int j = 0; j < M; ++j)
		{
			// Compute the region to crop from
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

encryptionImage encryptImage(cv::Mat img, int TILE_SIZE, string passphrase) {

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

	//display_opencl_info();

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

	int k = 0;

	//display_img(img);
	//cl_mem buffer_A, buffer_A_t;
	map<string, cl_kernel> kernels;
	map<string, cl_mem> buffers;
	map<string, std::vector<float>> measurments;
	//cl_kernel kernel_mat_transpose = kernels["transpose"];
	//cl_kernel kernel_mat_mat_mul_gpu = kernels["mat_mat_mul_gpu_sp"];
	//cl_kernel power_method = kernels["power_method"];
	//cl_kernel kernel_mat_vec_mul_gpu = kernels["mat_vec_mul_gpu_fp32"];
	kernels["transpose"] = clCreateKernel(program, "transpose", &err);
	kernels["mat_mat_mul_gpu_sp"] = clCreateKernel(program, "mat_mat_mul_gpu_sp", &err);
	kernels["power_method"] = clCreateKernel(program, "power_method", &err);
	kernels["mat_vec_mul_gpu_fp32"] = clCreateKernel(program, "mat_vec_mul_gpu_fp32", &err);
	kernels["vec_scalar_gpu_sp"] = clCreateKernel(program, "vec_scalar_gpu_sp", &err);
	kernels["vec_sub_gpu_sp"] = clCreateKernel(program, "vec_sub_gpu_sp", &err);
	kernels["shrink_gpu_sp"] = clCreateKernel(program, "shrink_gpu_sp", &err);
	kernels["ADM"] = clCreateKernel(program, "ADM", &err);

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

	generate_decryption_matrix(buffers, context, queue, kernels, TILE_SIZE, device, program, seeds);

	#pragma omp parallel for num_threads(4) schedule(dynamic)
	for (int i = 0; i < array_of_images.size(); i++) {
		//encrypt_image(array_of_images[i], buffers, context, queue, kernels, TILE_SIZE, i);
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

	//display_opencl_info();

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

	int k = 0;

	//display_img(img);
	//cl_mem buffer_A, buffer_A_t;
	map<string, cl_kernel> kernels;
	map<string, cl_mem> buffers;
	map<string, std::vector<float>> measurments;
	//cl_kernel kernel_mat_transpose = kernels["transpose"];
	//cl_kernel kernel_mat_mat_mul_gpu = kernels["mat_mat_mul_gpu_sp"];
	//cl_kernel power_method = kernels["power_method"];
	//cl_kernel kernel_mat_vec_mul_gpu = kernels["mat_vec_mul_gpu_fp32"];
	kernels["transpose"] = clCreateKernel(program, "transpose", &err);
	kernels["mat_mat_mul_gpu_sp"] = clCreateKernel(program, "mat_mat_mul_gpu_sp", &err);
	kernels["power_method"] = clCreateKernel(program, "power_method", &err);
	kernels["mat_vec_mul_gpu_fp32"] = clCreateKernel(program, "mat_vec_mul_gpu_fp32", &err);
	kernels["vec_scalar_gpu_sp"] = clCreateKernel(program, "vec_scalar_gpu_sp", &err);
	kernels["vec_sub_gpu_sp"] = clCreateKernel(program, "vec_sub_gpu_sp", &err);
	kernels["shrink_gpu_sp"] = clCreateKernel(program, "shrink_gpu_sp", &err);
	kernels["ADM"] = clCreateKernel(program, "ADM", &err);


	char seed[4];

	std::vector<unsigned int> seeds = passord_to_seeds(passphrase);

	generate_decryption_matrix(buffers, context, queue, kernels, img.TILE_SIZE, device, program, seeds);
	float max_eig = generate_dictionary(buffers, context, queue, kernels, img.TILE_SIZE, device, program);

	int index_step = 0;

	for (int i = 0; i < N * M; i++) {
		for (int j = 0; j < 3; j++) {
			int firstIndex = index_step;
			int lastIndex = index_step + (img.TILE_SIZE * img.TILE_SIZE);
			index_step = index_step + (img.TILE_SIZE * img.TILE_SIZE);
			vector<float>::const_iterator first = img.data_array.begin() + firstIndex;
			vector<float>::const_iterator last = img.data_array.begin() + lastIndex;
			vector<float> new_vec(first, last);
			buffers["buffer_b" + std::to_string(i) + "_" + std::to_string(j)] = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * img.TILE_SIZE * img.TILE_SIZE, NULL, NULL);
			clEnqueueWriteBuffer(queue, buffers["buffer_b" + std::to_string(i) + "_" + std::to_string(j)], CL_TRUE, 0, sizeof(float) * new_vec.size(), new_vec.data(), 0, NULL, NULL);

			vector<float> aux(img.TILE_SIZE * img.TILE_SIZE);
			clEnqueueReadBuffer(queue, buffers["buffer_b" + std::to_string(i) + "_" + std::to_string(j)], CL_TRUE, 0, sizeof(float) * aux.size(), aux.data(), 0, NULL, NULL);
			clFinish(queue);
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