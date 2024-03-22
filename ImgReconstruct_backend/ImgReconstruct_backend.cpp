#include <iostream>
#include <thread>
#include <vector>
//#include "sparse_representations.h"
#include "AVX_functions.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "sparse_representations_solver.h"
#include "sparse_representations_solver.cpp"
#include<tuple>

void mat_vec_mul_gpu(tuple <cl::Context, cl::CommandQueue, cl::Program> context, cl::Buffer &buffer_mat, cl::Buffer &buffer_vec, cl::Buffer &buffer_res, int rows, int cols)
{
	cl::Kernel matrixVectorMultiply;
	matrixVectorMultiply = cl::Kernel(get<2>(context), "mat_vec_mul_gpu_fp32");
	matrixVectorMultiply.setArg(0, buffer_mat);
	matrixVectorMultiply.setArg(1, buffer_vec);
	matrixVectorMultiply.setArg(2, buffer_res);
	matrixVectorMultiply.setArg(3, (int)rows);
	matrixVectorMultiply.setArg(4, (int)cols);
	get<1>(context).enqueueNDRangeKernel(matrixVectorMultiply, cl::NullRange, cl::NDRange((int)(rows)));
	get<1>(context).finish();
}

tuple <cl::Context, cl::CommandQueue, cl::Program> creat_opencl_context()
{
	std::vector<cl::Platform> all_platforms;
	cl::Platform::get(&all_platforms);
	cl::Platform default_platform = all_platforms[0];
	std::vector<cl::Device> all_devices;
	default_platform.getDevices(CL_DEVICE_TYPE_GPU, &all_devices);
	cl::Device default_device = all_devices[0];
	cl::Context context(default_device);
	std::ifstream src("gpu_kernels.cl");
	std::string str((std::istreambuf_iterator<char>(src)), std::istreambuf_iterator<char>());
	cl::Program::Sources sources;
	sources.push_back({ str.c_str(),str.length() });
	cl::Program program(context, sources);
	program.build({ default_device });
	cl::CommandQueue queue(context, default_device);

	tuple <cl::Context, cl::CommandQueue, cl::Program> cl_context(context, queue, program);

	return { context, queue, program };
}

void recostruct(cv::Mat& img, cv::Mat& out, float p)
{
	tuple <cl::Context, cl::CommandQueue, cl::Program> cl_context = creat_opencl_context();

	int width = img.size[1];
	int height = img.size[0];
	size_t n = width * height;
	size_t m = n - 64;// int(n * p);
	int T = 12;
	int k = 0;
	vector<float> ek(n);
	vector<float> psi(n);
	vector<float> x(n);
	vector<float> x1(n);
	vector<float> x_aux(n);
	vector<float> y(m);
	vector<float> s1(n);
	vector<float> Phi_alt(m * n);
	vector<vector<float>> Theta(m, vector<float>(n));
	vector<vector<float>> Theta_t(n, vector<float>(m));

	std::random_device e;
	std::default_random_engine generator(e());
	generator.seed(std::chrono::system_clock::now().time_since_epoch().count());
	static std::uniform_real_distribution<> dis(0, n - 1);

	//vec_fill(Phi_alt, 0.0f);

	cv::Mat floatImg;
	cv::Mat reconstructedImg;
	cv::Mat originalImg;
	img.convertTo(floatImg, CV_32FC1);
	img.convertTo(reconstructedImg, CV_32FC1);
	img.convertTo(originalImg, CV_32FC1);

	k = 0;

	for (int i = 0; i < floatImg.rows; i++)
	{
		for (int j = 0; j < floatImg.cols; j++)
		{
			x[k++] = floatImg.at<float>(j, i);
		}
	}
	vec_rand(Phi_alt);

	x_aux = x;

	vector<float> res(m);
	size_t t = 32;
	size_t m1 = 32;
	float sum = 0.0f;

	cl::Buffer buffer_A(get<0>(cl_context), CL_MEM_READ_ONLY, sizeof(float) * Phi_alt.size());
	cl::Buffer buffer_vec(get<0>(cl_context), CL_MEM_READ_WRITE, sizeof(float) * x_aux.size());
	cl::Buffer buffer_res(get<0>(cl_context), CL_MEM_READ_WRITE, sizeof(float) * res.size());

	get<1>(cl_context).enqueueWriteBuffer(buffer_A, CL_TRUE, 0, sizeof(float) * Phi_alt.size(), Phi_alt.data());
	get<1>(cl_context).enqueueWriteBuffer(buffer_vec, CL_TRUE, 0, sizeof(float) * x_aux.size(), x_aux.data());
	get<1>(cl_context).enqueueWriteBuffer(buffer_res, CL_TRUE, 0, sizeof(float) * res.size(), res.data());
	get<1>(cl_context).finish();

	mat_vec_mul_gpu(cl_context, buffer_A, buffer_vec, buffer_res, m, n);

	get<1>(cl_context).enqueueReadBuffer(buffer_res, CL_TRUE, 0, sizeof(float) * res.size(), res.data());
	get<1>(cl_context).finish();
	x_aux = res;

	y = x_aux;

	for (int i = 0; i < n; i++)
	{
		vec_fill(ek, 0.0f);
		ek[i] = 1.0f;
		cv::idct(ek, psi, 0);

		get<1>(cl_context).enqueueWriteBuffer(buffer_vec, CL_TRUE, 0, sizeof(float) * psi.size(), psi.data());
		mat_vec_mul_gpu(cl_context, buffer_A, buffer_vec, buffer_res, m, n);
		get<1>(cl_context).enqueueReadBuffer(buffer_res, CL_TRUE, 0, sizeof(float) * res.size(), res.data());
		get<1>(cl_context).finish();
		psi = res;

		Theta_t[i] = psi;
	}

	mat_transpose(Theta_t, Theta, T);

	Matrix<float> Theta_alt(m, n), y_alt(m);
	vector<float> Theta_alt_flat(n * m);
	flatten(Theta, Theta_alt_flat, 16);
	Theta_alt.data = Theta_alt_flat;
	y_alt.data = y;

	SparseRepSol<float> data(Theta_alt, y_alt);
	vector<float>sol_alt(n);
	sol_alt = data.solve_ADM_gpu(1000, 0.000001f, 0.000001f);
	s1 = sol_alt;

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

	for (int i = 0; i < floatImg.rows; i++)
	{
		for (int j = 0; j < floatImg.cols; j++)
		{
			reconstructedImg.at<float>(j, i) = x1[k++];
		}
	}
	k = 0;


	cv::Mat dst;
	reconstructedImg.convertTo(dst, CV_8U);


	out = dst;
}


int main()
{
	int k = 0;
	cv::Mat img = cv::imread("C:/images/gpu.png", cv::IMREAD_GRAYSCALE);

	if (img.empty())
	{
		std::cout << "!!! Failed imread(): image not found" << std::endl;
		// don't let the execution continue, else imshow() will crash.
	}

	float p = 2;

	cv::Mat channel_1;
	cv::Mat channel_2;
	cv::Mat channel_3;
	cv::Mat out_1;
	cv::Mat out_2;
	cv::Mat out_3;
	cv::Mat org_1;
	cv::Mat org_2;
	cv::Mat org_3;
	cv::Mat reconstructed_img;
	cv::Mat original_img;


	high_resolution_clock::time_point t1 = high_resolution_clock::now();
	recostruct(img, out_1, p);
	high_resolution_clock::time_point t2 = high_resolution_clock::now();
	float solve_time = duration_cast<milliseconds>(t2 - t1).count();
	cout << solve_time << endl;

	cv::imwrite("C:/images/gpu_out.png", out_1);
	imshow("Display window", out_1);
	k = cv::waitKey(0);

	return 1;
}