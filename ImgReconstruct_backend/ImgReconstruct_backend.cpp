#include <iostream>
#include <thread>
#include <vector>
//#include "sparse_representations.h"
#include "AVX_functions.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/photo.hpp>
#include "sparse_representations_solver.h"
#include "sparse_representations_solver.cpp"
#include <tuple>

void generateIDCT(vector<vector<float>> &IDCT)
{
	vector<float> ek(IDCT.size());
	vector<float> psi(IDCT.size());

	int numThreads = omp_get_max_threads();
	#pragma omp parallel for num_threads(numThreads) schedule(dynamic)
	for (int i = 0; i < IDCT.size(); i++)
	{
		vec_fill(ek, 0.0f);
		ek[i] = 1.0f;
		cv::idct(ek, psi, 0);
		IDCT[i] = psi;
	}
}

void recostruct(cv::Mat& img, cv::Mat& out)
{
	std::tuple <cl::Context, cl::CommandQueue, cl::Program> cl_context = creat_opencl_context();

	int width = img.size[1];
	int height = img.size[0];
	size_t n = width * height;
	size_t m = n;
	int T = 12;
	int k = 0;
	vector<float> res(m);
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
	vector<vector<float>> IDCT(n, vector<float>(n));

	cl::Buffer buffer_A(get<0>(cl_context), CL_MEM_READ_WRITE, sizeof(float) * Phi_alt.size());
	cl::Buffer buffer_vec(get<0>(cl_context), CL_MEM_READ_WRITE, sizeof(float) * x_aux.size());
	cl::Buffer buffer_res(get<0>(cl_context), CL_MEM_READ_WRITE, sizeof(float) * res.size());
	cl::Buffer buffer_theta(get<0>(cl_context), CL_MEM_READ_WRITE, sizeof(float) * m * n);

	vec_rand(Phi_alt);
	std::get<1>(cl_context).enqueueWriteBuffer(buffer_A, CL_TRUE, 0, sizeof(float) * Phi_alt.size(), Phi_alt.data());
	std::get<1>(cl_context).finish();
	Phi_alt = vector<float>();

	std::random_device e;
	std::default_random_engine generator(e());
	generator.seed(1);

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

	x_aux = x;

	float sum = 0.0f;

	for (int i = 0; i < n; i++)
	{
		vec_fill(ek, 0.0f);
		ek[i] = 1.0f;
		cv::idct(ek, psi, 0);
		IDCT[i] = psi;
	}

	for (int i = 0; i < n; i++)
	{
		psi = IDCT[i];

		std::get<1>(cl_context).enqueueWriteBuffer(buffer_vec, CL_TRUE, 0, sizeof(float) * psi.size(), psi.data());
		mat_vec_mul_gpu(cl_context, buffer_A, buffer_vec, buffer_res, m, n);
		std::get<1>(cl_context).enqueueReadBuffer(buffer_res, CL_TRUE, 0, sizeof(float) * res.size(), res.data());
		std::get<1>(cl_context).finish();
		psi = res;

		Theta_t[i] = psi;
	}

	IDCT = vector<vector<float>>();
	mat_transpose(Theta_t, Theta, T);
	Theta_t = vector<vector<float>>();

	Matrix<float> Theta_alt(m, n), y_alt(m);
	vector<float> Theta_alt_flat(n * m);

	flatten(Theta, Theta_alt_flat, 16);
	Theta = vector<vector<float>>();

	Theta_alt.rows = m;
	Theta_alt.cols = n;
	Theta_alt.data = Theta_alt_flat;
	Theta_alt_flat = vector<float>();

	std::get<1>(cl_context).enqueueWriteBuffer(buffer_vec, CL_TRUE, 0, sizeof(float) * x_aux.size(), x_aux.data());
	std::get<1>(cl_context).enqueueWriteBuffer(buffer_res, CL_TRUE, 0, sizeof(float) * res.size(), res.data());
	std::get<1>(cl_context).finish();

	mat_vec_mul_gpu(cl_context, buffer_A, buffer_vec, buffer_res, m, n);

	std::get<1>(cl_context).enqueueReadBuffer(buffer_res, CL_TRUE, 0, sizeof(float) * m, res.data());
	std::get<1>(cl_context).finish();

	y = res;

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
	cv::Mat img = cv::imread("C:/images/gpu.jpg", cv::IMREAD_COLOR);

	if (img.empty())
	{
		std::cout << "!!! Failed imread(): image not found" << std::endl;
		// don't let the execution continue, else imshow() will crash.
	}

	cv::Mat channels[3];
	cv::Mat out_channels[3];

	cv::Mat out;

	cv::Mat reconstructed_img;
	cv::Mat original_img;

	cv::split(img, channels);

	high_resolution_clock::time_point t1 = high_resolution_clock::now();

	int numThreads = 3;
	//#pragma omp parallel for num_threads(numThreads) schedule(dynamic)
	for (int i = 0; i < 3; i++) {
		recostruct(channels[i], out_channels[i]);
	}
;
	high_resolution_clock::time_point t2 = high_resolution_clock::now();
	float solve_time = duration_cast<milliseconds>(t2 - t1).count();
	cout << solve_time << endl;

	cv::merge(out_channels, 3, out);

	cv::fastNlMeansDenoising(out, out, 2);

	cv::imwrite("C:/images/gpu_out.png", out);
	imshow("Display window", out);
	k = cv::waitKey(0);

	return 1;
}