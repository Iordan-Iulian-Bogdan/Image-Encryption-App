#include "sparseImageEncryption.h"
#include "AVX_functions.h"
#include <cblas.h>
#include <future>
#include <queue>
#include <bitset>
#include <chrono>

std::mutex mtx_tile;

std::mutex mtx_print;

std::mutex mtx_cores;

int message_done = 0;
std::mutex mtx;
std::condition_variable done_variable;

int count_g = 0;

vector<vector<float>> eks_global;

std::vector<string> cores(std::thread::hardware_concurrency(), "Empty");

int num_tiles = 0;
int processed_tiles = 0;
std::mutex mtx_processed_tiles;

static vector<vector<float>> eksd;

// returns the index of a core where nothing has been scheduled on it yet
unsigned long get_free_core(string source) {

	std::random_device dev;
	std::mt19937 rng(dev());
	std::uniform_int_distribution<std::mt19937::result_type> dist(0, cores.size() - 1); 

	bool foundFreeCore = false;
	bool atLaestOneCoreFree = false;

	// checking if at least one core is empty, if none are we just return a random index
	// this should only happen on very low core count CPUs
	for (int i = 0; i < cores.size(); i++) {
		if (cores[i] == "Empty") {
			atLaestOneCoreFree = true;
		}
	}

	if (!atLaestOneCoreFree)
	{
		return 1 << dist(rng);
	}

	while (!foundFreeCore) {
		int index = dist(rng);

		// we only pick every other core to account for SMT/HT, it might be slower if two threds run on the same physical core
		if (cores[index] == "Empty" && index % 2 != 0)
		{
			cores[index] = source;
			foundFreeCore = true;
			mtx_print.lock();
			for (int i = 0; i < cores.size(); i++) {
				//std::cout << cores[i] << " " << i<<" "<<std::endl;
			}
			mtx_print.unlock();
			return 1 << index;
		}
	}
}

vector<vector<float>> fill_eks(int n) {

	vector<vector<float>> eks_aux(n, vector<float>(n));
	vector<float> ek(n);
	vector<float> psi(n);

	vec_fill(ek, 0.0f);
	vec_fill(psi, 0.0f);

	for (int i = 0; i < n; i++)
	{
		vec_fill(ek, 0.0f);
		ek[i] = 1.0f;
		cv::idct(ek, psi, 0);
		eks_aux[i] = psi;
	}

	return eks_aux;
}

// used to send messeges to the python code
void send_messege(StatusCallback callback, string s) {
	char message[50];
	sprintf(message, s.c_str());
	callback(message);
}


void generateIDCT(vector<float>& IDCT, int n)
{
	vector<float> ek(n);
	vector<float> psi(n);

	int numThreads = omp_get_max_threads();
	#pragma omp parallel for num_threads(numThreads) schedule(dynamic)
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

float generate_dictionary(map<string, cl::Buffer>& buffers_gpu, cl::Context context, cl::CommandQueue queue, map<string, cl::Kernel> kernels, uint32_t size, cl::Device device, cl::Program program, float seed = 1) {
	cl_int err;

	int width = size;
	int height = size;
	int n = width * height;
	int m = n;
	int T = 12;
	int k = 0;

	buffers_gpu["buffer_A"] = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(float) * n * m);

	k = 0;
	float sum = 0.0f;

	vector<float> IDCT_alt(n * n, 0.0f);
	generateIDCT(IDCT_alt, n);
	Matrix<float> transpose(n, n);
	transpose.data = IDCT_alt;
	transpose.data = transpose.getTransposedMatrix().data;
	buffers_gpu["buffer_IDCT"] = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(float) * IDCT_alt.size());
	queue.enqueueWriteBuffer(buffers_gpu["buffer_IDCT"], CL_TRUE, 0, sizeof(float) * IDCT_alt.size(), transpose.data.data());
	IDCT_alt = std::vector<float>();
	transpose.data = std::vector<float>();
	queue.finish();

	const int TS = 32;
	const int WPT = 8;

	kernels["mat_mat_mul_gpu_sp"].setArg(0, m);
	kernels["mat_mat_mul_gpu_sp"].setArg(1, n);
	kernels["mat_mat_mul_gpu_sp"].setArg(2, buffers_gpu["buffer_IDCT"]);
	kernels["mat_mat_mul_gpu_sp"].setArg(3, buffers_gpu["buffer_phi"]);
	kernels["mat_mat_mul_gpu_sp"].setArg(4, buffers_gpu["buffer_A"]);
	err = queue.enqueueNDRangeKernel(kernels["mat_mat_mul_gpu_sp"], cl::NullRange, cl::NDRange(n, n / WPT), cl::NDRange(TS, TS / WPT));
	queue.finish();
	buffers_gpu["buffer_A_t"] = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(float) * n * m);

	kernels["transpose"].setArg(0, buffers_gpu["buffer_A"]);
	kernels["transpose"].setArg(1, buffers_gpu["buffer_A_t"]);
	kernels["transpose"].setArg(2, n);
	kernels["transpose"].setArg(3, m);
	err = queue.enqueueNDRangeKernel(kernels["transpose"], cl::NullRange, cl::NDRange(n, m), cl::NDRange(16, 16));
	queue.finish();

	// this is the maxium eigen value of the dictionary, it's neeed for the algorithm that is used to reconstruct the original signal
	//float max_eig = generate_max_eig(context, queue, buffers, kernels, m, n);
	float max_eig = eigen_aprox_polynomial(width);

	float f = 0.000001f;
	///size_t globalSize_scalar_mul = n * m;

	kernels["vec_scalar_gpu_sp"].setArg(0, buffers_gpu["buffer_A_t"]);
	kernels["vec_scalar_gpu_sp"].setArg(1, 0.000001f);
	queue.enqueueNDRangeKernel(kernels["vec_scalar_gpu_sp"], cl::NullRange, cl::NDRange(n * m));
	queue.finish();

	return max_eig;
}

void createOpenCLcontext(openCLContext& cl_data, string device_name) {

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

	cl_data.context = context;
	cl_data.device = default_device;
	cl_data.platform = default_platform;
	cl_data.queue = queue;
	cl_data.program = program;
}

// sparse representation solver that runs on the CPU
// decrypts the data by solving the 'A * x = b' equation, where 'x' is the unkown decrypted singal and 'b' is the known encrypted signal 
// uses Alternating Direction Method optimization algorithm
void ADM_cpu(map<string, std::vector<float>>& buffers,
	int A_cols, int A_rows, float max_eig, float beta, float tau, int iterations, int index1, int index2, std::vector<float>& sol)
{

	int n = A_cols;
	int m = A_rows;

	float gamma = 1.99f - (tau * max_eig);

	std::vector<float> buffer_res(n);
	std::vector<float> buffer_y(m);
	std::vector<float> buffer_x(n);
	std::vector<float> buffer_res_aux(n);

	for (int i = 0; i < iterations; i++)
	{
		cblas_sgemv(CblasColMajor, CblasTrans, n, n, 1.0, buffers["buffer_A"].data(), n, buffer_x.data(), 1, 0.0, buffer_res.data(), 1);
		vec_sub_avx(buffer_res, buffers["buffer_b" + std::to_string(index1) + "_" + std::to_string(index2)], 1);
		vec_sub_avx(buffer_res, buffers["buffer_b" + std::to_string(index1) + "_" + std::to_string(index2)], 1);
		vec_scalar_avx(buffer_y, (1 / beta), 1);
		vec_sub_avx(buffer_res, buffer_y, 1);
		cblas_sgemv(CblasColMajor, CblasTrans, n, n, 1.0, buffers["buffer_A_t"].data(), n, buffer_res.data(), 1, 0.0, buffer_res_aux.data(), 1);
		vec_sub_avx(buffer_x, buffer_res_aux, 1);
		shrink(buffer_x, (tau / beta), 1);
		cblas_sgemv(CblasColMajor, CblasTrans, n, n, 1.0, buffers["buffer_A"].data(), n, buffer_x.data(), 1, 0.0, buffer_res_aux.data(), 1);
		vec_sub_avx(buffer_res_aux, buffers["buffer_b" + std::to_string(index1) + "_" + std::to_string(index2)], 1);
		vec_scalar_avx(buffer_res_aux, (gamma * beta), 1);
		vec_scalar_avx(buffer_y, (1 / (1 / beta)), 1);
		vec_sub_avx(buffer_y, buffer_res_aux, 1);
	}

	sol = buffer_x;
}

// sparse representation solver that runs on the CPU
// decrypts the data by solving the 'A * x = b' equation, where 'x' is the unkown decrypted singal and 'b' is the known encrypted signal 
// uses Alternating Direction Method optimization algorithm
std::vector <float> ADM_gpu(map<string, cl::Buffer>& buffers,
	int A_cols, int A_rows, float max_eig, float beta, float tau, int iterations,
	openCLContext cl_data, map<string, cl::Kernel> kernels, int index1, int index2)
{
	int n = A_cols;
	int m = A_rows;
	int err = 0;
	float gamma = 1.99f - (tau * max_eig);
	float f = 0.0f;
	std::vector<float> x1(n);

	cl::Buffer buffer_res(cl_data.context, CL_MEM_READ_WRITE, sizeof(float) * n, NULL, NULL);
	cl::Buffer buffer_y(cl_data.context, CL_MEM_READ_WRITE, sizeof(float) * m);
	cl::Buffer buffer_x(cl_data.context, CL_MEM_READ_WRITE, sizeof(float) * n);
	cl::Buffer buffer_res_aux(cl_data.context, CL_MEM_READ_WRITE, sizeof(float) * n);
	cl_data.queue.finish();

	for (int i = 0; i < iterations; i++)
	{
		mat_vec_mul_GPU(buffers["buffer_A"], buffer_x, buffer_res, m, n, cl_data, kernels);
		vec_sub_GPU(buffer_res, buffers["buffer_b" + std::to_string(index1) + "_" + std::to_string(index2)], m, cl_data, kernels);
		vec_scalar_GPU(buffer_y, (1 / beta), m, cl_data, kernels);
		vec_sub_GPU(buffer_res, buffer_y, m, cl_data, kernels);
		mat_vec_mul_GPU(buffers["buffer_A_t"], buffer_res, buffer_res_aux, n, m, cl_data, kernels);
		vec_sub_GPU(buffer_x, buffer_res_aux, m, cl_data, kernels);
		shrink_GPU(buffer_x, (tau / beta), n, cl_data, kernels);
		mat_vec_mul_GPU(buffers["buffer_A"], buffer_x, buffer_res_aux, m, n, cl_data, kernels);
		vec_sub_GPU(buffer_res_aux, buffers["buffer_b" + std::to_string(index1) + "_" + std::to_string(index2)], m, cl_data, kernels);
		vec_scalar_GPU(buffer_res_aux, (gamma * beta), m, cl_data, kernels);
		vec_scalar_GPU(buffer_y, (1 / (1 / beta)), m, cl_data, kernels);
		vec_sub_GPU(buffer_y, buffer_res_aux, m, cl_data, kernels);
	}

	cl_data.queue.enqueueReadBuffer(buffer_x, CL_TRUE, 0, sizeof(float) * x1.size(), x1.data());
	cl_data.queue.finish();

	return x1;
}

// this recontructs an image tile
void decrypt_data(cv::Mat& out, map<string, cl::Buffer>& buffers, openCLContext cl_data, map<string, cl::Kernel> kernels, uint32_t tile_size, vector<float>& sol_alt, float max_eig, int index1, int index2, int iterations) {
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

	s1 = sol_alt;
	float o = s1[n - 1];
	vec_fill(x1, 0.0f);
	vec_fill(ek, 0.0f);
	vec_fill(psi, 0.0f);

	// the contents of eksd are initially always the same so we can compute this just once, 
	// we need to make a copy becasuse this variable will be modified afterwards,
	// it's worth making a copy every time because it's still faster than computing it every time

	vector<vector<float>> eks = eksd;

	for (int i = 0; i < n; i++)
	{
		vec_scalar_avx(eks[i], s1[i], 1);
		vec_add_avx(x1, eks[i], 1);
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

void decrypt_data_gpu(StatusCallback callback, cv::Mat out, map<string, cl::Buffer> buffers, openCLContext cl_data, map<string, cl::Kernel> kernels, int TILE_SIZE, vector<vector<float>> sol_alts, float max_eig, int index, int iterations) {
	cv::Mat channels[3];
	cv::split(out, channels);

	for (int i = 0; i < 3; i++) {
		decrypt_data(channels[i], buffers, cl_data, kernels, TILE_SIZE, sol_alts[i], max_eig, index, i, iterations);
	}

	cv::merge(channels, 3, out);

	std::cout << "GPU decrypted tile " + std::to_string(index) + "\n";
	string s = "GPU decrypted tile " + std::to_string(index);
	std::thread t(send_messege, callback, s);
	t.detach();

	// incrementing the number of tiles that have finished decrypting 
	// when all tiles have been decrypted we notify the main thread since this thread has been detached from it
	mtx_processed_tiles.lock();
	processed_tiles++;
	mtx_processed_tiles.unlock();

	if (num_tiles == processed_tiles) {
		message_done = 1;
		done_variable.notify_one();
	}
}

void decrypt_image_gpu(openCLContext cl_data, StatusCallback callback, cv::Mat& out, map<string, cl::Buffer>& buffers, uint32_t TILE_SIZE, float max_eig, int index, int iterations) {

	map<string, cl::Kernel> kernels;

	kernels["transpose"] = cl::Kernel(cl_data.program, "transpose");
	kernels["mat_mat_mul_gpu_sp"] = cl::Kernel(cl_data.program, "mat_mat_mul_gpu_sp");
	kernels["mat_vec_mul_gpu_fp32"] = cl::Kernel(cl_data.program, "mat_vec_mul_gpu_fp32");
	kernels["vec_scalar_gpu_sp"] = cl::Kernel(cl_data.program, "vec_scalar_gpu_sp");
	kernels["vec_sub_gpu_sp"] = cl::Kernel(cl_data.program, "vec_sub_gpu_sp");
	kernels["shrink_gpu_sp"] = cl::Kernel(cl_data.program, "shrink_gpu_sp");

	float beta = 0.000001f;
	float tau = 0.000001f;

	int n = TILE_SIZE * TILE_SIZE;


	vector<vector<float>> sol_alts(3, vector<float>(n));
	std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

	for (int i = 0; i < 3; i++) {
		sol_alts[i] = ADM_gpu(buffers, n, n, max_eig, beta, tau, iterations, cl_data, kernels, index, i);
	}

	std::thread thread_decrypt_GPU(decrypt_data_gpu, callback, out, buffers, cl_data, kernels, TILE_SIZE, sol_alts, max_eig, index, iterations);
	mtx_cores.lock();
	static long core_decrypt_data_gpu = get_free_core("decrypt_data_gpu");
	SetThreadAffinityMask(thread_decrypt_GPU.native_handle(), core_decrypt_data_gpu);
	mtx_cores.unlock();
	// what needs to be calculated in decrypt_data_gpu blocks the execution of the next tile but we don't actually need to wait for this 
	// so we can detach the thread at this point
	thread_decrypt_GPU.detach();
}

// the dictionary is generated by multiplying the encryption matrix
// with a matrix composed of inverse cosine transforms for every position in the encrypted signal

float generate_dictionary(map<string, vector<float>>& buffers_cpu, uint32_t size, float seed = 1) {

	int width = size;
	int height = size;
	size_t n = width * height;
	size_t m = n;
	int T = 12;
	int k = 0;
	vector<float> buffer_A(n * m);
	vector<float> buffer_A_t(n * m);
	k = 0;
	float sum = 0.0f;

	vector<float> IDCT_alt(n * n, 0.0f);
	generateIDCT(IDCT_alt, n);
	Matrix<float> transpose(n, n);
	transpose.data = IDCT_alt;

	matrix_mult_avx512(transpose.data, buffers_cpu["buffer_phi"], buffer_A, n, m, n);

	float max_eig = eigen_aprox_polynomial(width);

	return max_eig;
}

float generate_dictionary(map<string, cl::Buffer>& buffers_gpu, map<string, vector<float>>& buffers_cpu, cl::Context context, cl::CommandQueue queue, map<string, cl::Kernel> kernels, uint32_t size, cl::Device device, cl::Program program, float seed = 1) {
	int err = 0;
	
	int width = size;
	int height = size;
	int n = width * height;
	int m = n;
	int T = 12;
	int k = 0;

	buffers_gpu["buffer_A"] = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(float) * n * m);

	k = 0;
	float sum = 0.0f;

	vector<float> IDCT_alt(n * n, 0.0f);
	generateIDCT(IDCT_alt, n);
	Matrix<float> transpose(n, n);
	transpose.data = IDCT_alt;
	transpose.data = transpose.getTransposedMatrix().data;
	buffers_gpu["buffer_IDCT"] = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(float) * IDCT_alt.size());
	queue.enqueueWriteBuffer(buffers_gpu["buffer_IDCT"], CL_TRUE, 0, sizeof(float) * IDCT_alt.size(), transpose.data.data());
	IDCT_alt = std::vector<float>();
	transpose.data = std::vector<float>();
	queue.finish();

	const int TS = 32;
	const int WPT = 8;

	kernels["mat_mat_mul_gpu_sp"].setArg(0, m);
	kernels["mat_mat_mul_gpu_sp"].setArg(1, n);
	kernels["mat_mat_mul_gpu_sp"].setArg(2, buffers_gpu["buffer_IDCT"]);
	kernels["mat_mat_mul_gpu_sp"].setArg(3, buffers_gpu["buffer_phi"]);
	kernels["mat_mat_mul_gpu_sp"].setArg(4, buffers_gpu["buffer_A"]);
	err = queue.enqueueNDRangeKernel(kernels["mat_mat_mul_gpu_sp"], cl::NullRange, cl::NDRange(n, n / WPT), cl::NDRange(TS, TS / WPT));
	queue.finish();
	buffers_gpu["buffer_A_t"] = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(float) * n * m);

	kernels["transpose"].setArg(0, buffers_gpu["buffer_A"]);
	kernels["transpose"].setArg(1, buffers_gpu["buffer_A_t"]);
	kernels["transpose"].setArg(2, n);
	kernels["transpose"].setArg(3, m);
	err = queue.enqueueNDRangeKernel(kernels["transpose"], cl::NullRange, cl::NDRange(n, m), cl::NDRange(16, 16));
	queue.finish();

	// this is the maxium eigen value of the dictionary, it's neeed for the algorithm that is used to reconstruct the original signal
	//float max_eig = generate_max_eig(context, queue, buffers, kernels, m, n);
	float max_eig = eigen_aprox_polynomial(width);

	float f = 0.000001f;
	///size_t globalSize_scalar_mul = n * m;

	kernels["vec_scalar_gpu_sp"].setArg(0, buffers_gpu["buffer_A_t"]);
	kernels["vec_scalar_gpu_sp"].setArg(1, 0.000001f);
	queue.enqueueNDRangeKernel(kernels["vec_scalar_gpu_sp"], cl::NullRange, cl::NDRange(n * m));
	queue.finish();
	vector<float> buffer_aux(n * m);

	queue.enqueueReadBuffer(buffers_gpu["buffer_A"], CL_TRUE, 0, sizeof(float) * buffer_aux.size(), buffer_aux.data());
	queue.finish();
	buffers_cpu["buffer_A"] = buffer_aux;
	queue.enqueueReadBuffer(buffers_gpu["buffer_A_t"], CL_TRUE, 0, sizeof(float) * buffer_aux.size(), buffer_aux.data());
	queue.finish();
	buffers_cpu["buffer_A_t"] = buffer_aux;

	return max_eig;
}

// encrypts a segment of data by multiplying the vectorized monochromatic image with the encryption matrix
// stores the encrypted data in the measurments vector
void encrypt_data(cv::Mat& img, map<string, cl::Buffer>& buffers, map<string, std::vector<float>>& measurments, openCLContext& cl_data, map<string, cl::Kernel> kernels, uint32_t tile_size, int index1, int index2)
{
	cl_int err;

	int width = img.size[1];
	int height = img.size[0];
	int n = width * height;
	int m = n;
	int k = 0;
	vector<float> res(m);
	vector<float> x(n);
	vector<float> x_aux(n);

	buffers["buffer_vec_decrypt" + std::to_string(index1) + "_" + std::to_string(index2)] = cl::Buffer(cl_data.context, CL_MEM_READ_WRITE, sizeof(float) * n);

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

	buffers["buffer_b" + std::to_string(index1) + "_" + std::to_string(index2)] = cl::Buffer(cl_data.context, CL_MEM_READ_WRITE, sizeof(float) * m);

	size_t globalSize[1] = { m };

	cl_data.queue.enqueueWriteBuffer(buffers["buffer_vec_decrypt" + std::to_string(index1) + "_" + std::to_string(index2)], CL_TRUE, 0, sizeof(float) * x_aux.size(), x_aux.data());

	kernels["mat_vec_mul_gpu_fp32"].setArg(0, buffers["buffer_phi"]);
	kernels["mat_vec_mul_gpu_fp32"].setArg(1, buffers["buffer_vec_decrypt" + std::to_string(index1) + "_" + std::to_string(index2)]);
	kernels["mat_vec_mul_gpu_fp32"].setArg(2, buffers["buffer_b" + std::to_string(index1) + "_" + std::to_string(index2)]);
	kernels["mat_vec_mul_gpu_fp32"].setArg(3, m);
	kernels["mat_vec_mul_gpu_fp32"].setArg(4, n);
	err = cl_data.queue.enqueueNDRangeKernel(kernels["mat_vec_mul_gpu_fp32"], cl::NullRange, cl::NDRange(m, 1));
	cl_data.queue.finish();

	cl_data.queue.enqueueReadBuffer(buffers["buffer_b" + std::to_string(index1) + "_" + std::to_string(index2)], CL_TRUE, 0, sizeof(float) * res.size(), res.data());

	// storing the encrypted data
	measurments["buffer_b" + std::to_string(index1) + "_" + std::to_string(index2)] = res;

	cl_data.queue.finish();
}

// encrypts an image, this will usally be a tile sized section of the original image to be encrypted
void encrypt_image(openCLContext cl_data, StatusCallback callback, cv::Mat& img, map<string, cl::Buffer>& buffers, map<string, std::vector<float>>& measurments, uint32_t TILE_SIZE, int index) {

	map<string, cl::Kernel> kernels;

	kernels["transpose"] = cl::Kernel(cl_data.program, "transpose");
	kernels["mat_mat_mul_gpu_sp"] = cl::Kernel(cl_data.program, "mat_mat_mul_gpu_sp");
	kernels["mat_vec_mul_gpu_fp32"] = cl::Kernel(cl_data.program, "mat_vec_mul_gpu_fp32");
	kernels["vec_scalar_gpu_sp"] = cl::Kernel(cl_data.program, "vec_scalar_gpu_sp");
	kernels["vec_sub_gpu_sp"] = cl::Kernel(cl_data.program, "vec_sub_gpu_sp");
	kernels["shrink_gpu_sp"] = cl::Kernel(cl_data.program, "shrink_gpu_sp");

	// extracting each color channel
	cv::Mat channels[3];
	cv::split(img, channels);

	// encrypting the data for each color channel
	for (int i = 0; i < 3; i++) {
		encrypt_data(channels[i], buffers, measurments, cl_data, kernels, TILE_SIZE, index, i);
	}

	std::cout << "GPU encrypted tile " + std::to_string(index) + "\n";
	string s = "GPU encrypted tile " + std::to_string(index);
	std::thread t(send_messege, callback, s);
	t.detach();
}

// the encryption matrix is a random matrix generated using the seeds from the passphrase
// every seed is used to generate a portion of the matrix

void generate_decryption_matrix(map<string, cl::Buffer>& buffers_gpu, map<string, vector<float>>& buffers_cpu, cl::Context context, cl::CommandQueue queue, map<string, cl::Kernel> kernels, uint32_t size, cl::Device device, cl::Program program, vector<unsigned int> seeds) {
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


	buffers_gpu["buffer_phi"] = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(float) * Phi_alt.size(), NULL, NULL);

	queue.enqueueWriteBuffer(buffers_gpu["buffer_phi"], CL_TRUE, 0, sizeof(float) * Phi_alt.size(), Phi_alt.data());

	buffers_cpu["buffer_phi"] = Phi_alt;

	queue.finish();
}

void generate_decryption_matrix(map<string, cl::Buffer>& buffers_gpu, cl::Context context, cl::CommandQueue queue, map<string, cl::Kernel> kernels, uint32_t size, cl::Device device, cl::Program program, vector<unsigned int> seeds) {
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

	buffers_gpu["buffer_phi"] = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(float) * Phi_alt.size(), NULL, NULL);
	queue.enqueueWriteBuffer(buffers_gpu["buffer_phi"], CL_TRUE, 0, sizeof(float) * Phi_alt.size(), Phi_alt.data());
	queue.finish();
}

void generate_decryption_matrix(map<string, vector<float>>& buffers_cpu, uint32_t size, vector<unsigned int> seeds) {
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

	buffers_cpu["buffer_phi"] = Phi_alt;
}

// splits an image into M by N tiles
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

void encrypt_data(cv::Mat& img, map<string, vector<float>>& buffers, map<string, std::vector<float>>& measurments, uint32_t tile_size, int index1, int index2)
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

	vector<float> b(m);
	buffers["buffer_b" + std::to_string(index1) + "_" + std::to_string(index2)] = b;
	measurments["buffer_b" + std::to_string(index1) + "_" + std::to_string(index2)] = b;
	matrix_vector_mult_avx512(buffers["buffer_phi"], x_aux, buffers["buffer_b" + std::to_string(index1) + "_" + std::to_string(index2)], n, m);

	// storing the encrypted data
	measurments["buffer_b" + std::to_string(index1) + "_" + std::to_string(index2)] = buffers["buffer_b" + std::to_string(index1) + "_" + std::to_string(index2)];
}

void encrypt_image(StatusCallback callback, cv::Mat& img, map<string, vector<float>>& buffers, map<string, std::vector<float>>& measurments, uint32_t TILE_SIZE, int index) {

	// extracting each color channel
	cv::Mat channels[3];
	cv::split(img, channels);

	// encrypting the data for each color channel
	for (int i = 0; i < 3; i++) {
		encrypt_data(channels[i], buffers, measurments, TILE_SIZE, index, i);
	}

	std::cout << "CPU encrypted tile " + std::to_string(index) + "\n";
	string s = "CPU encrypted tile " + std::to_string(index);
	std::thread t(send_messege, callback, s);
	t.detach();
}

void decrypt_data(cv::Mat& out, map<string, vector<float>>& buffers, uint32_t tile_size, vector<float>& sol_alt, float max_eig, int index1, int index2, int iterations) {
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

	s1 = sol_alt;
	float o = s1[n - 1];
	vec_fill(x1, 0.0f);
	vec_fill(ek, 0.0f);
	vec_fill(psi, 0.0f);

	//static vector<vector<float>> eksd = fill_eks(n);
	vector<vector<float>> eks = eksd;

	for (int i = 0; i < n; i++)
	{
		vec_scalar_avx(eks[i], s1[i], 1);
		vec_add_avx(x1, eks[i], 1);
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

void decrypt_data_cpu(StatusCallback callback, cv::Mat& out, map<string, vector<float>>& buffers, int TILE_SIZE, vector<vector<float>> sol_alts, float max_eig, int index, int iterations) {
	cv::Mat channels[3];
	cv::split(out, channels);

	for (int i = 0; i < 3; i++) {
		decrypt_data(channels[i], buffers, TILE_SIZE, sol_alts[i], max_eig, index, i, iterations);
	}

	cv::merge(channels, 3, out);
	std::cout << "CPU decrypted tile " + std::to_string(index) + "\n";
	string s = "CPU decrypted tile " + std::to_string(index);
	std::thread t(send_messege, callback, s);
	t.detach();

	// incrementing the number of tiles that have finished decrypting 
	// when all tiles have been decrypted we notify the main thread since this thread has been detached from it;
	mtx_processed_tiles.lock();
	processed_tiles++;
	mtx_processed_tiles.unlock();

	if (num_tiles == processed_tiles) {
		message_done = 1;
		done_variable.notify_one();
	}
}

void decrypt_image(StatusCallback callback, cv::Mat& out, map<string, vector<float>>& buffers, uint32_t TILE_SIZE, float max_eig, int index, int iterations) {
	//std::cout << "decrypt_image CPU got core : " << GetCurrentProcessorNumber() << std::endl;

	
	int n = TILE_SIZE * TILE_SIZE;
	vector<vector<float>> sol_alts(3, vector<float>(n));

	float beta = 0.000001f;
	float tau = 0.000001f;


	// every color channel can be decrypted in parallel
	vector<std::thread> CPUThreads(3);
	
	for (int i = 0; i < 3; i++) {
		CPUThreads[i] = std::thread(ADM_cpu, std::ref(buffers), n, n, max_eig, beta, tau, iterations, index, i, std::ref(sol_alts[i]));
	}

	mtx_cores.lock();
	static long core_channel_1 = get_free_core("core_channel_1");
	static long core_channel_2 = get_free_core("core_channel_2");
	static long core_channel_3 = get_free_core("core_channel_3");
	mtx_cores.unlock();

	SetThreadAffinityMask(CPUThreads[0].native_handle(), core_channel_1);
	SetThreadAffinityMask(CPUThreads[1].native_handle(), core_channel_2);
	SetThreadAffinityMask(CPUThreads[2].native_handle(), core_channel_3);

	for (int i = 0; i < 3; i++) {
		CPUThreads[i].join();
	}

	std::thread thread_decrypt_CPU(decrypt_data_cpu, callback, std::ref(out), std::ref(buffers), TILE_SIZE, sol_alts, max_eig, index, iterations);
	mtx_cores.lock();
	static long core_decrypt_data_cpu = get_free_core("decrypt_data_cpu");
	SetThreadAffinityMask(thread_decrypt_CPU.native_handle(), core_decrypt_data_cpu);
	mtx_cores.unlock();
	thread_decrypt_CPU.detach();
}


void CPUProcessingTaskEncryption(StatusCallback callback, std::vector<int>& available_tiles, std::vector<cv::Mat>& array_of_images, map<string, vector<float>>& buffers, map<string, std::vector<float>>& measurments, uint32_t TILE_SIZE) {
	mtx_tile.lock();
	int index = retAvailableTile(available_tiles); // we get a tile that hasn't been processed yet
	mtx_tile.unlock();
	if (index > -1) {
		encrypt_image(callback, array_of_images[index], buffers, measurments, TILE_SIZE, index);
		CPUProcessingTaskEncryption(callback, available_tiles, array_of_images, buffers, measurments, TILE_SIZE);
	}
}

void GPUProcessingTaskEncryption(openCLContext cl_data, StatusCallback callback, std::vector<int>& available_tiles, std::vector<cv::Mat>& array_of_images, map<string, cl::Buffer>& buffers, map<string, std::vector<float>>& measurments, uint32_t TILE_SIZE) {
	mtx_tile.lock();
	int index = retAvailableTile(available_tiles); // we get a tile that hasn't been processed yet
	mtx_tile.unlock();
	if (index > -1) {
		encrypt_image(cl_data, callback, array_of_images[index], buffers, measurments, TILE_SIZE, index);
		GPUProcessingTaskEncryption(cl_data, callback, available_tiles, array_of_images, buffers, measurments, TILE_SIZE);
	}
}

void CPUProcessingTask(StatusCallback callback, std::vector<int>& available_tiles, std::vector<cv::Mat>& array_of_images_out, map<string, vector<float>>& buffers, uint32_t TILE_SIZE, float max_eig, int iterations) {
	
	mtx_tile.lock();
	int index = retAvailableTile(available_tiles); // we get a tile that hasn't been processed yet
	mtx_tile.unlock();
	//std::cout << "CPU got tile " << std::to_string(index) << std::endl;
	if (index > -1) {
		decrypt_image(callback, array_of_images_out[index], buffers, TILE_SIZE, max_eig, index, iterations);
		CPUProcessingTask(callback, available_tiles, array_of_images_out, buffers, TILE_SIZE, max_eig, iterations);
	}
	
}

void GPUProcessingTask(openCLContext cl_data, StatusCallback callback, std::vector<int>& available_tiles, std::vector<cv::Mat>& array_of_images_out, map<string, cl::Buffer>& buffers, uint32_t TILE_SIZE, float max_eig, int iterations) {

	mtx_tile.lock();
	int index = retAvailableTile(available_tiles); // we get a tile that hasn't been processed yet
	mtx_tile.unlock();
	//std::cout << "GPU got tile " << std::to_string(index) << std::endl;
	if (index > -1) {
		decrypt_image_gpu(cl_data, callback, array_of_images_out[index], buffers, TILE_SIZE, max_eig, index, iterations);
		GPUProcessingTask(cl_data, callback, available_tiles, array_of_images_out, buffers, TILE_SIZE, max_eig, iterations);
	}
}

// encrypts an image and returns an encryptionImage struct that contains the encrypted data

encryptionImage encryptImage(StatusCallback callback,
	cv::Mat img, /* image to be encrypted */
	int TILE_SIZE, /* size of tiles in which the image is broken up and processed, larger tiles may provide better quality at the cost of memory and speed */
	string passphrase, /* passphare used to generate the encryption matrix */
	int acceleration,
	int threads /* number of tiles to be encrypted simultaneously */) {

	if (img.empty()) {
		throw std::runtime_error("Image empty");
	}

	if (TILE_SIZE < 32) {
		throw std::runtime_error("Tile size needs to be at least 32");
	}

	if (img.size[1] < 256 && img.size[0] < 256) {
		throw std::runtime_error("Image size must be at least 256x256");
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

	char message[50];
	sprintf(message, "Number of tiles %d", (N * M));
	callback(message);

	// initializing opencl context
	openCLContext cl_data;
	createOpenCLcontext(cl_data, "gfx1100");

	int k = 0;

	map<string, cl::Kernel> kernels;
	map<string, cl::Buffer> buffers_gpu;
	map<string, std::vector<float>> measurments;
	map<string, vector<float>> buffers_cpu;

	kernels["transpose"] = cl::Kernel(cl_data.program, "transpose");
	kernels["mat_mat_mul_gpu_sp"] = cl::Kernel(cl_data.program, "mat_mat_mul_gpu_sp");
	kernels["mat_vec_mul_gpu_fp32"] = cl::Kernel(cl_data.program, "mat_vec_mul_gpu_fp32");
	kernels["vec_scalar_gpu_sp"] = cl::Kernel(cl_data.program, "vec_scalar_gpu_sp");
	kernels["vec_sub_gpu_sp"] = cl::Kernel(cl_data.program, "vec_sub_gpu_sp");
	kernels["shrink_gpu_sp"] = cl::Kernel(cl_data.program, "shrink_gpu_sp");

	cv::Mat channels[3];
	cv::Mat out_channels[3];
	cv::split(img, channels);
	cv::Mat out = img;
	cv::split(img, out_channels);
	cv::Mat reconstructed_img;
	cv::Mat original_img;

	std::vector<cv::Mat> array_of_images = splitImage(img, N, M);
	std::vector<int> array_of_processed_images(array_of_images.size(), 1);
	std::vector<unsigned int> seeds = passord_to_seeds(passphrase);

	vector<std::thread> GPUProcessing(1);
	vector<std::thread> CPUProcessing(1);

	// starting threads which process each tile based on the type of acceleration
	switch (acceleration) {
	case HYBRID_ACCELERATION:
		generate_decryption_matrix(buffers_gpu, buffers_cpu, cl_data.context, cl_data.queue, kernels, TILE_SIZE, cl_data.device, cl_data.program, seeds);
		GPUProcessing[0] = std::thread(GPUProcessingTaskEncryption, std::ref(cl_data), std::ref(callback), std::ref(array_of_processed_images), std::ref(array_of_images), std::ref(buffers_gpu), std::ref(measurments), std::ref(TILE_SIZE));
		CPUProcessing[0] = std::thread(CPUProcessingTaskEncryption, std::ref(callback), std::ref(array_of_processed_images), std::ref(array_of_images), std::ref(buffers_cpu), std::ref(measurments), std::ref(TILE_SIZE));
		break;

	case CPU_ACCELERATION:
		generate_decryption_matrix(buffers_cpu, TILE_SIZE, seeds);
		CPUProcessing[0] = std::thread(CPUProcessingTaskEncryption, std::ref(callback), std::ref(array_of_processed_images), std::ref(array_of_images), std::ref(buffers_cpu), std::ref(measurments), std::ref(TILE_SIZE));
		break;

	case GPU_ACCELERATION:
		generate_decryption_matrix(buffers_gpu, cl_data.context, cl_data.queue, kernels, TILE_SIZE, cl_data.device, cl_data.program, seeds);
		GPUProcessing[0] = std::thread(GPUProcessingTaskEncryption, std::ref(cl_data), std::ref(callback), std::ref(array_of_processed_images), std::ref(array_of_images), std::ref(buffers_gpu), std::ref(measurments), std::ref(TILE_SIZE));
		break;

	default:
		generate_decryption_matrix(buffers_cpu, TILE_SIZE, seeds);
		CPUProcessing[0] = std::thread(CPUProcessingTaskEncryption, std::ref(callback), std::ref(array_of_processed_images), std::ref(array_of_images), std::ref(buffers_cpu), std::ref(measurments), std::ref(TILE_SIZE));
		break;
	}

	// waiting for the threads to finish
	switch (acceleration) {
	case HYBRID_ACCELERATION:
		CPUProcessing[0].join();
		GPUProcessing[0].join();
		break;

	case CPU_ACCELERATION:
		CPUProcessing[0].join();
		break;

	case GPU_ACCELERATION:
		GPUProcessing[0].join();
		break;

	default:
		CPUProcessing[0].join();
		break;
	}

	std::vector<float> data_array;

	for (int i = 0; i < array_of_images.size(); i++) {
		for (int j = 0; j < 3; j++) {
			data_array.insert(data_array.end(), measurments["buffer_b" + std::to_string(i) + "_" + std::to_string(j)].begin(), measurments["buffer_b" + std::to_string(i) + "_" + std::to_string(j)].end());
		}
	}

	encryptionImage img_encrypted{ TILE_SIZE, N * M, original_width, original_height, processed_width, processed_height , data_array };

	return img_encrypted;
}

//decrypts an image stored in a encryptionImage struct format and returns the decrypted image 

void decryptImage(StatusCallback callback,
	cv::Mat& out_img,
	encryptionImage img, /* struct containing encrypted image */
	string passphrase, /* passphare used to generate the encryption matrix, must be the same as the one used at encryption time */
	int acceleration,
	int threads, /* number of tiles to be encrypted simultaneously */
	int iterations, /* the larger the tile the less need for more iterations */
	bool removeNoise,
	tile_range range) { /* enables noise reduction */

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

	num_tiles = (N * M);

	char message[50];
	sprintf(message, "Threads : %d", threads);
	callback(message);
	sprintf(message, "Number of tiles %d", (N * M));
	callback(message);
	sprintf(message, "Acceleration type : %d", acceleration);
	callback(message);

	// initializing opencl context
	openCLContext cl_data;
	createOpenCLcontext(cl_data, "gfx1100");

	int k = 0;

	map<string, cl::Kernel> kernels;
	map<string, cl::Buffer> buffers;
	map<string, std::vector<float>> measurments;
	map<string, vector<float>> buffers_cpu;

	kernels["transpose"] = cl::Kernel(cl_data.program, "transpose");
	kernels["mat_mat_mul_gpu_sp"] = cl::Kernel(cl_data.program, "mat_mat_mul_gpu_sp");
	kernels["mat_vec_mul_gpu_fp32"] = cl::Kernel(cl_data.program, "mat_vec_mul_gpu_fp32");
	kernels["vec_scalar_gpu_sp"] = cl::Kernel(cl_data.program, "vec_scalar_gpu_sp");
	kernels["vec_sub_gpu_sp"] = cl::Kernel(cl_data.program, "vec_sub_gpu_sp");
	kernels["shrink_gpu_sp"] = cl::Kernel(cl_data.program, "shrink_gpu_sp");

	std::vector<unsigned int> seeds = passord_to_seeds(passphrase);

	float max_eig = eigen_aprox_polynomial(img.TILE_SIZE);
	int index_step = 0;

	for (int i = 0; i < N * M; i++) {
		for (int j = 0; j < 3; j++) {
			int firstIndex = index_step;
			int lastIndex = index_step + (img.TILE_SIZE * img.TILE_SIZE);
			index_step = index_step + (img.TILE_SIZE * img.TILE_SIZE);
			vector<float>::const_iterator first = img.data_array.begin() + firstIndex;
			vector<float>::const_iterator last = img.data_array.begin() + lastIndex;
			vector<float> new_vec(first, last);
			buffers["buffer_b" + std::to_string(i) + "_" + std::to_string(j)] = cl::Buffer(cl_data.context, CL_MEM_READ_WRITE, sizeof(float) * img.TILE_SIZE * img.TILE_SIZE);
			cl_data.queue.enqueueWriteBuffer(buffers["buffer_b" + std::to_string(i) + "_" + std::to_string(j)], CL_TRUE, 0, sizeof(float) * new_vec.size(), new_vec.data());
		}
	}

	index_step = 0;

	for (int i = 0; i < N * M; i++) {
		for (int j = 0; j < 3; j++) {
			int firstIndex = index_step;
			int lastIndex = index_step + (img.TILE_SIZE * img.TILE_SIZE);
			index_step = index_step + (img.TILE_SIZE * img.TILE_SIZE);
			vector<float>::const_iterator first = img.data_array.begin() + firstIndex;
			vector<float>::const_iterator last = img.data_array.begin() + lastIndex;
			vector<float> new_vec(first, last);
			buffers_cpu["buffer_b" + std::to_string(i) + "_" + std::to_string(j)] = new_vec;
		}
	}

	std::vector<cv::Mat> array_of_images_out = splitImage(outputImg, N, M);
	std::vector<int> array_of_processed_images(array_of_images_out.size(), 0);

	for (int i = range.first; i <= range.last; i++) {
		array_of_processed_images[i] = 1;
	}

	cv::Mat empty_mat(cv::Size(img.TILE_SIZE, img.TILE_SIZE), CV_8UC3);
	empty_mat.setTo(cv::Scalar(0, 0, 0));

	for (int i = 0; i < array_of_processed_images.size(); i++) {
		if (array_of_processed_images[i] == 0) {
			array_of_images_out[i] = empty_mat;
		}
	}

	num_tiles = N * M;
	eksd = fill_eks(img.TILE_SIZE * img.TILE_SIZE);
	//eks_global.resize(img.TILE_SIZE * img.TILE_SIZE, vector<float>(img.TILE_SIZE * img.TILE_SIZE));
	//fill_eks(eks_global, img.TILE_SIZE);
	vector<std::thread> GPUProcessing(1);
	vector<std::thread> CPUProcessing(1);
	int kdghsu = 32;
	// starting threads which process each tile based on the type of acceleration
	switch (acceleration) {
	case HYBRID_ACCELERATION:
		generate_decryption_matrix(buffers, cl_data.context, cl_data.queue, kernels, img.TILE_SIZE, cl_data.device, cl_data.program, seeds);
		generate_dictionary(buffers, buffers_cpu, cl_data.context, cl_data.queue, kernels, img.TILE_SIZE, cl_data.device, cl_data.program);
		CPUProcessing[0] = std::thread(CPUProcessingTask, std::ref(callback), std::ref(array_of_processed_images), std::ref(array_of_images_out), std::ref(buffers_cpu), std::ref(img.TILE_SIZE), std::ref(max_eig), std::ref(iterations));
		
		mtx_cores.lock();
		static long core_cpu = get_free_core("core_cpu");
		SetThreadAffinityMask(GetCurrentThread(), core_cpu);
		mtx_cores.unlock();
		
		
		GPUProcessing[0] = std::thread(GPUProcessingTask, std::ref(cl_data), std::ref(callback), std::ref(array_of_processed_images), std::ref(array_of_images_out), std::ref(buffers), std::ref(img.TILE_SIZE), std::ref(max_eig), std::ref(iterations));
		
		mtx_cores.lock();
		static long core_gpu = get_free_core("core_gpu");
		SetThreadAffinityMask(GetCurrentThread(), core_gpu);
		mtx_cores.unlock();

		break;

	case CPU_ACCELERATION:
		generate_decryption_matrix(buffers, cl_data.context, cl_data.queue, kernels, img.TILE_SIZE, cl_data.device, cl_data.program, seeds);
		generate_dictionary(buffers, buffers_cpu, cl_data.context, cl_data.queue, kernels, img.TILE_SIZE, cl_data.device, cl_data.program);
		CPUProcessing[0] = std::thread(CPUProcessingTask, std::ref(callback), std::ref(array_of_processed_images), std::ref(array_of_images_out), std::ref(buffers_cpu), std::ref(img.TILE_SIZE), std::ref(max_eig), std::ref(iterations));
		break;

	case GPU_ACCELERATION:
		generate_decryption_matrix(buffers, cl_data.context, cl_data.queue, kernels, img.TILE_SIZE, cl_data.device, cl_data.program, seeds);
		generate_dictionary(buffers, cl_data.context, cl_data.queue, kernels, img.TILE_SIZE, cl_data.device, cl_data.program);
		GPUProcessing[0] = std::thread(GPUProcessingTask, std::ref(cl_data), std::ref(callback), std::ref(array_of_processed_images), std::ref(array_of_images_out), std::ref(buffers), std::ref(img.TILE_SIZE), std::ref(max_eig), std::ref(iterations));

		mtx_cores.lock();
		static long core_gpu_only = get_free_core("core_gpu_only");
		SetThreadAffinityMask(GPUProcessing[0].native_handle(), core_gpu_only);
		mtx_cores.unlock();
		//mtx_cores.lock();
		//SetThreadAffinityMask(GPUProcessing[0].native_handle(), get_free_core(cores, true));
		//mtx_cores.unlock();
		break;

	default:
		generate_decryption_matrix(buffers, cl_data.context, cl_data.queue, kernels, img.TILE_SIZE, cl_data.device, cl_data.program, seeds);
		generate_dictionary(buffers, buffers_cpu, cl_data.context, cl_data.queue, kernels, img.TILE_SIZE, cl_data.device, cl_data.program);
		CPUProcessing[0] = std::thread(CPUProcessingTask, std::ref(callback), std::ref(array_of_processed_images), std::ref(array_of_images_out), std::ref(buffers_cpu), std::ref(img.TILE_SIZE), std::ref(max_eig), std::ref(iterations));
		break;
	}

	// waiting for the threads to finish
	switch (acceleration) {
	case HYBRID_ACCELERATION:
		CPUProcessing[0].join();
		GPUProcessing[0].join();
		break;

	case CPU_ACCELERATION:
		CPUProcessing[0].join();
		break;

	case GPU_ACCELERATION:
		GPUProcessing[0].join();
		break;

	default:
		CPUProcessing[0].join();
		break;
	}
	for (int i = 0; i < array_of_images_out.size(); i++) {
		//cv::imwrite("CPU decrypted tile " + std::to_string(i), array_of_images_out[i]);
	}
	
	std::unique_lock<std::mutex> lock(mtx);
	done_variable.wait(lock, [] { return message_done == 1; });

	// stitching the tiles back together and resizing the image to the original size
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

	// cleaning up noise
	if (removeNoise) {
		cv::fastNlMeansDenoising(final_image2, final_image2, 3);
	}
	out_img = final_image2;

	cv::imwrite("output.png", final_image2);
}

void decryptAndWriteFile(StatusCallback callback, const char* input, const char* output, const char* passphrase, int acceleration, int threads, int iterations, bool removeNoise) {

	uint32_t TILE_SIZE = 64;

	cv::Mat out;

	int N = 1, M = threads;
	encryptionImage img_encrypted;
	readFromFile(input, img_encrypted);
	cv::Mat outputImg(cv::Size(img_encrypted.original_width, img_encrypted.original_height), CV_8UC3);
	std::vector<tile_range> ranges(threads);

	int segment = img_encrypted.num_tiles / threads;

	ranges[0].first = 0;
	ranges[0].last = segment;

	for (int i = 1; i < threads; i++) {
		ranges[i].first = ranges[i - 1].last + 1;
		ranges[i].last = ranges[i].first + segment;
	}

	ranges[ranges.size() - 1].last = img_encrypted.num_tiles - 1;

	std::vector<cv::Mat> array_of_images_out = splitImage(outputImg, N, M);

	for (int i = 0; i < threads; i++) {
		array_of_images_out[i].setTo(cv::Scalar::all(0));
	}

	int numThreads = threads; // Number of cores
	omp_set_num_threads(numThreads);

	#pragma omp parallel for schedule(dynamic)
	for (int i = 0; i < threads; i++) {
		decryptImage(callback, array_of_images_out[i], img_encrypted, "5v48v5832v5924", acceleration, 1, 300, false, ranges[i]);
	}

	out = array_of_images_out[0];
	for (int i = 1; i < threads; i++) {
		cv::add(array_of_images_out[i], out, out);
	}

	cv::imwrite(output, out);

	char message[50];
	sprintf(message, "Finished decrypting");
	callback(message);
}

const char* replaceSubstring(const char* input, const char* oldSubstring, const char* newSubstring) {
	// Calculate lengths of the input and substrings
	size_t inputLen = std::strlen(input);
	size_t oldLen = std::strlen(oldSubstring);
	size_t newLen = std::strlen(newSubstring);

	// Estimate the maximum length of the result string
	size_t maxLen = inputLen + (newLen - oldLen) * 10; // Adjust the multiplier as needed
	char* result = new char[maxLen];
	result[0] = '\0';

	const char* pos = input;
	while ((pos = std::strstr(pos, oldSubstring)) != nullptr) {
		// Copy part before the old substring
		strncat(result, input, pos - input);
		// Append the new substring
		strcat(result, newSubstring);
		// Move past the old substring
		pos += oldLen;
		input = pos;
	}
	// Append the remaining part of the input string
	strcat(result, input);

	// Return the result as a const char*
	return result;
}

void encryptAndWriteFile(StatusCallback callback, const char* input, const char* output, const char* passphrase, int TILE_SIZE, int acceleration, int threads, bool upscaling_enable) {
	std::cout << input << std::endl;
	std::cout << output << std::endl;
	cv::Mat img = cv::imread(input, cv::IMREAD_COLOR);

	encryptionImage img_encrypted = encryptImage(callback, img, TILE_SIZE, passphrase, acceleration, threads);
	writeToFile(output, img_encrypted);

	if (upscaling_enable) {
		int original_width = img.size[1];
		int original_height = img.size[0];

		const char* oldSubstring = ".se";
		const char* newSubstring = "DOWNSAMPLED.se";

		const char* result = replaceSubstring(output, oldSubstring, newSubstring);

		cv::resize(img, img, cv::Size(original_width / 2, original_height / 2));
		encryptionImage img_encrypted_downsampled = encryptImage(callback, img, TILE_SIZE, passphrase, acceleration, threads);
		writeToFile(result, img_encrypted_downsampled);
	}

	char message[50];
	sprintf(message, "Finished encrypting");
	callback(message);
}

void deleteOriginalImage(StatusCallback callback, const char* input, bool scramble) {
	cv::Mat img = cv::imread(input, cv::IMREAD_COLOR);

	if (img.empty()) {
		std::cerr << "Could not open or find the image!" << std::endl;
	}

	if (scramble) {
		int rows = img.rows;
		int cols = img.cols;
		std::random_device e;
		std::default_random_engine generator(e());
		generator.seed(1);
		static std::uniform_int_distribution<> dis(0, 255);

		for (int row = 0; row < rows; ++row) {
			for (int col = 0; col < cols; ++col) {
				img.at<cv::Vec3b>(row, col)[0] = dis(generator);
				img.at<cv::Vec3b>(row, col)[1] = dis(generator);
				img.at<cv::Vec3b>(row, col)[2] = dis(generator);
			}
		}
	}

	cv::imwrite(input, img);

	if (std::remove(input) == 0) {
		std::cout << "File deleted successfully.\n";
	}
	else {
		std::cerr << "Error deleting file.\n";
	}

	if (scramble) {
		char message[50];
		sprintf(message, "Scrambled");
		callback(message);
	}
}