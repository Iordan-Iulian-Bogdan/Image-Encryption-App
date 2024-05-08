
#include "sparse_representations_solver.h"
#include <math.h>
#include <fstream>
#include <vector>

template <class T>
T sign(T f) {
	if (f > 0.0f)
		return 1;
	return (f == 0) ? 0 : -1;
}

template <class T>
T max_val(T a, T b)
{
	if (a > b)
		return a;
	else
		return b;
}

template <class T>
SparseRepSol<T>::SparseRepSol(Matrix<T>& A, Matrix<T>& b): A(A), b(b) {
	A_t = A;
	A_t.transposeMatrix();
}

template <class T>
Matrix<T> SparseRepSol<T>::getDictionary() const {
    return A;
}

template <class T>
Matrix<T> SparseRepSol<T>::getMeasurement() const {
    return b;
}

template <class T>
T SparseRepSol<T>::power_method_gpu() {
	return 1.0f;
}


template <class T>
T SparseRepSol<T>::power_method() {
    Matrix<T> X(A.getRows(), A.getRows());
    Matrix<T> b_k(A.getRows()), aux(A.getRows()), b_k1(A.getRows());
    T norm_b_k1 = 0.0f, eig = 0.0f;

    X = A * A.getTransposedMatrix();

	b_k.fill(1.0f);

	for (unsigned int i = 0; i < 10; i++)
	{
		b_k = X * b_k;
		b_k1 = b_k;
		norm_b_k1 = b_k1.norm();
		aux = b_k1;
		b_k1 = b_k1 * (1 / norm_b_k1);
		b_k = b_k1;
		b_k1 = aux;
	}

	aux = b_k;
	b_k = X * b_k;
	b_k = b_k.getTransposedMatrix() * aux;
	aux = aux.getTransposedMatrix() * aux;

	is_max_eig_set = true;

	return  b_k.at(0, 0) / aux.at(0, 0);
}

template <class T>
Matrix<T> SparseRepSol<T>::shrink(Matrix<T> M, T threshold) {

	for (size_t i = 0; i < M.getRows(); ++i)
		for (size_t j = 0; j < M.getCols(); ++j)
		{
			M.at(i,j) = sign(M.at(i, j)) * max_val(abs(M.at(i, j)) - threshold, 0.0f);
			if (abs(M.at(i, j)) <= 1.175e-38) //  this is done in order to get rid of "negative zeros" (-0.0f)
				M.at(i, j) = 0.0f;
		}

	return M;
}

template <class T>
std::vector<T> SparseRepSol<T>::solve_ADM(uint64_t iterations, T tau, T beta) {

	if (!is_max_eig_set) {
		power_method();
		is_max_eig_set = true;
	}

	T gamma = 1.99f - (tau * max_eig);
	Matrix<T> x_k(A.getCols()), y_k(A.getRows());
	Matrix<T> aux1(A.getCols()), aux2(A.getRows());
	x_k.fill(0.0f);
	y_k.fill(0.0f);
	Matrix<T> A_t = A.getTransposedMatrix();
	//Matrix<T> B = (A.getTransposedMatrix() * tau) * (A * x_k - b - (y_k * (1 / beta)));

	for (uint64_t i = 0; i <= iterations; i++)
	{
		x_k = shrink((x_k - (A_t * tau) * (A * x_k - b - (y_k * (1 / beta)))), tau / beta);
		y_k = y_k - (A * x_k - b) * (gamma * beta);
	}

	sol = x_k.data;
	return sol;
}

template <class T>
std::vector<T> SparseRepSol<T>::solve_ADM_gpu(uint64_t iterations, T beta, T tau) {
	//	checking if the maximum singular value has been set otherwise we'll compute it

	cl_platform_id platform;
	cl_device_id device;
	cl_context context;
	cl_command_queue queue;
	cl_program program;
	cl_int err;
	cl_mem buffer_A, buffer_A_t;

	clGetPlatformIDs(1, &platform, NULL);
	clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
	context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
	queue = clCreateCommandQueueWithProperties(context, device, 0, &err);

	cl_queue_properties properties[] = { CL_QUEUE_PROPERTIES, (cl_command_queue_properties)(CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE | CL_QUEUE_ON_DEVICE | CL_QUEUE_ON_DEVICE_DEFAULT), 0 };
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

	//buffer_A = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(T) * A.data.size(), NULL, NULL);
	//buffer_A_t = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(T) * A_t.data.size(), NULL, NULL);
	//buffer_A = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * A.data.size(), A.data.data(), NULL);
	//buffer_A_t = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * A_t.data.size(), A_t.data.data(), NULL);
	//buffer_A = A_gpu;
	//buffer_A_t = A_t_gpu;
	//clEnqueueCopyBuffer(queue, A_gpu, buffer_A, 0, 0, sizeof(T) * A.data.size(), 0, NULL, NULL);
	//clEnqueueCopyBuffer(queue, A_t_gpu, buffer_A_t, 0, 0, sizeof(T) * A.data.size(), 0, NULL, NULL);

	cl_kernel mat_vec_mul_gpu_sp = clCreateKernel(program, "mat_vec_mul_gpu_fp32", &err);
	cl_kernel vec_sub_gpu_sp = clCreateKernel(program, "vec_sub_gpu_sp", &err);
	cl_kernel vec_scalar_gpu_sp = clCreateKernel(program, "vec_scalar_gpu_sp", &err);
	cl_kernel shrink_gpu_sp = clCreateKernel(program, "shrink_gpu_sp", &err);

	int n = A.cols;
	int m = A.rows;
	//	gamma is required to be less than 2 in order to ensure convergence
	//	note that if the maximum singular value are too big this can no longer be true
	//	and you'd need to change tau and beta

	std::vector<T> aux1(n), aux2(n), aux3(n), aux_b(m), aux_x(n), b1(m), x1(n);
	std::vector<T> y(m);
	std::vector<T> r(m);

	//A_t = A_t * tau;
	b1 = b.data;

	//clReleasePlatform(platform);
	//clReleaseMemObject(buffer_A_t);
	buffer_A_t = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * A_t.data.size(), A_t.data.data(), NULL);
	buffer_A = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * A.data.size(), A.data.data(), NULL);

	T f = tau;
	size_t globalSize = n * m;
	clSetKernelArg(vec_scalar_gpu_sp, 0, sizeof(cl_mem), &buffer_A_t);
	clSetKernelArg(vec_scalar_gpu_sp, 1, sizeof(T), &f);
	err = clEnqueueNDRangeKernel(queue, vec_scalar_gpu_sp, 1, NULL, &globalSize, NULL, 0, NULL, NULL);
	clFinish(queue);
	//A_t = A_t * tau;
	//buffer_A_t = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * A_t.data.size(), A_t.data.data(), NULL);

	std::vector <float> result(A.data.size());
	clEnqueueReadBuffer(queue, buffer_A_t, CL_TRUE, 0, sizeof(float) * result.size(), result.data(), 0, NULL, NULL);
	clFinish(queue);

	float gamma = 1.99f - (tau * max_eig);

	//	Implemeting the following ADM algorithm :
	//	Input: τ, β, γ dictionary A, measurement b, x = 0, y = 0
	//	While not converge
	//	x(k)←shrink(x(k)-τA*(Ax(k)-b-y(k)/β),τ/β)
	//	y(k+1)←y(k)-γβ(Ax(k+1)-b)
	//	end while
	//	Output: x(k)

	//std::vector<float> A_flat(n * m);
	//std::vector<float> A_t_flat(n * m);
	std::vector<float> aux1_flat(n), res_flat(m);

	//A_flat = A.data;
	//A_t_flat = A_t.data;

	A.data = std::vector<T>();
	A_t.data = std::vector<T>();

	//cl::Buffer buffer_A(context, CL_MEM_READ_ONLY, sizeof(float) * A_flat.size());
	//cl::Buffer buffer_A_t(context, CL_MEM_READ_ONLY, sizeof(float) * A_t_flat.size());

	cl_mem buffer_aux1 = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * n, NULL, NULL);
	cl_mem buffer_x = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * n, NULL, NULL);
	cl_mem buffer_sol = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * n, NULL, NULL);
	cl_mem buffer_aux_y = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * m, NULL, NULL);
	cl_mem buffer_y = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * m, NULL, NULL);
	cl_mem buffer_res_x = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * m, NULL, NULL);
	cl_mem buffer_res_b = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * n, NULL, NULL);
	cl_mem buffer_b = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * b.data.size(), b.data.data(), NULL);

	T fill = 0.0f;
	//clEnqueueFillBuffer(queue, buffer_aux1, &fill, sizeof(T), 0, aux1_flat.size(), 0, NULL, NULL);
	//clEnqueueFillBuffer(queue, buffer_x, &fill, sizeof(T), 0, n, 0, NULL, NULL);
	//clEnqueueFillBuffer(queue, buffer_sol, &fill, sizeof(T), 0, n, 0, NULL, NULL);
	//clEnqueueFillBuffer(queue, buffer_aux_y, &fill, sizeof(T), 0, m, 0, NULL, NULL);
	//clEnqueueFillBuffer(queue, buffer_y, &fill, sizeof(T), 0, m, 0, NULL, NULL);
	//clEnqueueFillBuffer(queue, buffer_res_x, &fill, sizeof(T), 0, m, 0, NULL, NULL);
	//clEnqueueFillBuffer(queue, buffer_res_b, &fill, sizeof(T), 0, n, 0, NULL, NULL);
	//clEnqueueFillBuffer(queue, buffer_b, &fill, sizeof(T), 0, m, 0, NULL, NULL);

	clEnqueueWriteBuffer(queue, buffer_b, CL_TRUE, 0, sizeof(float) * b1.size(), b1.data(), 0, NULL, NULL);
	size_t globalSize_m[2] = { m, 1 };
	size_t globalSize_n[2] = { n, 1 };
	clFinish(queue);
	//A_flat = std::vector<T>();
	//A_t_flat = std::vector<T>();

	std::vector <float> d(b.data.size());
	clEnqueueReadBuffer(queue, buffer_b, CL_TRUE, 0, sizeof(float) * d.size(), d.data(), 0, NULL, NULL);
	clFinish(queue);

	std::vector <float> res(m);

	for (int i = 0; i <= iterations; i++)
	{
		clSetKernelArg(mat_vec_mul_gpu_sp, 0, sizeof(cl_mem), &buffer_A);
		clSetKernelArg(mat_vec_mul_gpu_sp, 1, sizeof(cl_mem), &buffer_aux1);
		clSetKernelArg(mat_vec_mul_gpu_sp, 2, sizeof(cl_mem), &buffer_res_x);
		clSetKernelArg(mat_vec_mul_gpu_sp, 3, sizeof(int), &m);
		clSetKernelArg(mat_vec_mul_gpu_sp, 4, sizeof(int), &n);
		err = clEnqueueNDRangeKernel(queue, mat_vec_mul_gpu_sp, 1, NULL, globalSize_m, NULL, 0, NULL, NULL);
		clEnqueueReadBuffer(queue, buffer_aux1, CL_TRUE, 0, sizeof(float) * res.size(), res.data(), 0, NULL, NULL);
		clFinish(queue);
		//mat_vec_mul_gpu_fp32.setArg(0, buffer_A);
		//mat_vec_mul_gpu_fp32.setArg(1, buffer_aux1);
		//mat_vec_mul_gpu_fp32.setArg(2, buffer_res_x);
		//mat_vec_mul_gpu_fp32.setArg(3, m);
		//mat_vec_mul_gpu_fp32.setArg(4, n);
		//queue.enqueueNDRangeKernel(mat_vec_mul_gpu_fp32, cl::NullRange, cl::NDRange(m));


		clSetKernelArg(vec_sub_gpu_sp, 0, sizeof(cl_mem), &buffer_res_x);
		clSetKernelArg(vec_sub_gpu_sp, 1, sizeof(cl_mem), &buffer_b);
		err = clEnqueueNDRangeKernel(queue, vec_sub_gpu_sp, 1, NULL, globalSize_m, NULL, 0, NULL, NULL);
		clEnqueueReadBuffer(queue, buffer_res_x, CL_TRUE, 0, sizeof(float) * res.size(), res.data(), 0, NULL, NULL);
		clFinish(queue);
		//vec_sub_gpu_sp.setArg(0, buffer_res_x);
		//vec_sub_gpu_sp.setArg(1, buffer_b);
		//queue.enqueueNDRangeKernel(vec_sub_gpu_sp, cl::NullRange, cl::NDRange(m));
		f = (1 / beta);
		clSetKernelArg(vec_scalar_gpu_sp, 0, sizeof(cl_mem), &buffer_aux_y);
		clSetKernelArg(vec_scalar_gpu_sp, 1, sizeof(T), &f);
		err = clEnqueueNDRangeKernel(queue, vec_scalar_gpu_sp, 1, NULL, globalSize_m, NULL, 0, NULL, NULL);
		clEnqueueReadBuffer(queue, buffer_aux_y, CL_TRUE, 0, sizeof(float) * res.size(), res.data(), 0, NULL, NULL);
		clFinish(queue);
		//vec_scalar_gpu_sp.setArg(0, buffer_aux_y);
		//vec_scalar_gpu_sp.setArg(1, (1 / beta));
		//queue.enqueueNDRangeKernel(vec_scalar_gpu_sp, cl::NullRange, cl::NDRange(m));

		clSetKernelArg(vec_sub_gpu_sp, 0, sizeof(cl_mem), &buffer_res_x);
		clSetKernelArg(vec_sub_gpu_sp, 1, sizeof(cl_mem), &buffer_aux_y);
		err = clEnqueueNDRangeKernel(queue, vec_sub_gpu_sp, 1, NULL, globalSize_m, NULL, 0, NULL, NULL);
		clEnqueueReadBuffer(queue, buffer_res_x, CL_TRUE, 0, sizeof(float) * res.size(), res.data(), 0, NULL, NULL);
		clFinish(queue);
		//vec_sub_gpu_sp.setArg(0, buffer_res_x);
		//vec_sub_gpu_sp.setArg(1, buffer_aux_y);
		//queue.enqueueNDRangeKernel(vec_sub_gpu_sp, cl::NullRange, cl::NDRange(m));

		clSetKernelArg(mat_vec_mul_gpu_sp, 0, sizeof(cl_mem), &buffer_A_t);
		clSetKernelArg(mat_vec_mul_gpu_sp, 1, sizeof(cl_mem), &buffer_res_x);
		clSetKernelArg(mat_vec_mul_gpu_sp, 2, sizeof(cl_mem), &buffer_res_b);
		clSetKernelArg(mat_vec_mul_gpu_sp, 3, sizeof(int), &n);
		clSetKernelArg(mat_vec_mul_gpu_sp, 4, sizeof(int), &m);
		err = clEnqueueNDRangeKernel(queue, mat_vec_mul_gpu_sp, 1, NULL, globalSize_n, NULL, 0, NULL, NULL);
		clEnqueueReadBuffer(queue, buffer_res_b, CL_TRUE, 0, sizeof(float) * res.size(), res.data(), 0, NULL, NULL);
		clFinish(queue);
		//mat_vec_mul_gpu_fp32.setArg(0, buffer_A_t);
		//mat_vec_mul_gpu_fp32.setArg(1, buffer_res_x);
		//mat_vec_mul_gpu_fp32.setArg(2, buffer_res_b);
		//mat_vec_mul_gpu_fp32.setArg(3, n);
		//mat_vec_mul_gpu_fp32.setArg(4, m);
		//queue.enqueueNDRangeKernel(mat_vec_mul_gpu_fp32, cl::NullRange, cl::NDRange(n));

		clEnqueueCopyBuffer(queue, buffer_sol, buffer_x, 0, 0, sizeof(T) * x1.size(), 0, NULL, NULL);

		//queue.enqueueCopyBuffer(buffer_sol, buffer_x, 0, 0, sizeof(T) * x1.size());

		clSetKernelArg(vec_sub_gpu_sp, 0, sizeof(cl_mem), &buffer_x);
		clSetKernelArg(vec_sub_gpu_sp, 1, sizeof(cl_mem), &buffer_res_b);
		err = clEnqueueNDRangeKernel(queue, vec_sub_gpu_sp, 1, NULL, globalSize_n, NULL, 0, NULL, NULL);

		//vec_sub_gpu_sp.setArg(0, buffer_x);
		//vec_sub_gpu_sp.setArg(1, buffer_res_b);
		//queue.enqueueNDRangeKernel(vec_sub_gpu_sp, cl::NullRange, cl::NDRange(n));
		f = (tau / beta);
		clSetKernelArg(shrink_gpu_sp, 0, sizeof(cl_mem), &buffer_x);
		clSetKernelArg(shrink_gpu_sp, 1, sizeof(T), &f);
		err = clEnqueueNDRangeKernel(queue, shrink_gpu_sp, 1, NULL, globalSize_n, NULL, 0, NULL, NULL);

		//shrink_gpu_sp.setArg(0, buffer_x);
		//shrink_gpu_sp.setArg(1, (tau / beta));
		//queue.enqueueNDRangeKernel(shrink_gpu_sp, cl::NullRange, cl::NDRange(n));

		clEnqueueCopyBuffer(queue, buffer_x, buffer_sol, 0, 0, sizeof(T) * n, 0, NULL, NULL);
		clEnqueueCopyBuffer(queue, buffer_x, buffer_aux1, 0, 0, sizeof(T) * x1.size(), 0, NULL, NULL);

		//queue.enqueueCopyBuffer(buffer_x, buffer_sol, 0, 0, sizeof(T) * n);
		//queue.enqueueCopyBuffer(buffer_x, buffer_aux1, 0, 0, sizeof(T) * x1.size());

		clSetKernelArg(mat_vec_mul_gpu_sp, 0, sizeof(cl_mem), &buffer_A);
		clSetKernelArg(mat_vec_mul_gpu_sp, 1, sizeof(cl_mem), &buffer_aux1);
		clSetKernelArg(mat_vec_mul_gpu_sp, 2, sizeof(cl_mem), &buffer_res_x);
		clSetKernelArg(mat_vec_mul_gpu_sp, 3, sizeof(int), &m);
		clSetKernelArg(mat_vec_mul_gpu_sp, 4, sizeof(int), &n);
		clEnqueueNDRangeKernel(queue, mat_vec_mul_gpu_sp, 1, NULL, globalSize_m, NULL, 0, NULL, NULL);

		//mat_vec_mul_gpu_fp32.setArg(0, buffer_A);
		//mat_vec_mul_gpu_fp32.setArg(1, buffer_aux1);
		//mat_vec_mul_gpu_fp32.setArg(2, buffer_res_x);
		//mat_vec_mul_gpu_fp32.setArg(3, m);
		//mat_vec_mul_gpu_fp32.setArg(4, n);
		//queue.enqueueNDRangeKernel(mat_vec_mul_gpu_fp32, cl::NullRange, cl::NDRange(m));

		clSetKernelArg(vec_sub_gpu_sp, 0, sizeof(cl_mem), &buffer_res_x);
		clSetKernelArg(vec_sub_gpu_sp, 1, sizeof(cl_mem), &buffer_b);
		clEnqueueNDRangeKernel(queue, vec_sub_gpu_sp, 1, NULL, globalSize_m, NULL, 0, NULL, NULL);

		//vec_sub_gpu_sp.setArg(0, buffer_res_x);
		//vec_sub_gpu_sp.setArg(1, buffer_b);
		//queue.enqueueNDRangeKernel(vec_sub_gpu_sp, cl::NullRange, cl::NDRange(m));
		f = (gamma * beta);
		clSetKernelArg(vec_scalar_gpu_sp, 0, sizeof(cl_mem), &buffer_res_x);
		clSetKernelArg(vec_scalar_gpu_sp, 1, sizeof(T), &f);
		clEnqueueNDRangeKernel(queue, vec_scalar_gpu_sp, 1, NULL, globalSize_m, NULL, 0, NULL, NULL);

		//vec_scalar_gpu_sp.setArg(0, buffer_res_x);
		//vec_scalar_gpu_sp.setArg(1, (gamma * beta));
		//queue.enqueueNDRangeKernel(vec_scalar_gpu_sp, cl::NullRange, cl::NDRange(m));

		clSetKernelArg(vec_sub_gpu_sp, 0, sizeof(cl_mem), &buffer_y);
		clSetKernelArg(vec_sub_gpu_sp, 1, sizeof(cl_mem), &buffer_res_x);
		err = clEnqueueNDRangeKernel(queue, vec_sub_gpu_sp, 1, NULL, globalSize_m, NULL, 0, NULL, NULL);

		//vec_sub_gpu_sp.setArg(0, buffer_y);
		//vec_sub_gpu_sp.setArg(1, buffer_res_x);
		//queue.enqueueNDRangeKernel(vec_sub_gpu_sp, cl::NullRange, cl::NDRange(m));

		clEnqueueCopyBuffer(queue, buffer_y, buffer_aux_y, 0, 0, sizeof(T) * m, 0, NULL, NULL);

		//queue.enqueueCopyBuffer(buffer_y, buffer_aux_y, 0, 0, sizeof(T) * m);
	}

	clEnqueueReadBuffer(queue, buffer_sol, CL_TRUE, 0, sizeof(float)* x1.size(), x1.data(), 0, NULL, NULL);
	clFinish(queue);

	clReleaseMemObject(buffer_aux1);
	clReleaseMemObject(buffer_x);
	clReleaseMemObject(buffer_sol);
	clReleaseMemObject(buffer_aux_y);
	clReleaseMemObject(buffer_y);
	clReleaseMemObject(buffer_res_x);
	clReleaseMemObject(buffer_res_b);
	clReleaseMemObject(buffer_b);
	//queue.enqueueReadBuffer(buffer_sol, CL_TRUE, 0, sizeof(float) * x1.size(), x1.data());
	//queue.finish();

	sol = x1;

	return sol;
}


template <class T>
Matrix<T> SparseRepSol<T>::solve_PALM(uint64_t iterations_outter_loop, uint64_t iterations_inner_loop, T zeta) {
	
	T t = 0.0f, L = 1.0f;
	Matrix<T> e_k(A.getRows()), x_k(A.getCols()), theta_k(A.getRows()), w(A.getCols()), w1(A.getCols()), z(A.getCols());
	e_k.fill(0.0f);
	x_k.fill(0.0f);
	theta_k.fill(0.0f);
	w.fill(0.0f);
	w1.fill(0.0f);
	z.fill(0.0f);
	Matrix<T> A_t = A.getTransposedMatrix();

	if (!is_max_eig_set) {
		max_eig = power_method();
		is_max_eig_set = true;
	}

	L = max_eig;

	for (uint64_t i = 0; i <= iterations_outter_loop; i++) {

		e_k = shrink((b - A * x_k + theta_k * (1 / zeta)), 1 / zeta);
		t = 1;
		z = x_k;
		w = x_k;

		for (uint64_t j = 0; j <= iterations_inner_loop; j++) {
			w1 = shrink(z + A_t * (1 / L) * (b - A * z - e_k + theta_k * (1 / zeta)), 1 / (zeta * L));
			t = 0.5f * (1 + sqrt(1 + 4 * t * t));
			z = w1 + (w1 - w) * ((t - 1) / (1 + 1));
		}
		x_k = w1;
		theta_k = theta_k + (b - A * x_k - e_k) * zeta;
	}

	return x_k;

}

template <class T>
SparseRepSol<T>::~SparseRepSol() {
}