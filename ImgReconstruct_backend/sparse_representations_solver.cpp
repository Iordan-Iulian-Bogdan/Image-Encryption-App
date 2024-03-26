#define CL_USE_DEPRECATED_OPENCL_2_0_APIS
#include <CL/cl.hpp>
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
SparseRepSol<T>::SparseRepSol(Matrix<T>& A, Matrix<T>& b) :A(A), b(b) {
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

		std::tuple <cl::Context, cl::CommandQueue, cl::Program> cl_context = creat_opencl_context();

		Matrix<float> X(A.rows, A.rows);


		cl::Buffer buffer_A(std::get<0>(cl_context), CL_MEM_READ_WRITE, sizeof(T) * A.data.size());
		cl::Buffer buffer_A_t(std::get<0>(cl_context), CL_MEM_READ_WRITE, sizeof(T) * A_t.data.size());
		cl::Buffer buffer_X(std::get<0>(cl_context), CL_MEM_READ_WRITE, sizeof(T) * X.data.size());


		std::get<1>(cl_context).enqueueWriteBuffer(buffer_A, CL_TRUE, 0, sizeof(T) * A.data.size(), A.data.data());
		std::get<1>(cl_context).enqueueWriteBuffer(buffer_A_t, CL_TRUE, 0, sizeof(T) * A_t.data.size(), A_t.data.data());
		std::get<1>(cl_context).enqueueWriteBuffer(buffer_X, CL_TRUE, 0, sizeof(T) * X.data.size(), X.data.data());

		mat_mat_mul_gpu(cl_context, buffer_A_t, buffer_A, buffer_X, (int)A.rows, (int)A.cols);
		std::get<1>(cl_context).enqueueReadBuffer(buffer_X, CL_TRUE, 0, sizeof(T) * X.data.size(), X.data.data());

		std::get<1>(cl_context).finish();


		Matrix<T> b_k(A.getRows()), aux(A.getRows()), b_k1(A.getRows());
		T norm_b_k1 = 0.0f, eig = 0.0f;

		b_k.fill(1.0f);

		cl::Buffer buffer_b_k(std::get<0>(cl_context), CL_MEM_READ_WRITE, sizeof(T) * b_k.data.size());
		cl::Buffer buffer_b_k_res(std::get<0>(cl_context), CL_MEM_READ_WRITE, sizeof(T) * b_k.data.size());
		std::get<1>(cl_context).finish();

		for (unsigned int i = 0; i < 10; i++)
		{
			std::get<1>(cl_context).enqueueWriteBuffer(buffer_b_k, CL_TRUE, 0, sizeof(T) * b_k.data.size(), b_k.data.data());
			mat_vec_mul_gpu(cl_context, buffer_X, buffer_b_k, buffer_b_k_res, (int)A.rows, (int)A.rows);
			std::get<1>(cl_context).enqueueReadBuffer(buffer_b_k_res, CL_TRUE, 0, sizeof(float) * b_k.data.size(), b_k.data.data());

			b_k1 = b_k;
			norm_b_k1 = b_k1.norm();
			aux = b_k1;
			b_k1 = b_k1 * (1 / norm_b_k1);
			b_k = b_k1;
			b_k1 = aux;
		}

		aux = b_k;

		std::get<1>(cl_context).enqueueWriteBuffer(buffer_b_k, CL_TRUE, 0, sizeof(T) * b_k.data.size(), b_k.data.data());
		mat_vec_mul_gpu(cl_context, buffer_X, buffer_b_k, buffer_b_k_res, (int)A.rows, (int)A.rows);
		std::get<1>(cl_context).enqueueReadBuffer(buffer_b_k_res, CL_TRUE, 0, sizeof(float) * b_k.data.size(), b_k.data.data());

		b_k = b_k.getTransposedMatrix() * aux;
		aux = aux.getTransposedMatrix() * aux;

		return  b_k.at(0, 0) / aux.at(0, 0);
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
	if (!is_max_eig_set)
	{
		max_eig = power_method_gpu();
		//max_eig = 274;
	}
	int n = A.cols;
	int m = A.rows;
	//	gamma is required to be less than 2 in order to ensure convergence
	//	note that if the maximum singular value are too big this can no longer be true
	//	and you'd need to change tau and beta
	float gamma = 1.99f - (tau * max_eig);

	std::vector<T> aux1(n), aux2(n), aux3(n), aux_b(m), aux_x(n), b1(m), x1(n);
	std::vector<T> y(m);
	std::vector<T> r(m);
	A_t = A_t * tau;
	b1 = b.data;

	//	Implemeting the following ADM algorithm :
	//	Input: τ, β, γ dictionary A, measurement b, x = 0, y = 0
	//	While not converge
	//	x(k)←shrink(x(k)-τA*(Ax(k)-b-y(k)/β),τ/β)
	//	y(k+1)←y(k)-γβ(Ax(k+1)-b)
	//	end while
	//	Output: x(k)

	std::vector<cl::Platform> all_platforms;
	cl::Platform::get(&all_platforms);
	cl::Platform default_platform = all_platforms[0];
	std::vector<cl::Device> all_devices;
	default_platform.getDevices(CL_DEVICE_TYPE_GPU, &all_devices);
	cl::Device default_device = all_devices[0];
	//std::cout << all_devices[0].getInfo<CL_DEVICE_NAME>() << std::endl;
	cl::Context context(default_device);
	std::ifstream src("gpu_kernels.cl");
	std::string str((std::istreambuf_iterator<char>(src)), std::istreambuf_iterator<char>());
	cl::Program::Sources sources;
	sources.push_back({ str.c_str(),str.length() });
	cl::Program program(context, sources);
	program.build({ default_device });
	cl::CommandQueue queue(context, default_device);
	cl::Kernel mat_vec_mul_gpu_fp32;
	cl::Kernel vec_sub_gpu_sp;
	cl::Kernel vec_scalar_gpu_sp;
	cl::Kernel shrink_gpu_sp;
	mat_vec_mul_gpu_fp32 = cl::Kernel(program, "mat_vec_mul_gpu_fp32");
	vec_sub_gpu_sp = cl::Kernel(program, "vec_sub_gpu_sp");
	vec_scalar_gpu_sp = cl::Kernel(program, "vec_scalar_gpu_sp");
	shrink_gpu_sp = cl::Kernel(program, "shrink_gpu_sp");

	std::vector<float> A_flat(n * m);
	std::vector<float> A_t_flat(n * m);
	std::vector<float> aux1_flat(n), res_flat(m);

	A_flat = A.data;
	A_t_flat = A_t.data;

	A.data = std::vector<T>();
	A_t.data = std::vector<T>();

	cl::Buffer buffer_A(context, CL_MEM_READ_ONLY, sizeof(float) * A_flat.size());
	cl::Buffer buffer_A_t(context, CL_MEM_READ_ONLY, sizeof(float) * A_t_flat.size());
	cl::Buffer buffer_aux1(context, CL_MEM_READ_WRITE, sizeof(float) * aux1_flat.size());
	cl::Buffer buffer_x(context, CL_MEM_READ_WRITE, sizeof(float) * n);
	cl::Buffer buffer_sol(context, CL_MEM_READ_WRITE, sizeof(float) * n);
	cl::Buffer buffer_aux_y(context, CL_MEM_READ_WRITE, sizeof(float) * m);
	cl::Buffer buffer_y(context, CL_MEM_READ_WRITE, sizeof(float) * m);
	cl::Buffer buffer_res_x(context, CL_MEM_READ_WRITE, sizeof(float) * m);
	cl::Buffer buffer_res_b(context, CL_MEM_READ_WRITE, sizeof(float) * n);
	cl::Buffer buffer_b(context, CL_MEM_READ_WRITE, sizeof(float) * m);
	queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, sizeof(float) * A_flat.size(), A_flat.data());
	queue.enqueueWriteBuffer(buffer_A_t, CL_TRUE, 0, sizeof(float) * A_t_flat.size(), A_t_flat.data());
	queue.enqueueWriteBuffer(buffer_b, CL_TRUE, 0, sizeof(float) * b1.size(), b1.data());

	A_flat = std::vector<T>();
	A_t_flat = std::vector<T>();

	queue.finish();

	for (int i = 0; i <= iterations; i++)
	{
		mat_vec_mul_gpu_fp32.setArg(0, buffer_A);
		mat_vec_mul_gpu_fp32.setArg(1, buffer_aux1);
		mat_vec_mul_gpu_fp32.setArg(2, buffer_res_x);
		mat_vec_mul_gpu_fp32.setArg(3, m);
		mat_vec_mul_gpu_fp32.setArg(4, n);
		queue.enqueueNDRangeKernel(mat_vec_mul_gpu_fp32, cl::NullRange, cl::NDRange(m));
		std::vector<T> auxt(n);
		queue.enqueueReadBuffer(buffer_sol, CL_TRUE, 0, sizeof(float) * auxt.size(), auxt.data());

		vec_sub_gpu_sp.setArg(0, buffer_res_x);
		vec_sub_gpu_sp.setArg(1, buffer_b);
		queue.enqueueNDRangeKernel(vec_sub_gpu_sp, cl::NullRange, cl::NDRange(m));
		std::vector<T> auxb(m);
		queue.enqueueReadBuffer(buffer_res_x, CL_TRUE, 0, sizeof(float) * auxb.size(), auxb.data());

		vec_scalar_gpu_sp.setArg(0, buffer_aux_y);
		vec_scalar_gpu_sp.setArg(1, (1 / beta));
		queue.enqueueNDRangeKernel(vec_scalar_gpu_sp, cl::NullRange, cl::NDRange(m));

		vec_sub_gpu_sp.setArg(0, buffer_res_x);
		vec_sub_gpu_sp.setArg(1, buffer_aux_y);
		queue.enqueueNDRangeKernel(vec_sub_gpu_sp, cl::NullRange, cl::NDRange(m));

		mat_vec_mul_gpu_fp32.setArg(0, buffer_A_t);
		mat_vec_mul_gpu_fp32.setArg(1, buffer_res_x);
		mat_vec_mul_gpu_fp32.setArg(2, buffer_res_b);
		mat_vec_mul_gpu_fp32.setArg(3, n);
		mat_vec_mul_gpu_fp32.setArg(4, m);
		queue.enqueueNDRangeKernel(mat_vec_mul_gpu_fp32, cl::NullRange, cl::NDRange(n));

		queue.enqueueCopyBuffer(buffer_sol, buffer_x, 0, 0, sizeof(T) * x1.size());

		vec_sub_gpu_sp.setArg(0, buffer_x);
		vec_sub_gpu_sp.setArg(1, buffer_res_b);
		queue.enqueueNDRangeKernel(vec_sub_gpu_sp, cl::NullRange, cl::NDRange(n));

		shrink_gpu_sp.setArg(0, buffer_x);
		shrink_gpu_sp.setArg(1, (tau / beta));
		queue.enqueueNDRangeKernel(shrink_gpu_sp, cl::NullRange, cl::NDRange(n));

		queue.enqueueCopyBuffer(buffer_x, buffer_sol, 0, 0, sizeof(T) * n);
		queue.enqueueCopyBuffer(buffer_x, buffer_aux1, 0, 0, sizeof(T) * x1.size());

		mat_vec_mul_gpu_fp32.setArg(0, buffer_A);
		mat_vec_mul_gpu_fp32.setArg(1, buffer_aux1);
		mat_vec_mul_gpu_fp32.setArg(2, buffer_res_x);
		mat_vec_mul_gpu_fp32.setArg(3, m);
		mat_vec_mul_gpu_fp32.setArg(4, n);
		queue.enqueueNDRangeKernel(mat_vec_mul_gpu_fp32, cl::NullRange, cl::NDRange(m));

		vec_sub_gpu_sp.setArg(0, buffer_res_x);
		vec_sub_gpu_sp.setArg(1, buffer_b);
		queue.enqueueNDRangeKernel(vec_sub_gpu_sp, cl::NullRange, cl::NDRange(m));

		vec_scalar_gpu_sp.setArg(0, buffer_res_x);
		vec_scalar_gpu_sp.setArg(1, (gamma * beta));
		queue.enqueueNDRangeKernel(vec_scalar_gpu_sp, cl::NullRange, cl::NDRange(m));

		vec_sub_gpu_sp.setArg(0, buffer_y);
		vec_sub_gpu_sp.setArg(1, buffer_res_x);
		queue.enqueueNDRangeKernel(vec_sub_gpu_sp, cl::NullRange, cl::NDRange(m));

		queue.enqueueCopyBuffer(buffer_y, buffer_aux_y, 0, 0, sizeof(T) * m);
	}

	queue.enqueueReadBuffer(buffer_sol, CL_TRUE, 0, sizeof(float) * x1.size(), x1.data());
	queue.finish();

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