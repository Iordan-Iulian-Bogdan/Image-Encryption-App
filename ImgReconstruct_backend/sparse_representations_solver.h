#ifndef _SPARSE_REPRESENTATIONS_SOLVER_H_
#define _SPARSE_REPRESENTATIONS_SOLVER_H_

#include "Matrix.cpp"
#include "Matrix.h"
#include <chrono>

using namespace std::chrono;

template<class T>
class SparseRepSol
{
public:
	T max_eig, error, solve_time = 0.0f;
	Matrix<T> A, A_t, b, x;
	std::vector<T> sol;
	bool is_max_eig_set = false;

	SparseRepSol(Matrix<T>& A, Matrix<T>& b);
	Matrix<T> getDictionary() const;
	Matrix<T> getMeasurement() const;
	Matrix<T> shrink(Matrix<T> M, T threshold);
	
	T power_method();
	T power_method_gpu();
	std::vector<T> solve_ADM(uint64_t iterations, T tau, T beta);
	std::vector<T> solve_ADM_gpu(uint64_t iterations, T tau, T beta);
	Matrix<T> solve_PALM(uint64_t iterations_outter_loop, uint64_t iterations_inner_loop, T zeta = 0.001f);
};

#endif