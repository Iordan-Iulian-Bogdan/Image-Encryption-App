#include "sparse_representations_solver.h"
#include <math.h>

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
Matrix<T> SparseRepSol<T>::solve_ADM(uint64_t iterations, T tau, T beta) {

	if (!is_max_eig_set) {
		max_eig = power_method();
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

	return x_k; 
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