// TestTemp.cpp
#include "Matrix.h"
#include <cmath>

template <class T>
Matrix<T>::Matrix(size_t rows, size_t cols) : rows(rows), cols(cols) {
    // Initialize the matrix with zeros
    data.resize(rows * cols, T());
    if (cols == 1)
    {

    }
    //std::cout << "Here";
}

template <class T>
Matrix<T>::Matrix(std::vector <T> data, size_t rows, size_t cols) : rows(rows), cols(cols), data(data) {
    // Initialize the matrix with zeros
    data.resize(rows * cols, T());
    if (cols == 1)
    {

    }
    //std::cout << "Here";
}


template <class T>
Matrix<T>::Matrix(const Matrix& other) : rows(other.rows), cols(other.cols), data(other.data) { std::cout << "Here"; }

template <class T>
Matrix<T>::Matrix() {}

template <class T>
size_t Matrix<T>::getRows() const { return rows; }

template <class T>
size_t Matrix<T>::getCols() const { return cols; }

template <class T>
// Indexing function
T& Matrix<T>::at(size_t row, size_t col) {
    if (row >= rows || col >= cols) {
        throw std::out_of_range("Index out of bounds");
    }
    return data[row * cols + col];
}

template <class T>
// Indexing function
T& Matrix<T>::at(size_t row) {
    if (row >= rows) {
        throw std::out_of_range("Index out of bounds");
    }
    return data[row++];
}

template <class T>
const T& Matrix<T>::at(size_t row, size_t col) const {
    if (row >= rows || col >= cols) {
        throw std::out_of_range("Index out of bounds");
    }
    return data[row * cols + col];
}

template <class T>
const T& Matrix<T>::at(size_t row) const {
    if (row >= rows) {
        throw std::out_of_range("Index out of bounds");
    }
    return data[row++];
}

template <class T>
// Overloaded assignment operator
Matrix<T>& Matrix<T>::operator=(const Matrix& other) {
    if (this != &other) {
        rows = other.rows;
        cols = other.cols;
        data = other.data;
    }
    return *this;
}

template <class T>
// Overloaded move assignment operator
Matrix<T>& Matrix<T>::operator=(Matrix&& other) noexcept {
    if (this != &other) {
        rows = other.rows;
        cols = other.cols;
        data = std::move(other.data);

        // Reset the other matrix
        other.rows = 0;
        other.cols = 0;
    }
    return *this;
}

template <class T>
// Overloaded addition operator
Matrix<T> Matrix<T>::operator+(const Matrix& other) const {
    if (rows != other.rows || cols != other.cols) {
        throw std::invalid_argument("Matrices must have the same dimensions for addition");
    }

    Matrix result(rows, cols);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result.at(i, j) = at(i, j) + other.at(i, j);
        }
    }
    return result;
}

template <class T>
// Overloaded subtraction operator
Matrix<T> Matrix<T>::operator-(const Matrix& other) const {
    if (rows != other.rows || cols != other.cols) {
        throw std::invalid_argument("Matrices must have the same dimensions for subtraction");
    }

    Matrix result(rows, cols);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result.at(i, j) = at(i, j) - other.at(i, j);
        }
    }
    return result;
}

template <class T>
// Overloaded multiplication operator (matrix multiplication)
Matrix<T> Matrix<T>::operator*(const Matrix& other) const {
    if (cols != other.rows) {
        throw std::invalid_argument("Number of columns in the first matrix must be equal to the number of rows in the second matrix for multiplication");
    }

    Matrix result(rows, other.cols);

    if (other.cols == 1) {
        multiplyMatrices(data, other.data, result.data, rows, cols, other.cols);
    }
    else {
        multiplyMatricesAVX512(data, other.data, result.data, rows, cols, other.cols);
        //multiplyMatrices(data, other.data, result.data, rows, cols, other.cols);
    }

    return result;
}

template <class T>
// Overloaded multiplication operator (scalar multiplication)
Matrix<T> Matrix<T>::operator*(T scalar) const {
    Matrix result(rows, cols);
    int numThreads = omp_get_max_threads();
    #pragma omp parallel for num_threads(numThreads) schedule(dynamic)
    for (int64_t i = 0; i < rows; ++i) {
        for (int64_t j = 0; j < cols; ++j) {
            result.at(i, j) = at(i, j) * scalar;
        }
    }
    return result;
}

template <class T>
Matrix<T> Matrix<T>::getTransposedMatrix() {
    // Check if the matrix is non-empty
    if (data.empty()) {
        throw std::invalid_argument("Matrix is empty.");
    }

    // Create a temporary vector to store the transposed matrix
    std::vector<T> transposedMatrix(data.size());
    int numThreads = omp_get_max_threads();
    #pragma omp parallel for num_threads(numThreads) schedule(dynamic)
    for (int64_t i = 0; i < rows; ++i) {
        for (int64_t j = 0; j < cols; ++j) {
            transposedMatrix[j * rows + i] = data[i * cols + j];
        }
    }

    Matrix<T> Mat_t(transposedMatrix, cols, rows);

    return Mat_t;
}

template <class T>
void Matrix<T>::transposeMatrix() {
    // Check if the matrix is non-empty
    if (data.empty()) {
        throw std::invalid_argument("Matrix is empty.");
    }

    // Create a temporary vector to store the transposed matrix
    std::vector<T> transposedMatrix(data.size());

    // Transpose the matrix
    int numThreads = omp_get_max_threads();
    #pragma omp parallel for num_threads(numThreads) schedule(dynamic)
    for (int64_t i = 0; i < rows; ++i) {
        for (int64_t j = 0; j < cols; ++j) {
            transposedMatrix[j * rows + i] = data[i * cols + j];
        }
    }

    // Copy the transposed matrix back to the original matrix
    data = transposedMatrix;
    std::swap(rows, cols);
}

template <class T>
// Function to display the matrix
void Matrix<T>::display() const {
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            std::cout << at(i, j) << " ";
        }
        std::cout << std::endl;
    }
}

template <class T>
// Function to display the matrix
void Matrix<T>::display_sparse() const {
    for (size_t i = 0; i < rows * cols; ++i) {
        if ((data[i] - 0.0f) >= 0.00001f) {
            std::cout << data[i] << " ";
            std::cout << std::endl;
        }
    }
}

template <class T>
void Matrix<T>::fill_rand() {
    fillRandom(data);
}

template <class T>
void Matrix<T>::fill_randSparse(const uint64_t sparsity) {
    fillRandomSparse(data, sparsity);
}

template <class T>
void Matrix<T>::fill_randSparseBinary(const uint64_t sparsity) {
    fillRandomSparseBinary(data, sparsity);
}

template <class T>
void Matrix<T>::fill(T a) {
    for (size_t i = 0; i < data.size(); i++)
        data[i] = a;
}

template <class T>
T Matrix<T>::norm() {
    T norm = 0.0f;

    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j)
        norm += at(i, j) * at(i, j);

    return (T)sqrt(norm);
}

template <class T>
void Matrix<T>::abs() {
    T sum = 0;
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            sum += at(i, j);
        }
    }
}

template <class T>
void Matrix<T>::initialize_opencl_context() {

}


template <class T>
void Matrix<T>::fillRandom(std::vector<T>& data, T lower_bound, T upper_bound) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<T> distribution(lower_bound, upper_bound);

    int numThreads = omp_get_max_threads();
    #pragma omp parallel for num_threads(numThreads) schedule(dynamic)
    for (int64_t i = 0; i < data.size(); ++i) {
        data[i] = distribution(gen);
    }
}

template <class T>
void Matrix<T>::fillRandomSparse(std::vector<T>& data, uint64_t sparsity, T lower_bound, T upper_bound) {
    static std::default_random_engine e;
    std::uniform_real_distribution<T> distribution(lower_bound, upper_bound);
    static std::uniform_int_distribution<> distribution_sparse(0, data.size() - 1);

    int numThreads = omp_get_max_threads();
    #pragma omp parallel for num_threads(numThreads) schedule(dynamic)
    for (int64_t i = 0; i < sparsity; ++i) {
        data[distribution_sparse(e)] = distribution(e);
    }
}

template <class T>
void Matrix<T>::fillRandomSparseBinary(std::vector<T>& data, uint64_t sparsity, T lower_bound, T upper_bound) {
    static std::default_random_engine e;
    std::uniform_real_distribution<T> distribution(lower_bound, upper_bound);
    static std::uniform_int_distribution<> distribution_sparse(0, data.size() - 1);

    int numThreads = omp_get_max_threads();
#pragma omp parallel for num_threads(numThreads) schedule(dynamic)
    for (int64_t i = 0; i < sparsity; ++i) {
        data[distribution_sparse(e)] = 1;
    }
}