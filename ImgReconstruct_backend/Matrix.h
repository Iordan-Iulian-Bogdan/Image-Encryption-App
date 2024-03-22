#ifndef _MATRIX_H_
#define _MATRIX_H_

#define CL_USE_DEPRECATED_OPENCL_2_0_APIS
#include <CL/cl.hpp>
#include <iostream>
#include <vector>
#include <random>
#include <string>
#include <sstream>
#include <omp.h>
#include <chrono>

template<class T>
class Matrix
{
public:

    std::vector<T> data;
    size_t rows;
    size_t cols;

    Matrix(size_t rows, size_t cols = 1);
    Matrix(std::vector <T> data, size_t rows, size_t cols = 1);
    Matrix();
    // Copy constructor
    Matrix(const Matrix& other);
    // Getter functions
    size_t getRows() const;
    size_t getCols() const;
    // Indexing function
    T& at(size_t row, size_t col);
    T& at(size_t row);
    const T& at(size_t row, size_t col) const;
    const T& at(size_t row) const;
    // Overloaded assignment operator
    Matrix<T>& operator=(const Matrix& other);
    // Overloaded move assignment operator
    Matrix<T>& operator=(Matrix&& other) noexcept;
    // Overloaded addition operator
    Matrix<T> operator+(const Matrix& other) const;
    // Overloaded subtraction operator
    Matrix<T> operator-(const Matrix& other) const;
    // Overloaded multiplication operator (matrix multiplication)
    Matrix<T> operator*(const Matrix& other) const;
    // Overloaded multiplication operator (scalar multiplication)
    Matrix<T> operator*(T scalar) const;
    Matrix<T> getTransposedMatrix();
    //Calculate norm
    T norm();
    void transposeMatrix();
    // Function to display the matrix
    void display() const;
    void display_sparse() const;
    void fill_rand();
    void fill_randSparse(uint64_t sparsity);
    void fill_randSparseBinary(uint64_t sparsity);
    void fill(T a);
    void abs();
    void fillRandom(std::vector<T>& data, T lower_bound = 0, T upper_bound = 1);
    void fillRandomSparse(std::vector<T>& data, uint64_t sparsity, T lower_bound = -2, T upper_bound = 2);
    void fillRandomSparseBinary(std::vector<T>& data, uint64_t sparsity, T lower_bound = -2, T upper_bound = 2);
    void initialize_opencl_context();
    void update_gpu_buffer();

    cl::Context default_context;
    cl::Device default_device;
    cl::Buffer buffer_data;

};

#endif