#include "sparse_representations_solver.h"
#include "sparse_representations_solver.cpp"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgproc/imgproc.hpp>

void recostruct(cv::Mat& img, cv::Mat& out, float p) {

    int64_t width = img.size[1], height = img.size[0], n = width * height, m = n / p, k = 0;
    Matrix<float> x(height * width), theta(m, n), psi(m, n), y(m), e(n), e_idct(n), aux(m), sol(n), x_1(n);
    cv::Mat floatImg;
    cv::Mat reconstructedImg;
    img.convertTo(floatImg, CV_32FC1);
    img.convertTo(reconstructedImg, CV_32FC1);

    psi.fill_rand();


    for (int i = 0; i < floatImg.rows; i++){
        for (int j = 0; j < floatImg.cols; j++){

            x.at(k++) = floatImg.at<float>(j, i);
        }
    }

    y = psi * x;

    for (int64_t i = 0; i < n; i++) {

        e.fill(0.0f);
        e.at(i) = 1.0f;
        cv::idct(e.data, e_idct.data, 0);
        aux = psi * e_idct;

        for (int64_t j = 0; j < m; j++) {
            theta.data[j * n + i] = aux.at(j);
        }
    }

    SparseRepSol<float>data_reconstruct(theta, y);
    sol = data_reconstruct.solve_PALM(300, 3);
    //sol = data_reconstruct.solve_ADM(1000, 0.000001f, 0.000001f);

    for (int64_t i = 0; i < n; i++) {

        e.fill(0.0f);
        e.at(i) = 1.0f;
        cv::idct(e.data, e_idct.data, 0);
        x_1 = x_1 + e_idct * sol.at(i);
    }

    k = 0;

    for (int i = 0; i < floatImg.rows; i++)
    {
        for (int j = 0; j < floatImg.cols; j++)
        {
            reconstructedImg.at<float>(j, i) = x_1.at(k++);
        }
    }

    cv::Mat dst;
    reconstructedImg.convertTo(dst, CV_8U);

    out = dst;
}

int main() {

    cv::Mat img_in = cv::imread("C:\\images\\gpu.png", cv::IMREAD_GRAYSCALE);
    cv::Mat img_out;

    if (img_in.empty())
    {
        std::cout << "!!! Failed imread(): image not found" << std::endl;
        // don't let the execution continue, else imshow() will crash.
    }

    recostruct(img_in, img_out, 2);

    cv::imwrite("C:/images/gpu_out.png", img_out);
    imshow("Display window", img_out);

    return 0;
}