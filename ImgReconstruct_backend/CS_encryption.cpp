#include "CS_encryption.hpp"

void CSencryption::returnRandomIndices(std::vector<int>& ri_x, std::vector<int>& ri_y, int xm, int ym, int numOfIndices, int seed) {
    std::random_device e;
    std::default_random_engine generator(e());
    generator.seed(seed);
    std::uniform_int_distribution<> disx(0, xm - 1);
    std::uniform_int_distribution<> disy(0, ym - 1);

    cv::Mat xy(xm, ym, CV_8U, cv::Scalar(0));

    int k = 0;

    while (k < numOfIndices) {
        int x = disx(generator);
        int y = disy(generator);

        if (xy.at<unsigned char>(x, y) == 1) {
            continue;
        }
        else {
            ri_x[k] = x;
            ri_y[k] = y;
            xy.at<unsigned char>(x, y) = 1;
            k++;
        }
    }
}