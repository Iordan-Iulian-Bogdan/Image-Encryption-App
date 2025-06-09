#ifndef CS_ENCRYPTION_HPP
#define CS_ENCRYPTION_HPP


#include "helper_functions.hpp"

class CSencryption {
protected:
    cv::Mat encrypted_img, input_img;
    cv::Mat decrypted_img;
    cv::Size org_size;
    int rows, cols, m, n;
    std::vector<int> ri_x, ri_y;
   
    void returnRandomIndices(std::vector<int>& ri_x, std::vector<int>& ri_y, int xm, int ym, int numOfIndices, std::string password, float pixel_p);

public:
    static int params;
};

#endif