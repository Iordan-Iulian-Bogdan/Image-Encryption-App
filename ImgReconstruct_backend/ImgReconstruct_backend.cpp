#include <iostream>
#include <thread>
#include <condition_variable>
#include <mutex>
#include <vector>
#include <string>
#include "sparseImageEncryption.h"

int main() {

    cv::Mat img = cv::imread("test_image.png", cv::IMREAD_COLOR);
    uint32_t TILE_SIZE = 64;

    if (img.empty())
    {
        std::cout << "!!! Failed imread(): image not found" << std::endl;
    }

    encryptionImage img_encrypted = encryptImage(img, TILE_SIZE, "5v48v5832v5924", HYBRID_ACCELERATION);
    cv::Mat out = decryptImage(img_encrypted, "5v48v5832v5924", HYBRID_ACCELERATION, 12, 300);
    imshow("Display window", out);
    cv::imwrite("test_image_decrypted.png", out);

    return 1;
}
