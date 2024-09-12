#include <condition_variable>
#include <mutex>
#include <vector>
#include <string>
#include "sparseImageEncryption.h"
#include <chrono>


void myFunction(const char* message) {}

int main() {

    StatusCallback myCallback;
    myCallback = myFunction;

    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
    encryptAndWriteFile(myCallback, "test_image.png", "test_image.se", "5v48v5832v5924", 64, 1, false);
    decryptAndWriteFile(myCallback, "test_image.se", "test_image_decrypted.png", "5v48v5832v5924", 1, 300, false);
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    double solve_time = std::chrono::duration_cast<std::chrono::minutes>(t2 - t1).count();
    std::cout << solve_time << std::endl;
    return 1;
}