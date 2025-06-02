#include "image_decryption.hpp"
#include "image_encryption.hpp"

int main(int argc, char* argv)
{
    const std::string password = "5v48d254h33432";

    std::vector<const char*> inputs(4);
    inputs[0] = "IMG_3690.png";
    inputs[1] = "IMG_1297.png";
    inputs[2] = "IMG_0007.png";
    inputs[3] = "IMG_9321.png";

    auto start = std::chrono::high_resolution_clock::now();

    encrypt_image_tiled(inputs[0], "encrypted_img_g.png", "5v48d254h33432", 1.0f);
    decrypt_image_tiled("encrypted_img_g.png", "decrypted_image.png", "5v48d254h33432", AUTO_PARAM);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    std::cout << "Execution time: " << duration << " milliseconds" << std::endl;

    return 0;
}