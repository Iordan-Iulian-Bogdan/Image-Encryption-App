#include "CS_encryption.hpp"

void CSencryption::returnRandomIndices(std::vector<int>& ri_x, std::vector<int>& ri_y, int xm, int ym, int numOfIndices, std::string password, float pixel_p) {

    auto seeds = generate_seeds(password);
    std::mt19937 generator;
    generator.seed(seeds[seeds[0]%10]);
    std::uniform_real_distribution<> distribution_pixels(0, 1);
    std::uniform_int_distribution<> distribution_seeds(0, seeds.size() - 1);

    unsigned long picked_seed = distribution_seeds(generator);

    generator.seed(picked_seed);

    int k = 0;

    for (int i = 0; i < xm; ++i) {
        for (int j = 0; j < ym && k < numOfIndices; ++j) {
            if (distribution_pixels(generator) < pixel_p) {
                ri_x[k] = i;
                ri_y[k] = j;
                k++;
            }
        }
    }

    shuffle(ri_x, picked_seed);
    shuffle(ri_y, picked_seed);
}