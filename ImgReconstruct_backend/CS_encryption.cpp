#include "CS_encryption.hpp"

void CSencryption::returnRandomIndices(std::vector<int>& ri_x, std::vector<int>& ri_y, int xm, int ym, int numOfIndices, std::string password) {

    auto seeds = generate_seeds(password);
    std::mt19937 generator;

    std::uniform_int_distribution<> disx(0, xm - 1);
    std::uniform_int_distribution<> disy(0, ym - 1);

    bool** xy = new bool* [xm];

    for (int i = 0; i < xm; ++i) {
        xy[i] = new bool[ym];
        for (int j = 0; j < ym; ++j) {
            xy[i][j] = false;
        }
    }

    for (int i = 0; i < seeds.size(); ++i) {

        generator.seed(seeds[i]);

        int k = 0;

        for (; k < numOfIndices / seeds.size();) {
            int x = disx(generator);
            int y = disy(generator);

            if (xy[x][y] != true) {
                ri_x[i * (numOfIndices / seeds.size()) + k] = x;
                ri_y[i * (numOfIndices / seeds.size()) + k] = y;
                xy[x][y] = true;
                k++;
            }
        }
    }

    for (int i = 0; i < xm; ++i) {
        delete[] xy[i];
    }

    delete[] xy;
}