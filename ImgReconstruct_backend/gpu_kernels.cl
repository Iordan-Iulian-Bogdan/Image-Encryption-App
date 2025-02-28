#define M_PI 3.141592653589793

__kernel void dctMatrix(__global float* dct, int n, int m) {
    int i = get_global_id(0);
    int j = get_global_id(1);

    if (i < n && j < m) {
        float alpha = (i == 0) ? sqrt(1.0 / n) : sqrt(2.0 / n);
        dct[i * m + j] = alpha * cos(M_PI * (2 * j + 1) * i / (2.0 * n));
    }
}