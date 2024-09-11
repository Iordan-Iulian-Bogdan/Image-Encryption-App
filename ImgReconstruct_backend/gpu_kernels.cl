#define TS 32    
#define WPT 8   
#define RTS (TS/WPT)

__kernel void mat_vec_mul_gpu_fp32(
    __global float* matrix,
    __global float* vector,
    __global float* result,
    int rows,
    int cols
) {

    // Each work-item calculates one element of the result vector
    int row = get_global_id(0);

    float sum1 = 0.0f;
    float sum2 = 0.0f;

    // Iterate over matrix columns in pairs
    for (int col = 0; col < cols; col += 2) {
        // Perform the multiply-accumulate operations
        sum1 = fma(matrix[row * cols + col], vector[col], sum1); // Fused multiply-add
        sum2 = fma(matrix[row * cols + col + 1], vector[col + 1], sum2); // Fused multiply-add
    }

    // If the number of columns is odd, process the last column separately
    if (cols % 2 != 0) {
        float mat_val = matrix[row * cols + cols - 1];
        float vec_val = vector[cols - 1];
        sum1 = fma(mat_val, vec_val, sum1);
    }

    // Combine the two sums
    result[row] = sum1 + sum2;
}

__kernel void kernel vec_scalar_gpu_sp(__global float* x, float a){
    x[get_global_id(0)] *= a;
}

__kernel void kernel shrink_gpu_sp(__global float* x, float threshold)
{
    const int id = get_global_id(0);

    local float aux;
    aux = sign(x[id]) * x[id] - threshold;

    x[id] = sign(x[id]) * fmax(aux, 0.0f);

    if ((sign(x[id]) * x[id]) < 1.175e-38)
        x[id] = 0.0f;
}

__kernel void kernel vec_sub_gpu_sp(__global float* vec1, __global const float* vec2)
{
    const int id = get_global_id(0);
    vec1[id] = vec1[id] - vec2[id];
}

__kernel void norm_sp(__global float* x, __global float* norm, int n)
{
    int i = 0;

    for (i = 0; i < n; i++)
    {
        norm[0] += x[i] * x[i];
    }

    norm[0] = sqrt(norm[0]);
}

__kernel void mat_mat_mul_gpu_sp(const int M, const int K,
    const __global float* A,
    const __global float* B,
    __global float* C) {

    // Thread identifiers
    const int row = get_local_id(0); // Local row ID (max: TS)
    const int col = get_local_id(1); // Local col ID (max: TS/WPT == RTS)
    const int globalRow = TS * get_group_id(0) + row; // Row ID of C (0..M)
    const int globalCol = TS * get_group_id(1) + col; // Col ID of C (0..N)

    // Local memory to fit a tile of TS*TS elements of A and B
    __local float Asub[TS][TS];
    __local float Bsub[TS][TS];

    // Initialise the accumulation registers
    float acc[WPT];
    for (int w = 0; w < WPT; w++) {
        acc[w] = 0.0f;
    }

    // Loop over all tiles
    const int numTiles = K / TS;
    for (int t = 0; t < numTiles; t++) {

        // Load one tile of A and B into local memory
        for (int w = 0; w < WPT; w++) {
            const int tiledRow = TS * t + row;
            const int tiledCol = TS * t + col;
            Asub[col + w * RTS][row] = A[(tiledCol + w * RTS) * M + globalRow];
            Bsub[col + w * RTS][row] = B[(globalCol + w * RTS) * K + tiledRow];
        }

        // Synchronise to make sure the tile is loaded
        barrier(CLK_LOCAL_MEM_FENCE);

        // Perform the computation for a single tile
        for (int k = 0; k < TS; k++) {
            for (int w = 0; w < WPT; w++) {
                acc[w] += Asub[k][row] * Bsub[col + w * RTS][k];
            }
        }

        // Synchronise before loading the next tile
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Store the final results in C
    for (int w = 0; w < WPT; w++) {
        C[(globalCol + w * RTS) * M + globalRow] = acc[w];
    }
}

__kernel void transpose(__global float* input,
    __global float* output,
    const int width,
    const int height) {
    int global_row = get_global_id(0);
    int global_col = get_global_id(1);

    if (global_row < height && global_col < width) {
        int index_in = global_row * width + global_col;
        int index_out = global_col * height + global_row;
        output[index_out] = input[index_in];
    }
}