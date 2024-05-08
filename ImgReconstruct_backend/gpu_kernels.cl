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

    //printf("mat_vec_mul_gpu_fp32");
    // Get the global ID
    int globalId = get_global_id(0);

    // Ensure that the global ID is within the bounds of the result vector
    if (globalId < rows) {
        // Initialize the result for the current row
        float sum = 0.0f;

        // Perform the matrix-vector multiplication
        for (int i = 0; i < cols; ++i) {
            sum += matrix[globalId * cols + i] * vector[i];
        }

        // Store the result in the output vector
        result[globalId] = sum;
        //printf("%f", result[globalId]);
        //result[globalId] = 128.0f;

        //if (get_global_id(0) == 0)
            //printf("mat mul");
    }
}

__kernel void vectorAdd(__global const float* A, __global const float* B, __global float* C, const int length) {
    int i = get_global_id(0);
    if (i < length) {
        C[i] = A[i] + B[i];
    }
}

__kernel void vectorSub(__global const float* A, __global const float* B, __global float* C, const int length) {
    int i = get_global_id(0);
    if (i < length) {
        C[i] = A[i] - B[i];
    }
}

__kernel void kernel vec_scalar_gpu_sp_device(__global global float* x, float a, int size)
{
    for (int i = 0; i < size; ++i) {
        x[i] *= a;
    }
}

__kernel void kernel vec_scalar_gpu_sp(__global global float* x, float a)
{
    x[get_global_id(0)] *= a;
    //if (get_global_id(0) == 0)
       // printf("vec scalar");
}

__kernel void kernel shrink_gpu_sp(__global global float* x, float threshold)
{

    const int id = get_global_id(0);

    local float aux;
    aux = sign(x[id]) * x[id] - threshold;

    x[id] = sign(x[id]) * fmax(aux, 0.0f);

    if ((sign(x[id]) * x[id]) < 1.175e-38)
        x[id] = 0.0f;
    //if (get_global_id(0) == 0)
        //printf("shrink");

}

__kernel void kernel shrink_gpu_sp_device(__global global float* x, float threshold, int size)
{
    for (int i = 0; i < size; ++i) {
        int id = i;

        float aux;
        aux = sign(x[id]) * x[id] - threshold;

        x[id] = sign(x[id]) * fmax(aux, 0.0f);

        if ((sign(x[id]) * x[id]) < 1.175e-38)
            x[id] = 0.0f;
    }
}

__kernel void kernel vec_sub_gpu_sp(__global global float* vec1, __global global const float* vec2)
{
    const int id = get_global_id(0);
    vec1[id] = vec1[id] - vec2[id];
    //if (get_global_id(0) == 0)
        //printf("vec sub");
}

__kernel void kernel vec_sub_gpu_sp_device(__global float* vec1, __global float* vec2, int size)
{
    for (int i = 0; i < size; ++i) {
        vec1[i] = vec1[i] - vec2[i];
    }
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

__kernel void matrixMultiplication(__global float* A,
                                   __global float* B,
                                   __global float* C,
                                   int rowsA,
                                   int colsA,
                                   int colsB) {
    int globalRow = get_global_id(0);
    int globalCol = get_global_id(1);

    float sum = 0.0f;
    for (int k = 0; k < colsA; ++k) {
        sum += A[globalRow * colsA + k] * B[k * colsB + globalCol];
    }

    C[globalRow * colsB + globalCol] = sum;
}

float dot_product_device(const float* A, const float* B, int size) {
    float result = 0.0;
    for (int i = 0; i < size; i++) {
        result += A[i] * B[i];
    }
    return result;
}

float norm_sp_device(float* x, int n)
{
    int i = 0;
    float norm = 0.0f;

    for (i = 0; i < n; i++)
    {
        norm += x[i] * x[i];
    }

    return sqrt(norm);
}

__kernel void copy_gpu(__global float* dest, __global float* src, int n)
{
    for (unsigned int k = 0; k < n; k++)
        dest[k] = src[k];
}

__kernel void helper_func1(__global float* b_k, __global float* b_k_res, __global float* b_k1, __global float* aux, int rows)
{
    float norm_b_k1 = 0.0f;

    copy_gpu(b_k, b_k_res, rows);
    copy_gpu(b_k1, b_k, rows);

    norm_b_k1 = norm_sp_device(b_k1, rows);

    copy_gpu(aux, b_k1, rows);

    for (unsigned int k = 0; k < rows; k++)
        b_k1[k] = b_k1[k] * (1 / norm_b_k1);

    copy_gpu(b_k, b_k1, rows);
    copy_gpu(b_k1, aux, rows);
    copy_gpu(aux, b_k, rows);
}

__kernel void helper_func2(__global float* X, __global float* b_k, __global float* b_k_res, __global float* aux, __global float* b_k1, __global float* max_eig, int rows)
{
    copy_gpu(b_k, b_k_res, rows);

    float res1 = dot_product_device(b_k, aux, rows);
    float res2 = dot_product_device(aux, aux, rows);

    max_eig[0] = res1 / res2;
    printf("%f", res1);
}

__kernel void power_method( __global float* X, 
                            __global float* b_k, 
                            __global float* b_k_res, 
                            __global float* b_k1, 
                            __global float* aux,  
                            __global float* max_eig, 
                            int rows, 
                            int cols)
{

    clk_event_t event_mat_vec_mul, event_helper_func1;
    clk_event_t marker_event;

    for (unsigned int k = 0; k < rows; k++)
        b_k[k] = 1.0f;

    for (unsigned int i = 0; i < 10; i++)
    {
        enqueue_kernel(get_default_queue(), CLK_ENQUEUE_FLAGS_NO_WAIT, ndrange_1D(rows, 1), 0, NULL, &event_mat_vec_mul, ^{ mat_vec_mul_gpu_fp32(X, b_k, b_k_res, rows, cols); });
        enqueue_marker(get_default_queue(), 1, &event_mat_vec_mul, &marker_event);
        enqueue_kernel(get_default_queue(), CLK_ENQUEUE_FLAGS_NO_WAIT, ndrange_1D(1, 1), 1, &marker_event, NULL, ^{ helper_func1(b_k, b_k_res, b_k1, aux, rows); });
    }

    enqueue_kernel(get_default_queue(), CLK_ENQUEUE_FLAGS_NO_WAIT, ndrange_1D(rows, 1), 0, &marker_event, NULL, ^{ mat_vec_mul_gpu_fp32(X, b_k, b_k_res, rows, cols); });
    enqueue_marker(get_default_queue(), 1, &event_mat_vec_mul, &marker_event);
    enqueue_kernel(get_default_queue(), CLK_ENQUEUE_FLAGS_NO_WAIT, ndrange_1D(1, 1), 1, &marker_event, NULL, ^{ helper_func2(X, b_k1, b_k_res, aux, b_k1, max_eig, rows); });
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

__kernel void ADM_helper_func1(__global float* buffer_res_x, __global float* buffer_b, __global float* buffer_aux_y, float beta, int rows){
    int m = rows;
    clk_event_t event_1, event_2;

    enqueue_kernel(get_default_queue(), CLK_ENQUEUE_FLAGS_NO_WAIT, ndrange_1D(m, 1), 0, NULL, &event_1, ^ { vec_sub_gpu_sp(buffer_res_x, buffer_b); });
    enqueue_kernel(get_default_queue(), CLK_ENQUEUE_FLAGS_NO_WAIT, ndrange_1D(m, 1), 1, &event_1, &event_2, ^ { vec_scalar_gpu_sp(buffer_aux_y, (1 / beta)); });
    enqueue_kernel(get_default_queue(), CLK_ENQUEUE_FLAGS_NO_WAIT, ndrange_1D(m, 1), 1, &event_2, NULL, ^ { vec_sub_gpu_sp(buffer_res_x, buffer_aux_y); });

    printf("%f", buffer_b[0]);
    printf("%f", buffer_b[1]);
    printf("%f", buffer_b[2]);
    printf("%f", buffer_b[3]);

    //vec_sub_gpu_sp_device(buffer_res_x, buffer_b, m);
    //vec_scalar_gpu_sp_device(buffer_aux_y, (1 / beta), m);
    //vec_sub_gpu_sp_device(buffer_res_x, buffer_aux_y, m);
}

__kernel void ADM_helper_func2(__global float* buffer_x, __global float* buffer_sol, __global float* buffer_res_b, __global float* buffer_aux1, float beta, float tau, int cols){
    int n = cols;
    clk_event_t event_1, event_2;

    enqueue_kernel(get_default_queue(), CLK_ENQUEUE_FLAGS_NO_WAIT, ndrange_1D(1, 1), 1, &event_2, &event_1, ^ { copy_gpu(buffer_x, buffer_sol, n); });
    enqueue_kernel(get_default_queue(), CLK_ENQUEUE_FLAGS_NO_WAIT, ndrange_1D(n, 1), 1, &event_1, &event_2, ^ { vec_sub_gpu_sp(buffer_x, buffer_res_b); });
    enqueue_kernel(get_default_queue(), CLK_ENQUEUE_FLAGS_NO_WAIT, ndrange_1D(n, 1), 1, &event_2, &event_1, ^ { shrink_gpu_sp(buffer_x, (tau / beta)); });
    enqueue_kernel(get_default_queue(), CLK_ENQUEUE_FLAGS_NO_WAIT, ndrange_1D(1, 1), 1, &event_1, &event_2, ^ { copy_gpu(buffer_sol, buffer_x, n); });
    enqueue_kernel(get_default_queue(), CLK_ENQUEUE_FLAGS_NO_WAIT, ndrange_1D(1, 1), 1, &event_2, &event_1, ^ { copy_gpu(buffer_aux1, buffer_x, n); });

    copy_gpu(buffer_x, buffer_sol, n);
    vec_sub_gpu_sp_device(buffer_x, buffer_res_b, n);
    shrink_gpu_sp_device(buffer_x, (tau / beta), n);
    copy_gpu(buffer_sol, buffer_x, n);
    copy_gpu(buffer_aux1, buffer_x, n);
}

__kernel void ADM_helper_func3(__global float* buffer_res_x, __global float* buffer_y, __global float* buffer_aux_y, __global float* buffer_b, float beta, float gamma, int rows) {
    int m = rows;
    clk_event_t event_1, event_2;

    enqueue_kernel(get_default_queue(), CLK_ENQUEUE_FLAGS_NO_WAIT, ndrange_1D(m, 1), 1, &event_1, &event_2, ^ { vec_sub_gpu_sp(buffer_res_x, buffer_b); });
    enqueue_kernel(get_default_queue(), CLK_ENQUEUE_FLAGS_NO_WAIT, ndrange_1D(m, 1), 1, &event_2, &event_1, ^ { vec_scalar_gpu_sp(buffer_res_x, (gamma * beta)); });
    enqueue_kernel(get_default_queue(), CLK_ENQUEUE_FLAGS_NO_WAIT, ndrange_1D(m, 1), 1, &event_1, &event_2, ^ { vec_sub_gpu_sp(buffer_y, buffer_res_x); });
    enqueue_kernel(get_default_queue(), CLK_ENQUEUE_FLAGS_NO_WAIT, ndrange_1D(1, 1), 1, &event_2, &event_1, ^ { copy_gpu(buffer_aux_y, buffer_y, m); });

    vec_sub_gpu_sp_device(buffer_res_x, buffer_b, m);
    vec_scalar_gpu_sp_device(buffer_res_x, (gamma * beta), m);
    vec_sub_gpu_sp_device(buffer_y, buffer_res_x, m);
    copy_gpu(buffer_aux_y, buffer_y, m);
}

__kernel void sync(__global int* buffer_count)
{
    //buffer_count[0]++;
    //printf("sync %d", buffer_count[0]);
}

__kernel void ADM(__global float* buffer_A, __global float* buffer_A_t, __global float* buffer_b,
                  __global float* buffer_res, __global float* buffer_y, __global float* buffer_x, __global float* buffer_res2, __global int* buffer_count,
                  float max_eig, float beta, float tau, float gamma,
                  int rows, int cols, int iterations) {
    
    int m = rows;
    int n = cols;
    int err = 0;

    clk_event_t event_1, event_2;
    
    enqueue_kernel(get_default_queue(), CLK_ENQUEUE_FLAGS_NO_WAIT, ndrange_1D(m), 0, NULL, &event_2, ^ { mat_vec_mul_gpu_fp32(buffer_A, buffer_x, buffer_res, m, n); });
    enqueue_kernel(get_default_queue(), CLK_ENQUEUE_FLAGS_NO_WAIT, ndrange_1D(m), 1, &event_2, &event_1, ^ { vec_sub_gpu_sp(buffer_res, buffer_b); });
    enqueue_kernel(get_default_queue(), CLK_ENQUEUE_FLAGS_NO_WAIT, ndrange_1D(m), 1, &event_1, &event_2, ^ { vec_scalar_gpu_sp(buffer_y, (1 / beta)); });
    enqueue_kernel(get_default_queue(), CLK_ENQUEUE_FLAGS_NO_WAIT, ndrange_1D(m), 1, &event_2, &event_1, ^ { vec_sub_gpu_sp(buffer_res, buffer_y); });
    enqueue_kernel(get_default_queue(), CLK_ENQUEUE_FLAGS_NO_WAIT, ndrange_1D(n), 1, &event_1, &event_2, ^ { mat_vec_mul_gpu_fp32(buffer_A_t, buffer_res, buffer_res2, n, m); });
    enqueue_kernel(get_default_queue(), CLK_ENQUEUE_FLAGS_NO_WAIT, ndrange_1D(n), 1, &event_2, &event_1, ^ { vec_sub_gpu_sp(buffer_x, buffer_res2); });
    enqueue_kernel(get_default_queue(), CLK_ENQUEUE_FLAGS_NO_WAIT, ndrange_1D(n), 1, &event_1, &event_2, ^ { shrink_gpu_sp(buffer_x, (tau / beta)); });
    enqueue_kernel(get_default_queue(), CLK_ENQUEUE_FLAGS_NO_WAIT, ndrange_1D(m), 1, &event_2, &event_1, ^ { mat_vec_mul_gpu_fp32(buffer_A, buffer_x, buffer_res2, m, n); });
    enqueue_kernel(get_default_queue(), CLK_ENQUEUE_FLAGS_NO_WAIT, ndrange_1D(m), 1, &event_1, &event_2, ^ { vec_sub_gpu_sp(buffer_res2, buffer_b); });
    enqueue_kernel(get_default_queue(), CLK_ENQUEUE_FLAGS_NO_WAIT, ndrange_1D(m), 1, &event_2, &event_1, ^ { vec_scalar_gpu_sp(buffer_res2, (gamma * beta)); });
    enqueue_kernel(get_default_queue(), CLK_ENQUEUE_FLAGS_NO_WAIT, ndrange_1D(m), 1, &event_1, &event_2, ^ { vec_scalar_gpu_sp(buffer_y, (1 / (1 / beta))); });
    enqueue_kernel(get_default_queue(), CLK_ENQUEUE_FLAGS_NO_WAIT, ndrange_1D(m), 1, &event_2, &event_1, ^ { vec_sub_gpu_sp(buffer_y, buffer_res2); });

    for (int i = 0; i <= iterations; i++) {
        enqueue_kernel(get_default_queue(), CLK_ENQUEUE_FLAGS_NO_WAIT, ndrange_1D(m), 1, &event_1, &event_2, ^ { mat_vec_mul_gpu_fp32(buffer_A, buffer_x, buffer_res, m, n); });
        enqueue_kernel(get_default_queue(), CLK_ENQUEUE_FLAGS_NO_WAIT, ndrange_1D(m), 1, &event_2, &event_1, ^ { vec_sub_gpu_sp(buffer_res, buffer_b); });
        enqueue_kernel(get_default_queue(), CLK_ENQUEUE_FLAGS_NO_WAIT, ndrange_1D(m), 1, &event_1, &event_2, ^ { vec_scalar_gpu_sp(buffer_y, (1 / beta)); });
        enqueue_kernel(get_default_queue(), CLK_ENQUEUE_FLAGS_NO_WAIT, ndrange_1D(m), 1, &event_2, &event_1, ^ { vec_sub_gpu_sp(buffer_res, buffer_y); });
        enqueue_kernel(get_default_queue(), CLK_ENQUEUE_FLAGS_NO_WAIT, ndrange_1D(n), 1, &event_1, &event_2, ^ { mat_vec_mul_gpu_fp32(buffer_A_t, buffer_res, buffer_res2, n, m); });
        enqueue_kernel(get_default_queue(), CLK_ENQUEUE_FLAGS_NO_WAIT, ndrange_1D(n), 1, &event_2, &event_1, ^ { vec_sub_gpu_sp(buffer_x, buffer_res2); });
        enqueue_kernel(get_default_queue(), CLK_ENQUEUE_FLAGS_NO_WAIT, ndrange_1D(n), 1, &event_1, &event_2, ^ { shrink_gpu_sp(buffer_x, (tau / beta)); });
        enqueue_kernel(get_default_queue(), CLK_ENQUEUE_FLAGS_NO_WAIT, ndrange_1D(m), 1, &event_2, &event_1, ^ { mat_vec_mul_gpu_fp32(buffer_A, buffer_x, buffer_res2, m, n); });    
        enqueue_kernel(get_default_queue(), CLK_ENQUEUE_FLAGS_NO_WAIT, ndrange_1D(m), 1, &event_1, &event_2, ^ { vec_sub_gpu_sp(buffer_res2, buffer_b); });
        enqueue_kernel(get_default_queue(), CLK_ENQUEUE_FLAGS_NO_WAIT, ndrange_1D(m), 1, &event_2, &event_1, ^ { vec_scalar_gpu_sp(buffer_res2, (gamma * beta)); });
        enqueue_kernel(get_default_queue(), CLK_ENQUEUE_FLAGS_NO_WAIT, ndrange_1D(m), 1, &event_1, &event_2, ^ { vec_scalar_gpu_sp(buffer_y, (1 / (1 / beta))); });
        enqueue_kernel(get_default_queue(), CLK_ENQUEUE_FLAGS_NO_WAIT, ndrange_1D(m), 1, &event_2, &event_1, ^ { vec_sub_gpu_sp(buffer_y, buffer_res2); });
        //printf("err %d, %d", err, i);
    }

}