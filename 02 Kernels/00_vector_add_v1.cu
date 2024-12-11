#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

#define N 10000000 // Vector size = 10 million
#define BLOCK_SIZE 1024

// Example:
// A = [1, 2, 3, 4, 5]
// B = [6, 7, 8, 9, 10]
// C = A + B = [7, 9, 11, 13, 15]

// CPU vector addition function
// This function runs on the host (CPU) and performs a simple vector addition in a serial manner.
void vector_add_cpu(float *a, float *b, float *c, int n)
{
    // Loop over all elements in the arrays
    for (int i = 0; i < n; i++)
    {
        // For each index i, add the corresponding elements of 'a' and 'b'
        // and store the result in 'c'
        c[i] = a[i] + b[i];
    }
}

// CUDA kernel for vector addition
// This function runs on the device (GPU) and is executed by multiple threads in parallel.
// Each thread computes one element of the resulting vector 'c'.
__global__ void vector_add_gpu(float *a, float *b, float *c, int n)
{
    // Compute a global thread index 'i' that uniquely identifies the thread
    // blockIdx.x: The index of the block in the grid (in the x-dimension)
    // blockDim.x: The number of threads per block (in the x-dimension)
    // threadIdx.x: The index of the thread within the current block (in the x-dimension)
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Check if this thread's index is within the valid range of the array size 'n'
    // Some threads might not have valid work if (total threads > n)
    if (i < n)
    {
        // Perform vector addition for the element at index i
        c[i] = a[i] + b[i];
    }
    // Threads with indices >= n do nothing (due to the if-statement guard)
}

// Initialize vector with random values
void init_vector(float *vec, int n) {
    for (int i = 0; i < n; i++) {
        vec[i] = (float)rand() / RAND_MAX;
    }
}

// Function to measure execution time
double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main()
{
    float *h_a, *h_b, *h_c_cpu, *h_c_gpu; // Host pointers
    float *d_a, *d_b, *d_c;               // Device pointers
    size_t size = N * sizeof(float);

    // Alocate host memory
    h_a = (float*)malloc(size);
    h_b = (float*)malloc(size);
    h_c_cpu = (float*)malloc(size);
    h_c_gpu = (float*)malloc(size);

    // Initialize vectors
    srand(time(NULL));
    init_vector(h_a, N);
    init_vector(h_b, N);

    // Alocate device memory 
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    // Copy Data to device 
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    int num_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    // N = 1024, BLOCK_SIZE = 256, num_blocks = 4
    // (N + BLOCK_SIZE - 1) / BLOCK_SIZE = ( (1024 + 256 - 1) / 256 ) = 1279 / 256 = 4 rounded 

    //Warm-up runs
    printf("Performing warm-up runs...\n");
    for (int i = 0; i < 3; i++)
    {
        vector_add_cpu(h_a, h_b, h_c_cpu, N);
        vector_add_gpu<<<num_blocks, BLOCK_SIZE>>>(d_a, d_b, d_c, N);
        cudaDeviceSynchronize();   
        
    }

    // Benchmark CPU implementation
    printf("Benchmarking CPU implementation...\n");
    double cpu_total_time = 0.0;
    for (int i = 0; i < 20; i++) {
        double start_time = get_time();
        vector_add_cpu(h_a, h_b, h_c_cpu, N);
        double end_time = get_time();
        cpu_total_time += end_time - start_time;
    }
    double cpu_avg_time = cpu_total_time / 20.0;

    // Benchmark GPU implementation
    printf("Benchmarking GPU implementation...\n");
    double gpu_total_time = 0.0;
    for (int i = 0; i < 20; i++) {
        double start_time = get_time();
        vector_add_gpu<<<num_blocks, BLOCK_SIZE>>>(d_a, d_b, d_c, N);
        cudaDeviceSynchronize();
        double end_time = get_time();
        gpu_total_time += end_time - start_time;
    }
    double gpu_avg_time = gpu_total_time / 20.0;

    // Print results
    printf("CPU average time: %f milliseconds\n", cpu_avg_time*1000);
    printf("GPU average time: %f milliseconds\n", gpu_avg_time*1000);
    printf("Speedup: %fx\n", cpu_avg_time / gpu_avg_time);

    // Verify results (optional)
    cudaMemcpy(h_c_gpu, d_c, size, cudaMemcpyDeviceToHost);
    bool correct = true;
    for (int i = 0; i < N; i++) {
        if (fabs(h_c_cpu[i] - h_c_gpu[i]) > 1e-5) {
            correct = false;
            break;
        }
    }
    printf("Results are %s\n", correct ? "correct" : "incorrect");

    // Free memory
    free(h_a);
    free(h_b);
    free(h_c_cpu);
    free(h_c_gpu);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
