#include <stdio.h>

__global__ void whoami(void) {
    int block_id =
        blockIdx.x +    
        blockIdx.y * gridDim.x +    
        blockIdx.z * gridDim.x * gridDim.y;   

    int block_offset =
        block_id * 
        blockDim.x * blockDim.y * blockDim.z; 

    int thread_offset =
        threadIdx.x +  
        threadIdx.y * blockDim.x +
        threadIdx.z * blockDim.x * blockDim.y;

    int id = block_offset + thread_offset; //global thread id

    printf("%04d | Block(%d %d %d) = %3d | Thread(%d %d %d) = %3d\n",
        id,
        blockIdx.x, blockIdx.y, blockIdx.z, block_id,
        threadIdx.x, threadIdx.y, threadIdx.z, thread_offset);
    // printf("blockIdx.x: %d, blockIdx.y: %d, blockIdx.z: %d, threadIdx.x: %d, threadIdx.y: %d, threadIdx.z: %d\n", blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z);
}

int main(int argc, char **argv) {
    const int b_x = 2, b_y = 3, b_z = 4;
    const int t_x = 4, t_y = 4, t_z = 4; // the max warp size is 32, so 
    // we will get 2 warp of 32 threads per block

    int blocks_per_grid = b_x * b_y * b_z;
    int threads_per_block = t_x * t_y * t_z;

    printf("%d blocks/grid\n", blocks_per_grid);
    printf("%d threads/block\n", threads_per_block);
    printf("%d total threads\n", blocks_per_grid * threads_per_block);

    dim3 blocksPerGrid(b_x, b_y, b_z); // 3d cube of shape 2*3*4 = 24
    dim3 threadsPerBlock(t_x, t_y, t_z); // 3d cube of shape 4*4*4 = 64

    whoami<<<blocksPerGrid, threadsPerBlock>>>();
    cudaDeviceSynchronize();
}


// In CUDA, gridDim is a built-in variable that represents the dimensions of the grid (the entire set of blocks) in the x, y, and z directions. It is used to determine how many blocks are in the grid along each axis.

// Analogy
// Think of the grid as a city made up of neighborhoods (blocks):

// gridDim.x: The number of blocks (neighborhoods) along the x-direction in the city.
// gridDim.y: The number of blocks along the y-direction.
// gridDim.z: The number of blocks along the z-direction.

// Details
// Type: gridDim is of type dim3, a structure that holds three integer components: x, y, and z.
// Scope: gridDim is accessible from within device (GPU) code and provides information about how the grid is organized.

// Example
// If the grid is defined as:

// dim3 blocksPerGrid(2, 3, 4);
// Then:

// gridDim.x = 2 (2 blocks in the x-direction).
// gridDim.y = 3 (3 blocks in the y-direction).
// gridDim.z = 4 (4 blocks in the z-direction).

// Usage
// gridDim is often used in calculations to:

// Determine the global block index (like block_id in the provided code).
// Navigate through blocks for algorithms that require inter-block communication or global data indexing.
// For example, calculating a unique block ID:

// int block_id = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
// This formula computes a 1D block index by flattening the 3D grid structure, based on the grid dimensions.
