#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <sys/time.h>

#define ARRAY_SIZE (1 << 20) // Total number of elements in the array
#define THREADS_PER_BLOCK 256

// Kernel to access array elements with variable stride
__global__ void accessArrayWithStride(float *array, int stride) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;

    for (int i = tid; i < ARRAY_SIZE; i += stride) {
        sum += array[i];
    }

    // Ensure all threads have finished computation
    __syncthreads();

    // Only one thread per block writes the sum
    if (threadIdx.x == 0) {
        array[blockIdx.x] = sum;
    }
}

int main() {
    float *d_array;
    float *h_array = (float*)malloc(sizeof(float) * ARRAY_SIZE);
    struct timeval start, end;

    // Initialize array with sequential values
    for (int i = 0; i < ARRAY_SIZE; i++) {
        h_array[i] = i;
    }

    // Allocate memory on GPU
    cudaMalloc(&d_array, sizeof(float) * ARRAY_SIZE);
    cudaMemcpy(d_array, h_array, sizeof(float) * ARRAY_SIZE, cudaMemcpyHostToDevice);

    // Run the kernel with different stride values and measure time
    printf("Stride\tBandwidth (GB/s)\n");
    for (int stride = 1; stride <= ARRAY_SIZE / 2; stride *= 2) {
        gettimeofday(&start, NULL);

        accessArrayWithStride<<<ARRAY_SIZE / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(d_array, stride);
        cudaDeviceSynchronize(); // Wait for all kernels to finish

        gettimeofday(&end, NULL);

        double elapsedTime = (end.tv_sec - start.tv_sec) * 1000.0; // Convert to milliseconds
        elapsedTime += (end.tv_usec - start.tv_usec) / 1000.0;

        double dataSize = sizeof(float) * ARRAY_SIZE; // Total data size in bytes
        double bandwidth = dataSize / (elapsedTime * 1e-3) / (1024 * 1024 * 1024); // Bandwidth in GB/s

        printf("%d\t%f\n", stride, bandwidth);
    }

    // Free memory
    cudaFree(d_array);
    free(h_array);

    return 0;
}
