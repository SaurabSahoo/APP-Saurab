#include <stdio.h>
#include <math.h>

__global__ void cos_sin_half(float *x, float *result, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N / 2) {
        result[idx] = cosf(x[idx]);
    } else if (idx < N) {
        result[idx] = sinf(x[idx]);
    }
}

__global__ void cos_sin_even_odd(float *x, float *result, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx % 2 == 0 && idx < N) {
        result[idx] = cosf(x[idx]);
    } else if (idx % 2 == 1 && idx < N) {
        result[idx] = sinf(x[idx]);
    }
}

int main() {
    int N = 1024; // Example size of vectors
    float *x, *result;
    float *d_x, *d_result;

    // Allocate memory on host
    x = (float*)malloc(N * sizeof(float));
    result = (float*)malloc(N * sizeof(float));

    // Initialize input vector
    for (int i = 0; i < N; i++) {
        x[i] = (float)i;
    }

    // Allocate memory on device
    cudaMalloc(&d_x, N * sizeof(float));
    cudaMalloc(&d_result, N * sizeof(float));

    // Copy input vector from host to device
    cudaMemcpy(d_x, x, N * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel for cos_sin_half
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    cos_sin_half<<<numBlocks, blockSize>>>(d_x, d_result, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("cos_sin_half kernel execution time: %f milliseconds\n", milliseconds);

    // Copy result vector from device to host
    cudaMemcpy(result, d_result, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Launch kernel for cos_sin_even_odd
    cudaEventRecord(start);
    cos_sin_even_odd<<<numBlocks, blockSize>>>(d_x, d_result, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("cos_sin_even_odd kernel execution time: %f milliseconds\n", milliseconds);

    // Copy result vector from device to host
    cudaMemcpy(result, d_result, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_x);
    cudaFree(d_result);

    // Free host memory
    free(x);
    free(result);

    return 0;
}
