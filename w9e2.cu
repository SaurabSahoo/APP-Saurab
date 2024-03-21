#include <stdio.h>

#define N 16  // Example size of vectors (perfect square)

__global__ void saxpy(float *x, float *y, float a, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = row * n + col;

    if (row < n && col < n) {
        y[idx] = a * x[idx] + y[idx];
    }
}

int main() {
    float *x, *y;
    float *d_x, *d_y;
    float a = 2.0f;

    // Allocate memory for host vectors
    x = (float*)malloc(N * N * sizeof(float));
    y = (float*)malloc(N * N * sizeof(float));

    // Initialize host vectors
    for (int i = 0; i < N * N; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    // Allocate memory for device vectors
    cudaMalloc(&d_x, N * N * sizeof(float));
    cudaMalloc(&d_y, N * N * sizeof(float));

    // Copy host vectors to device
    cudaMemcpy(d_x, x, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, N * N * sizeof(float), cudaMemcpyHostToDevice);

    // Define block size and grid dimensions
    dim3 blockSize(8, 8); // Each block has 8x8 threads
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (N + blockSize.y - 1) / blockSize.y);

    // Launch kernel and measure execution time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    saxpy<<<gridSize, blockSize>>>(d_x, d_y, a, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Copy result vector from device to host
    cudaMemcpy(y, d_y, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    // Print result (for verification)
    printf("Result vector Y:\n");
    for (int i = 0; i < N * N; i++) {
        printf("%.2f ", y[i]);
        if ((i + 1) % N == 0) {
            printf("\n");
        }
    }

    printf("SAXPY computation with 2D block decomposition completed.\n");
    printf("Execution time: %f milliseconds.\n", milliseconds);

    // Free device memory
    cudaFree(d_x);
    cudaFree(d_y);

    // Free host memory
    free(x);
    free(y);

    return 0;
}
