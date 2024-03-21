#include <stdio.h>

#define BLOCK_SIZE 256

__global__ void findPrimes(int N, int K, int *primes, int *count) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int num = N + tid;

    if (num > N + K) return;

    // Check if num is prime
    int is_prime = 1;
    for (int i = 2; i * i <= num; i++) {
        if (num % i == 0) {
            is_prime = 0;
            break;
        }
    }

    // Record prime number if found
    if (is_prime) {
        int index = atomicAdd(count, 1);
        primes[index] = num;
    }
}

int main() {
    int N = 1000; // Starting number
    int K = 10000; // Range size
    int max_range = N + K;
    int *primes, *d_primes, *d_count;
    int prime_count = 0;

    // Allocate memory on host and device
    primes = (int *)malloc(K * sizeof(int));
    cudaMalloc(&d_primes, K * sizeof(int));
    cudaMalloc(&d_count, sizeof(int));
    cudaMemcpy(d_count, &prime_count, sizeof(int), cudaMemcpyHostToDevice);

    // Calculate number of blocks and threads
    int numBlocks = (K + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Check if N+K is not too large for the precision
    if (max_range > 46340 * 46340) {
        printf("N+K is too large for the precision selected.\n");
        return 1;
    }

    // Launch CUDA kernel
    findPrimes<<<numBlocks, BLOCK_SIZE>>>(N, K, d_primes, d_count);

    // Copy results back to host
    cudaMemcpy(&prime_count, d_count, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(primes, d_primes, prime_count * sizeof(int), cudaMemcpyDeviceToHost);

    // Print prime numbers
    printf("Prime numbers between %d and %d:\n", N, N + K);
    for (int i = 0; i < prime_count; i++) {
        printf("%d ", primes[i]);
    }
    printf("\n");

    // Free memory
    free(primes);
    cudaFree(d_primes);
    cudaFree(d_count);

    return 0;
}
