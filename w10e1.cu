#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>

#define THREADS_PER_BLOCK 256
#define ARRAY_SIZE 131072
#define NUM_BLOCKS ARRAY_SIZE/THREADS_PER_BLOCK
#define NUM_TESTS 10

float hostArray[ARRAY_SIZE];
__device__ float deviceArray[ARRAY_SIZE];
__device__ float deviceMin;

int timeDifference(double *result, struct timeval *start, struct timeval *end) {
    struct timeval diff;

    diff.tv_sec = end->tv_sec - start->tv_sec;
    diff.tv_usec = end->tv_usec - start->tv_usec;
    *result = ((double)diff.tv_usec) / 1e6 + (double)diff.tv_sec;

    return end->tv_sec < start->tv_sec;
}

__global__ void initializeKernel() {
    deviceMin = INFINITY;
}

__global__ void findMinKernel() {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < ARRAY_SIZE) {
        atomicMin((int*)&deviceMin, __float_as_int(deviceArray[idx]));
    }
}

int main(int argc, char **argv) {
    struct timeval startTime, endTime, elapsedTime;
    double timeElapsed, min0;
    float min;
    int deviceId, deviceCount, err;

    cudaGetDevice(&deviceId);
    if (cudaGetDeviceCount(&deviceCount) || deviceCount == 0) {
        printf("No CUDA devices found!\n");
        exit(1);
    } else {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, deviceId);
        printf("Number of CUDA devices, Device ID: %d %d\n", deviceCount, deviceId);
        printf("Device: %s\n", deviceProp.name);
        printf("[Device major.minor]: [%d.%d]\n\n", deviceProp.major, deviceProp.minor);
    }

    srand(time(NULL)); // Seed for random number generation

    for (int testIdx = 0; testIdx < NUM_TESTS; testIdx++) {
        for (int i = 0; i < ARRAY_SIZE; i++) {
            hostArray[i] = (float)rand() / (float)RAND_MAX;
        }

        min0 = INFINITY;
        for (int i = 0; i < ARRAY_SIZE; i++)
            min0 = fminf(min0, hostArray[i]);

        if (err = cudaMemcpyToSymbol(deviceArray, hostArray, ARRAY_SIZE * sizeof(float), 0, cudaMemcpyHostToDevice)) {
            printf("Error %d\n", err);
            exit(err);
        }

        initializeKernel<<<1, 1>>>();

        if (err = cudaDeviceSynchronize()) {
            printf("Error %d\n", err);
            exit(err);
        }

        gettimeofday(&startTime, NULL);

        findMinKernel<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>();

        if (err = cudaDeviceSynchronize()) {
            printf("Error %d\n", err);
            exit(err);
        }
        gettimeofday(&endTime, NULL);
        elapsedTime = startTime;
        timeDifference(&timeElapsed, &endTime, &elapsedTime);

        if (err = cudaMemcpyFromSymbol(&min, deviceMin, sizeof(float), 0, cudaMemcpyDeviceToHost)) {
            printf("Error %d\n", err);
            exit(err);
        }

        printf("Minimum value found: %e (relative error %e)\n", min, fabs(min - min0) / min0);
        printf("Time taken: %e\n", timeElapsed);
    }

    return 0;
}
