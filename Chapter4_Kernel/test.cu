// CUDA hello, world
#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void helloFromGPU() {
    printf("Hello World from GPU!\n");
}

int main() {
    // Hello from CPU
    printf("Hello World from CPU!\n");

    // Hello from GPU
    helloFromGPU<<<2, 2>>>();
    cudaDeviceReset();

    return 0;
}

