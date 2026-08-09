#include <cuda_runtime.h>

#include "cuda_check.cuh"

int main()
{
    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));

    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaFree(nullptr));

    return 0;
}