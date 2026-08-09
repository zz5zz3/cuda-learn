#pragma once

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#include "cuda_check.cuh"

inline void set_cuda_device(int device_id = 0)
{
    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));

    if (device_count == 0)
    {
        std::fprintf(stderr, "No CUDA device found.\n");
        std::exit(EXIT_FAILURE);
    }

    if (device_id < 0 || device_id >= device_count)
    {
        std::fprintf(
            stderr,
            "Invalid device ID: %d, available range: [0, %d)\n",
            device_id,
            device_count
        );
        std::exit(EXIT_FAILURE);
    }

    CUDA_CHECK(cudaSetDevice(device_id));

    cudaDeviceProp device_prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&device_prop, device_id));

    std::printf("CUDA device count: %d\n", device_count);
    std::printf("Using device %d: %s\n", device_id, device_prop.name);
    std::printf(
        "Compute capability: %d.%d\n",
        device_prop.major,
        device_prop.minor
    );
}