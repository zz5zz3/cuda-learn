#pragma once

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                   \
do {                                                                       \
    cudaError_t error = (call);                                            \
    if (error != cudaSuccess) {                                            \
        std::fprintf(                                                      \
            stderr,                                                        \
            "CUDA error at %s:%d\n"                                        \
            "  code: %d\n"                                                 \
            "  name: %s\n"                                                 \
            "  message: %s\n",                                             \
            __FILE__,                                                      \
            __LINE__,                                                      \
            static_cast<int>(error),                                       \
            cudaGetErrorName(error),                                       \
            cudaGetErrorString(error)                                      \
        );                                                                 \
        std::exit(EXIT_FAILURE);                                           \
    }                                                                      \
} while (0)