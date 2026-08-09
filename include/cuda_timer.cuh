#pragma once

#include <cuda_runtime.h>

#include "cuda_check.cuh"

class CudaTimer
{
public:
    CudaTimer()
    {
        CUDA_CHECK(cudaEventCreate(&start_event_));
        CUDA_CHECK(cudaEventCreate(&stop_event_));
    }

    ~CudaTimer()
    {
        // 析构函数中不再调用 CUDA_CHECK，避免析构失败时直接退出
        cudaEventDestroy(start_event_);
        cudaEventDestroy(stop_event_);
    }

    void start()
    {
        CUDA_CHECK(cudaEventRecord(start_event_));
    }

    float stop()
    {
        CUDA_CHECK(cudaEventRecord(stop_event_));
        CUDA_CHECK(cudaEventSynchronize(stop_event_));

        float elapsed_ms = 0.0f;

        CUDA_CHECK(
            cudaEventElapsedTime(
                &elapsed_ms,
                start_event_,
                stop_event_
            )
        );

        return elapsed_ms;
    }

private:
    cudaEvent_t start_event_{};
    cudaEvent_t stop_event_{};
};