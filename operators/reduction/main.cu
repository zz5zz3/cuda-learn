#include<stdio.h>
#include"cuda_check.cuh"
#include"cuda_device.cuh"
#include"cuda_timer.cuh"

__global__ void reduce_sum( const float* input,float* block_sums,int n){
    extern __shared__ float shared_data[];  //为什么要加extern ：这个是手动声明，因为它的大小不是固定的，需要看shared_memory_bytes
    const int bid=blockIdx.x;
    const int tid=threadIdx.x;
    int id=bid*blockDim.x+tid;
    if(id<n){
        shared_data[tid]=input[id];
    }
    else{
        shared_data[tid]=0.0f;  //这是啥：共享内存在物理上存在，不会自动初始化，可能是会残留上次计算的物理量
    }

     __syncthreads();//这又是啥, 线程间同步

    for(int stride=blockDim.x/2;stride>0;stride/=2){
        if(tid<stride){
        shared_data[tid]+=shared_data[tid+stride];
        }
    __syncthreads();
    }

    if (tid == 0)
    {
        block_sums[blockIdx.x] = shared_data[0];
    }
}

void initialData(float *addr, int elemCount)
{
    for (int i = 0; i < elemCount; i++)
    {
        addr[i] = (float)(rand() & 0xFF) / 10.f;
    }
    return;
}

int main(){

    set_cuda_device(0);
    int elem=512;
//    int byte=2048;
    int byte = elem*sizeof(float);
    float* hostA;
    float* hostB;
    float* hostC;
    hostA=(float*)malloc(byte);
    hostB=(float*)malloc(byte);
    hostC=(float*)malloc(byte);

    initialData(hostA,elem);
    initialData(hostB,elem);

    float* cudaA;
    float* cudaB;
    float* cudaC;
    CUDA_CHECK(cudaMalloc((float**)&cudaA,byte));
    cudaMalloc(&cudaB,byte);
    cudaMalloc(&cudaC,byte);

    cudaMemcpy(cudaA,hostA,byte,cudaMemcpyHostToDevice);
    cudaMemcpy(cudaB,hostB,byte,cudaMemcpyHostToDevice);

    dim3 block(32);
    dim3 grid(elem/32);
    int shared_memory_bytes = block.x*sizeof(float);

    reduce_sum<<<grid,block,shared_memory_bytes>>>(cudaA,cudaB,elem);
    cudaDeviceSynchronize();
    cudaMemcpy(hostB,cudaB,byte,cudaMemcpyDeviceToHost);

    printf("%0.2f",hostB[0]);
}



/*#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#include "cuda_check.cuh"
#include "cuda_device.cuh"
#include "cuda_timer.cuh"


__global__ void reduce_sum(
    const float* input,
    float* block_sums,
    int n
)
{
    extern __shared__ float shared_data[];

    const int tid = threadIdx.x;
    const int global_index =
        blockIdx.x * blockDim.x + threadIdx.x;

    // 越界线程填入0，不影响加法结果
    if (global_index < n)
    {
        shared_data[tid] = input[global_index];
    }
    else
    {
        shared_data[tid] = 0.0f;
    }

    // 保证整个线程块都完成数据加载
    __syncthreads();

    // 在共享内存中进行树形规约
    for (int stride = blockDim.x / 2;
         stride > 0;
         stride /= 2)
    {
        if (tid < stride)
        {
            shared_data[tid] +=
                shared_data[tid + stride];
        }

        __syncthreads();
    }

    // 每个线程块只写出一个部分和
    if (tid == 0)
    {
        block_sums[blockIdx.x] = shared_data[0];
    }
}


void initialize_data(float* data, int n)
{
    for (int i = 0; i < n; ++i)
    {
        data[i] =
            static_cast<float>(std::rand() & 0xFF) / 10.0f;
    }
}


double reduce_sum_cpu(const float* data, int n)
{
    double sum = 0.0;

    for (int i = 0; i < n; ++i)
    {
        sum += static_cast<double>(data[i]);
    }

    return sum;
}


int main()
{
    set_cuda_device(0);
    CUDA_CHECK(cudaFree(nullptr));

    constexpr int element_count = 1 << 20;
    constexpr int block_size = 256;

    const int grid_size =
        (element_count + block_size - 1) / block_size;

    const std::size_t input_bytes =
        static_cast<std::size_t>(element_count) * sizeof(float);

    const std::size_t output_bytes =
        static_cast<std::size_t>(grid_size) * sizeof(float);

    // 每个线程在共享内存中保存一个float
    const std::size_t shared_memory_bytes =
        static_cast<std::size_t>(block_size) * sizeof(float);

    float* host_input =
        static_cast<float*>(std::malloc(input_bytes));

    float* host_block_sums =
        static_cast<float*>(std::malloc(output_bytes));

    if (host_input == nullptr || host_block_sums == nullptr)
    {
        std::fprintf(stderr, "Failed to allocate host memory.\n");
        std::free(host_input);
        std::free(host_block_sums);
        return EXIT_FAILURE;
    }

    std::srand(0);
    initialize_data(host_input, element_count);

    const double cpu_result =
        reduce_sum_cpu(host_input, element_count);

    float* device_input = nullptr;
    float* device_block_sums = nullptr;

    CUDA_CHECK(cudaMalloc(&device_input, input_bytes));
    CUDA_CHECK(cudaMalloc(&device_block_sums, output_bytes));

    CUDA_CHECK(cudaMemcpy(
        device_input,
        host_input,
        input_bytes,
        cudaMemcpyHostToDevice
    ));

    CudaTimer timer;
    timer.start();

    reduce_sum<<<
        grid_size,
        block_size,
        shared_memory_bytes
    >>>(
        device_input,
        device_block_sums,
        element_count
    );

    CUDA_CHECK(cudaGetLastError());

    const float elapsed_ms = timer.stop();

    CUDA_CHECK(cudaMemcpy(
        host_block_sums,
        device_block_sums,
        output_bytes,
        cudaMemcpyDeviceToHost
    ));

    // CPU负责把各线程块的部分和再次相加
    double gpu_result = 0.0;

    for (int i = 0; i < grid_size; ++i)
    {
        gpu_result +=
            static_cast<double>(host_block_sums[i]);
    }

    const double absolute_error =
        std::fabs(cpu_result - gpu_result);

    // 规约会改变浮点加法顺序，不能要求结果完全相等
    const double tolerance =
        1.0e-5 * std::fabs(cpu_result) + 1.0e-3;

    const bool correct =
        absolute_error <= tolerance;

    const double bandwidth_gb_s =
        static_cast<double>(input_bytes) /
        (static_cast<double>(elapsed_ms) * 1.0e6);

    std::printf("Element count:   %d\n", element_count);
    std::printf("Grid size:       %d\n", grid_size);
    std::printf("Block size:      %d\n", block_size);
    std::printf("Partial sums:    %d\n", grid_size);
    std::printf("Kernel time:     %.6f ms\n", elapsed_ms);
    std::printf("Bandwidth:       %.2f GB/s\n", bandwidth_gb_s);
    std::printf("CPU result:      %.6f\n", cpu_result);
    std::printf("GPU result:      %.6f\n", gpu_result);
    std::printf("Absolute error:  %.6f\n", absolute_error);
    std::printf(
        "Result:          %s\n",
        correct ? "correct" : "incorrect"
    );

    CUDA_CHECK(cudaFree(device_input));
    CUDA_CHECK(cudaFree(device_block_sums));

    std::free(host_input);
    std::free(host_block_sums);

    return correct ? EXIT_SUCCESS : EXIT_FAILURE;
}*/