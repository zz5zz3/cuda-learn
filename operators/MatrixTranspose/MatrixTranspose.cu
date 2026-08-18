#include<stdio.h>
#include"cuda_check.cuh"
#include"cuda_device.cuh"
#include"cuda_timer.cuh"
#include <time.h>



__global__ void MatrixTranspose(float* A,float* B,float* C,int raw,int col){
     __shared__ float tile[32][33];

    int bid = blockIdx.y * gridDim.x +blockIdx.x;
    int tid = threadIdx.y + blockDim.x + threadIdx.x;
    int id  = bid * blockDim.x *blockDim.y + tid;

    

    int x   = blockDim.x * blockIdx.x + threadIdx.x;
    int y   = blockDim.y * blockIdx.y + threadIdx.y;
    int tx  = threadIdx.x;
    int ty  = threadIdx.y;
    
    if(x<col&&y<raw){
        tile[ty][tx] = A[y*col + x];
    }

    __syncthreads();
/*我认为我透彻了，先选中block，也就是把不同的tail给转置过来，这个时候要反着来，
然后从写入的角度来思考，线程束在block上按照thread顺序写入这些转置后的矩阵，也
就是说连续的线程从共享内存读出来就转置好了，只要改tx和ty的位置就行，这样每个线
程束写入的顺序就能连续，共享内存的读取速度不用管，直接按照转置读出来，然后写入
别忘了C的形状xy长度其实变了，所以索引边界也要改
**/
    int output_x= blockDim.y * blockIdx.y + threadIdx.x;
    int output_y= blockDim.x * blockIdx.x + threadIdx.y;
    if(output_x<raw&& output_y<col){
        C[output_y*raw+output_x]=tile[tx][ty];
    }
}


/*    // 输出 Tile 的位置交换
    int out_x =
        blockIdx.y * blockDim.y + tx;

    int out_y =
        blockIdx.x * blockDim.x + ty;

    // Shared -> Global
    if(out_x < raw && out_y < col)
    {
        C[out_y * raw + out_x] =
            tile[tx][ty];
    }
}*/
void initialData(float *addr, int elemCount)
{
    for (int i = 0; i < elemCount; i++)
    {
        addr[i] = (float)(rand() & 0xFF) / 10.f;
    }
    return;
}

int main(){
/*0 初始化区*/
    set_cuda_device(0);
    int rows = 2;
    int cols = 3;

    int elem = rows * cols;
   // int elem=1 << 24;
//    int byte=2048;
    int byte = elem*sizeof(float);
/*1 主机准备区*/

    float* hostA;
    float* hostB;
    float* hostC;
    float* hostRef;
    hostA=(float*)malloc(byte);
    hostB=(float*)malloc(byte);
    hostC=(float*)malloc(byte);
hostA[0] = 1;
hostA[1] = 2;
hostA[2] = 3;
hostA[3] = 4;
hostA[4] = 5;
hostA[5] = 6;
    hostRef=(float*)malloc(byte);


/*2 设备准备区*/

    float* cudaA;
    float* cudaB;
    float* cudaC;
    
    CudaTimer cudatimer;
    
    CUDA_CHECK(cudaMalloc((float**)&cudaA,byte));
    CUDA_CHECK(cudaMalloc(&cudaB,byte));
    CUDA_CHECK(cudaMalloc(&cudaC,byte));
    CUDA_CHECK(cudaMemset(cudaA,0,byte));
    CUDA_CHECK(cudaMemset(cudaB,0,byte));
    CUDA_CHECK(cudaMemset(cudaC,0,byte));

    CUDA_CHECK(cudaMemcpy(cudaA,hostA,byte,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(cudaB,hostB,byte,cudaMemcpyHostToDevice));

/*3 GPU计算区*/
    dim3 block(16,16);

    int memory_size=block.y*block.x*sizeof(float);
    dim3 grid(((cols + block.x - 1) / block.x),(rows + block.y - 1) / block.y);

    cudatimer.start();
        MatrixTranspose<<<grid,block,memory_size>>>(cudaA,cudaB,cudaC,rows,cols);

    float elapsed_ms=cudatimer.stop();

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(hostC,cudaC,byte,cudaMemcpyDeviceToHost));

/*4 CPU计算区*/
    clock_t  start=clock();
    for(int i=0;i<rows;i++)
        for(int j=0;j<cols;j++){
            hostRef[j * rows + i]=hostA[i*cols + j];
        }
    
    clock_t  stop=clock();
    double time=(stop-start)*1000/CLOCKS_PER_SEC;
/* 5 结果验证区 */

for(int i=0;i<6;i++){
    printf("host:%0.2f\n",hostRef[i]);
}
for(int i=0;i<6;i++){
    printf("cuda:%0.2f\n",hostC[i]);
}
printf("cudatime:%0.2fms,cputime:%0.2fms\n",elapsed_ms,time);

double total_bytes =
    (double) (elem * sizeof(float)+grid.x* sizeof(float));

double bandwidth =
    total_bytes /
    (elapsed_ms / 1000.0) /
    1e9;

printf("Memory Bandwidth: %.2f GB/s\n", bandwidth);

/* 6 资源释放区 */

    CUDA_CHECK(cudaFree(cudaA));
    CUDA_CHECK(cudaFree(cudaB));
    CUDA_CHECK(cudaFree(cudaC));

    free(hostA);
    free(hostB);
    free(hostC);
    free(hostRef);

    return 0;
}