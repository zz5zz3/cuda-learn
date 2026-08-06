#include<stdio.h>
using namespace std;
int main(){
    int iDevicecount=0;
    cudaError_t error = cudaGetDeviceCount(&iDevicecount);
    printf("the devicecount is%d\n",iDevicecount);

    int idev=0;
    error = cudaSetDevice(idev);
    printf("error:%d,success:%d\n",error,cudaSuccess);



    return 0;
}