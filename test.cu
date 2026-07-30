#include<stdio.h>
using namespace std;

__global__ void hellofromgpu(){
    printf("hello\n");
}

int main(){
    hellofromgpu<<<4,4>>>();
    cudaDeviceSynchronize();
    return 0;
}
