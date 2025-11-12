#include <iostream>
#include <cuda_runtime.h>
#define N 1024 


__global__ void matrixMulNaive(const float* A, const float* B, float* C, int n) {
    int row=blockIdx.y*blockDim.y +threadIdx.y;
    int col=blockIdx.x*blockDim.x +threadIdx.x;
    if (row <n && col<n) {
        float val=0.0f;
        for (int k=0; k< n; ++k)
            val += A[row*n + k]*B[k*n +col];
        C[row*n +col] = val;
    }
}



int main() {
    int size = N * N*sizeof(float);
   float *h_A=(float*)malloc(size);
    float *h_B= (float*)malloc(size);
    float *h_C= (float*)malloc(size);
    for (int i=0; i< N*N; i++) {
        h_A[i]=static_cast<float>(rand())/RAND_MAX;
        h_B[i]=static_cast<float>(rand())/RAND_MAX;
    }

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    dim3 threads(16, 16);
    dim3 blocks((N + threads.x - 1)/threads.x, (N + threads.y - 1)/threads.y);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // run kernel
    cudaEventRecord(start);
    matrixMulNaive<<<blocks, threads>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    double flops = 2.0*N*N*N;
    double gflops = flops / (ms * 1e6);
    std::cout << "Native CUDA Matrix Multiplication:\n";
    std::cout << ms << "ms,  Performance:" << gflops << "GFLOPS\n";

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    return 0;
}



//nvcc -arch=sm_89 tiled.cu -o tiled.exe
//nvprof ./tiled.exe
