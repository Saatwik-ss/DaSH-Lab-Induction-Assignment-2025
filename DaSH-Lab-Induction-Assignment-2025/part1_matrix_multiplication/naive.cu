#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>


#define N 1024  // 


// logic
__global__ void matrixMulNaive(const float* A, const float* B, float* C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n && col < n) {
        float val = 0.0f;
        for (int k = 0; k < n; ++k)
            val += A[row * n + k] * B[k * n + col];
        C[row * n + col] = val;
    }
}


int main() {
    int size = N * N * sizeof(float);
    float *h_A, *h_B, *h_C;
    h_A = (float*)malloc(size);
    h_B = (float*)malloc(size);
    h_C = (float*)malloc(size);

    // Initialize matrices
    for (int i = 0; i < N * N; i++) {
        h_A[i] = static_cast<float>(rand()) / RAND_MAX;
        h_B[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // Device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    dim3 threads(16, 16);
    dim3 blocks((N + threads.x - 1) / threads.x, (N + threads.y - 1) / threads.y);

    //kernel timings

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    matrixMulNaive<<<blocks, threads>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float msNaive = 0.0f;
    cudaEventElapsedTime(&msNaive, start, stop);
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Compute GFLOPs
    double flops = 2.0 * N * N * N;
    double gflopsNaive = flops / (msNaive * 1e6);

    std::cout <<"Naive-2 CUDA matrix multiplication:" << std::endl;
    std::cout <<msNaive<< " ms,  Performance: " << gflopsNaive << " GFLOPS" << std::endl;
 

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);


    return 0;
}



//nvcc -O3 -lcublas -arch=sm_75 matrix_mul_compare.cu -o matrix_mul_compare
//./matrix_mul_compare
