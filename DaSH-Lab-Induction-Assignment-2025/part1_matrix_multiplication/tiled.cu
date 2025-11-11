%%writefile matrix_mul_tiled_prof.cu
#include <stdio.h>
#include <cuda_runtime.h>

#define TILE 16

__global__ void matmul_naive(const float *A,const float *B,float *C,int M,int N,int K){
 int row=blockIdx.y*blockDim.y+threadIdx.y;
 int col=blockIdx.x*blockDim.x+threadIdx.x;
 if(row<M && col<K){
   float val=0;
   for(int i=0;i<N;i++)
     val+=A[row*N+i]*B[i*K+col];
   C[row*K+col]=val;
 }
}

__global__ void matmul_tiled(const float *A,const float *B,float *C,int M,int N,int K){
 __shared__ float As[TILE][TILE];
 __shared__ float Bs[TILE][TILE];
 int row=blockIdx.y*TILE+threadIdx.y;
 int col=blockIdx.x*TILE+threadIdx.x;
 float val=0.0f;
 for(int t=0;t<(N+TILE-1)/TILE;t++){
   if(row<M && t*TILE+threadIdx.x<N)
     As[threadIdx.y][threadIdx.x]=A[row*N+t*TILE+threadIdx.x];
   else As[threadIdx.y][threadIdx.x]=0.0f;

   if(col<K && t*TILE+threadIdx.y<N)
     Bs[threadIdx.y][threadIdx.x]=B[(t*TILE+threadIdx.y)*K+col];
   else Bs[threadIdx.y][threadIdx.x]=0.0f;
   __syncthreads();

   for(int i=0;i<TILE;i++)
     val+=As[threadIdx.y][i]*Bs[i][threadIdx.x];
   __syncthreads();
 }
 if(row<M && col<K) C[row*K+col]=val;
}

void fill(float *a,int m,int n){
  for(int i=0;i<m*n;i++) a[i]=(float)(rand()%10);
}

int main(){
 int M=1024,N=1024,K=1024;
 size_t sA=M*N*sizeof(float), sB=N*K*sizeof(float), sC=M*K*sizeof(float);
 float *hA=(float*)malloc(sA);
 float *hB=(float*)malloc(sB);
 float *hC=(float*)malloc(sC);
 fill(hA,M,N); fill(hB,N,K);
 float *dA,*dB,*dC;
 cudaMalloc(&dA,sA); cudaMalloc(&dB,sB); cudaMalloc(&dC,sC);
 cudaMemcpy(dA,hA,sA,cudaMemcpyHostToDevice);
 cudaMemcpy(dB,hB,sB,cudaMemcpyHostToDevice);
 dim3 block(TILE,TILE);
 dim3 grid((K+TILE-1)/TILE,(M+TILE-1)/TILE);

 cudaEvent_t start,stop;
 cudaEventCreate(&start); cudaEventCreate(&stop);

 cudaEventRecord(start);
 matmul_naive<<<grid,block>>>(dA,dB,dC,M,N,K);
 cudaEventRecord(stop);
 cudaEventSynchronize(stop);
 float ms1;
 cudaEventElapsedTime(&ms1,start,stop);
 double gflops1=2.0*M*N*K/(ms1/1000.0)/1e9;
 printf("Naive: %.4f ms  %.2f GFLOPS\n",ms1,gflops1);

 cudaMemset(dC,0,sC);
 cudaEventRecord(start);
 matmul_tiled<<<grid,block>>>(dA,dB,dC,M,N,K);
 cudaEventRecord(stop);
 cudaEventSynchronize(stop);
 float ms2;
 cudaEventElapsedTime(&ms2,start,stop);
 double gflops2=2.0*M*N*K/(ms2/1000.0)/1e9;
 printf("Tiled: %.4f ms  %.2f GFLOPS\n",ms2,gflops2);

 cudaFree(dA); cudaFree(dB); cudaFree(dC);
 free(hA); free(hB); free(hC);
 return 0;
}



//!nvcc -O3 -arch=sm_75 matrix_mul_tiled_prof.cu -o matrix_mul_tiled_prof
//!nvprof ./matrix_mul_tiled_prof
