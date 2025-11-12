# Background:


#### I have worked on modifying(not fully writing) CUDA kernels for SAiDL and have practiced some questions on LEETGPU, so i had some experience in reading and filling the code but not fully writing kernels from scratch.
---



# Naive CUDA Matrix Multiplication

## 1. Objective
The first step was to implement a simple matrix multiplier, something i have done in C++ and torch for some previous works, so wasn't very difficult to implement. Took the basic LeetGPU format of code and implemented the basic formula needed.

Implemented a method to multiply any two matrices of dimensions $(A × B)$ and $(B × C)$, which can be formulated as below.

---

## 2. Mathematical Background

For two matrices **A** and **B** of size $N × M$ and $(M × L)$, their product **C = A × B** is defined as:

$$
C[i][j] = \sum_{k=0}^{M-1} A[i][k] \times B[k][j]
$$

Each element $C[i][j]$ represents the **dot product** of the $i^{th}$ row of **A** and the $j^{th}$ column of **B**.

---

## 3. Implementation Concept

In this, each thread is assigned to compute one element of the output matrix C.

The global thread coordinates are computed as below:

$$
\text{row} = \text{blockIdx.y} \times \text{blockDim.y} + \text{threadIdx.y}
$$

$$
\text{col} = \text{blockIdx.x} \times \text{blockDim.x} + \text{threadIdx.x}
$$

Since 2D matrices are flattned in GPU, this formula works.

### Problems faced: 

Switched order of x and y threads over rows and columns.

<img width="1171" height="550" alt="image" src="https://github.com/user-attachments/assets/6bf0e174-ad33-48bf-a141-d38f93850379" />


Just looking at the profiler, it seemed as the gflops were much higher than expected. The GFLOPS for CuBlas infact are around 410 for the same compute, putting my effeciency around 56%, indicating some issues in profiling.

I then tested my kernel in kaggle and other machines too but the GFlops were constantly around 230.

It was also noticed that usage of correct architecture gave number of FLOPS an increase as best result was 300 FLOPS in about 7 seconds.

### Bottleneck:

The kernel shows a clear memory bandwidth bottleneck. About 68% of GPU time (≈9.17 ms) is spent inside matrixMulNaive, while global memory transfers (HtoD + DtoH) take another ~6 ms. Each thread performs N³ = 1024³ ≈ 1.07×10⁹ multiply-adds but also issues roughly 2×N³ = 2.1×10⁹ global memory reads/writes with almost no data reuse. The achieved throughput is ~230 GFLOPS, only around 55% of cuBLAS peak on a T4 GPU in colab. The non-coalesced column access to matrix B and lack of shared-memory tiling makes it heavily memory-bound rather than compute-bound.


After testing CuBlas on my local machine, CuBLas acheived,

**Time: 0.3807 ms**

**cuBLAS SGEMM GFLOPS: 5640.88**

That makes my effeciency to be around 4% of max cuBLAS efficiency .

     
---



## Tiled CUDA Matrix Multiplication  
1. Objective  
After completing the naive version, the next logical step was to optimize memory usage by implementing the well known tilled matrix multiplication using shared memory. The goal was to reduce global memory accesses and exploit data reuse within each block. Tiles or small sub-blocks of matrices, instead of each thread being fetched from entire rows and columns from global memory,  are loaded into shared memory and reused by all threads in the block thus reducing load on bandwidth.

2. Mathematical Background  
The mathematical formulation remains identical to the naive approach:

$C[i][j] = Σ ( A[i][k] × B[k][j] ),  where k = 0 → M−1$

Difference lies in how the computation is divided across threads and how data is fetched from memory. The large matrices A and B are divided into smaller T×T tiles (T=16 here), and each block of threads collaboratively computes one T×T tile of matrix C.  

3. Implementation Concept  
Each thread block is responsible for computing a single tile of the output matrix C.  
Within a block, what happens ?
- Threads first load a T×T chunk of A and a T×T chunk of B into shared memory.  
- Each thread then performs partial multiplication and accumulation on these tiles.  
- After processing all tiles along the shared dimension (N), each thread writes its final computed value to the global memory.  
Also used a fill function to populate the matrices rather than doing them in the main.

Thread indexing remains similar as:  
row = blockIdx.y × TILE + threadIdx.y  
col = blockIdx.x × TILE + threadIdx.x  

Using shared memory allows reuse of matrix elements by 16 threads in a block, reducing redundant global reads by nearly 16x. The synchronization barrier (`__syncthreads()`) ensures all threads finish loading data before computation begins on a tile.  

Problems Faced:  
The main challenge was the implementation, watched some youtube videos and some example implementations and tried to stray a bit further from prior implementations without getting errors. The errors especially during testing of the code with a main function.

Profiler & Performance:  
Compared to the naive kernel (~9.3 ms, 230 GFLOPS), the tiled version runs in about 5.8 ms, achieving ~370 GFLOPS on a T4 GPU. That’s roughly a 1.6x speedup, primarily due to reduced global memory bandwidth pressure and increased data reuse. The shared-memory footprint per block was around 2×16×16×4 bytes = 2 KB, well within hardware limits.  


<img width="1237" height="516" alt="image" src="https://github.com/user-attachments/assets/9344c0ed-a7b4-4e42-91dd-63b564780c03" />

Bottleneck:  

### Global Memory Bandwidth:
Thee kernel still spending the majority of its execution in matmul_tiled.
However:
Even though matmul_tiled dominates total GPU time, it’s still memory bound because:

Its achieved FLOPS are way below peak compute capability.

Global memory bandwidth usage is close to saturation.


### Effeciency:
CuBlas testing showed it having around 5640 FLOPS, with that baseline, Tiled version acheives about: $6.5%$ effeciency as that of CuBlas which seemed very less. 


After trying more implementations i actually got it working around 700GFLOPS, achieving 12% efficiency of cuBLAS kernels.





---

## Register Blocking:

After implementing shared-memory tiling, the next attempt was to squeeze a bit less time doing the memory transfer. I tried to read some other methods to acheive this and saw about Register blocking, it wokrs by optimizing register reuse essentially, reducing shared memory and global memory traffic even further. The idea is to let each thread compute multiple elements of the output tile for example, a 2x2 or 4x4 sub-block(I used 8x8), rather than just one. This technique is known as register blocking, where a small portion of matrix C is kept in registers during the accumulation phase.

The register file in each SM is extremely fast, several times faster than shared memory and reusing data within registers drastically improves arithmetic intensity (the ratio of computation to memory access). The aim was to reduce memory bottlenecks that still persisted in the tiled version


## Mathematical background:
The core formula remains identical, The difference lies entirely in how the computations are batched per thread.
Instead of each thread computing a single $C[i][j]$, each thread now computes multiple neighboring output elements, say $C[i][j], C[i][j+1], C[i+1][j], and C[i+1][j+1]$, in one go.


## Implementation:
Just an added layer on Naive implementation with synchronisation of threads , this time the implementation actually went smoother.
Since i was adding it on the Naive portion, using 2x2 4x4 tiles actually reduced the time and GFLOPS of the process, i was afraid to increase the load as 
registers have low memory, but performance increased with TILE = 8, and decreased with TILE = 8.

## Efficiency:
Register blocking gives another clear jump. Threads compute multiple outputs at once, reusing values directly from registers, the fastest memory on the GPU.
It achieves ~7.7% of cuBLAS, meaning nearly double the efficiency of the naive kernel, and is the first to approach that might be in compute-dominated regime.


---
## Tiled Register Blocking:
Implemented the Register blocking method on the Tiled implementation instead of doing it on Naive implementation. very similar in implementation.


## Efficiency:

This version dramatically improves throughput, roughly 6.9x faster than the naive kernel and achieving ~29 % of cuBLAS performance.
Improvements came from actually implementing the idea along with Tiling.

It achieved **Register-blocked: 1.3339 ms  1609.98 GFLOPS** results. Unfortunately i was unable to run any kind of profiler in my machine due to being stuck in version hell.





