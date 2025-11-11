# Background:
I have worked on modifying(not fully writing) CUDA kernels for SAiDL and have practiced some questions on LEETGPU, so i had some experience in reading and filling the code but not fully writing kernels from scratch.
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


<img width="934" height="468" alt="image" src="https://github.com/user-attachments/assets/280a4852-6ff4-471a-9e92-438938b5c27e" />


Th einitial profiling was very poor and had certain issues inlcuding no kernels being profiled.



---





