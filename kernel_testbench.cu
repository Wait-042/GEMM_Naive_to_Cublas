#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include <cstdio>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <cuda/pipeline>
#include <cooperative_groups.h>
#include <vector>
#include <functional>
#include <fstream>

#define CEIL(a, b) (((a) + (b) - 1) / (b))
#define FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])
#define CFLOAT4(pointer) (reinterpret_cast<const float4*>(&(pointer))[0])

static void CUDA_CHECK(cudaError_t err) {
    if (err != cudaSuccess) {
        printf("CUDA ERROR: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}


// CPU Naive 矩阵乘法 (Row-Major: C = A * B)
void matrix_multiplication_naive(const float* A, const float* B, float* C, int M, int N, int K) {
    for (int i = 0; i < M * N; i++) {
        C[i] = 0.0f;
    }

    for (int m = 0; m < M; m++) {
        for (int k = 0; k < K; k++) {
            for (int n = 0; n < N; n++) {
                C[m * N + n] += A[m * K + k] * B[k * N + n];
            }
        }
    }
}

// GPU Naive 矩阵乘法
__global__ void gemm_naive_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    int col = blockDim.y * blockIdx.y + threadIdx.y;

    if (row >= M || col >= N) return;

    float c_val = 0.0f;

    for (int i = 0; i < K; i++) {
        c_val += A[row * K + i] * B[i * N + col];
    }
    C[row * N + col] = c_val;
}

// GPU 矩阵乘法合并访存
__global__ void gemm_coalescing_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) return;

    float c_val = 0.0f;

    for (int i = 0; i < K; i++) {
        c_val += A[row * K + i] * B[i * N + col];
    }
    C[row * N + col] = c_val;
}

// GPU 矩阵乘法共享访存
template<const int TILE_SIZE>
__global__ void gemm_smem_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    if (row >= M || col >= N) return;

    // 分配共享内存
    __shared__ float smem_a[TILE_SIZE][TILE_SIZE];
    __shared__ float smem_b[TILE_SIZE][TILE_SIZE];

    float c_val = 0.0f;

    // 沿 K 维度步进加载分块
    for (int s = 0; s < K; s += TILE_SIZE) {
        // 协同加载：每个线程负责搬运一个元素到共享内存
        if (row < M && tx + s < K) {
            smem_a[ty][tx] = A[row * K + s + tx];
        }
        else {
            smem_a[ty][tx] = 0.0f;
        }

        if (ty + s < K && col < N) {
            smem_b[ty][tx] = B[(ty + s) * N + col];
        }
        else {
            smem_b[ty][tx] = 0.0f;
        }
        // 同步，确保整个 Block 的数据都已加载完毕
        __syncthreads();

        // 在极低延迟的共享内存中完成当前 Tile 的乘加计算
#pragma unroll
        for (int i = 0; i < TILE_SIZE; i++) {
            c_val += smem_a[ty][i] * smem_b[i][tx];
        }

        // 同步，防止进入下一个 step 时，有线程过早覆盖掉共享内存
        __syncthreads();
    }

    C[row * N + col] = c_val;

}

// GPU 矩阵乘法 1D分块
template<const int BM, const int BN, const int BK, const int TM>
__global__ void gemm_tile1d_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int row = blockDim.y * by + ty;
    int col = blockDim.x * bx + tx;

    if (row >= M || col >= N) return;

    // 分配共享内存
    __shared__ float a_smem[BM][BK];
    __shared__ float b_smem[BK][BN];
    float c_val[TM] = { 0.0f };

    // 沿 K 维度步进加载分块
    for (int s = 0; s < K; s += BK) {
        // 协同加载：每个线程负责搬运一个元素到共享内存

#pragma unroll
        for (int m = 0; m < TM; m++) {
            int as_row = ty * TM + m;
            int as_col = tx;
            int a_row = by * BM + as_row;
            int a_col = s + as_col;

            if (a_row < M && a_col < K) {
                a_smem[as_row][as_col] = A[a_row * K + a_col];
            }
            else {
                a_smem[as_row][as_col] = 0.0f;
            }
        }

        int bs_row = ty;
        int bs_col = tx;
        int b_row = s + bs_row;
        int b_col = bx * BN + tx;
        if (b_row < K && b_col < N) {
            b_smem[bs_row][bs_col] = B[b_row * N + b_col];
        }
        else {
            b_smem[bs_row][bs_col] = 0.0f;
        }

        __syncthreads();

        // 在极低延迟的共享内存中完成当前 Tile 的乘加计算
#pragma unroll
        for (int k = 0; k < BK; k++) {
#pragma unroll
            for (int m = 0; m < TM; m++) {
                c_val[m] += a_smem[ty * TM + m][k] * b_smem[k][tx];
            }
        }

        // 同步，防止进入下一个 step 时，有线程过早覆盖掉共享内存
        __syncthreads();
    }

#pragma unroll
    for (int m = 0; m < TM; m++) {
        int c_row = by * BM + ty * TM + m;
        int c_col = bx * BN + tx;
        if (c_row < M && c_col < N) {
            C[c_row * N + c_col] = c_val[m];
        }
    }

}

// GPU 矩阵乘法 2D分块
template<const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void gemm_tile2d_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int row = blockDim.y * by + ty;
    int col = blockDim.x * bx + tx;

    if (row >= M || col >= N) return;

    // 分配共享内存
    __shared__ float a_smem[BM][BK];
    __shared__ float b_smem[BK][BN];
    float c_val[TM][TN] = { {0.0f} };

    // 沿 K 维度步进加载分块
    for (int s = 0; s < K; s += BK) {
        // 协同加载：每个线程负责搬运一个元素到共享内存

#pragma unroll
        for (int m = 0; m < TM; m++) {
            int as_row = ty * TM + m;
            int as_col = tx;
            int a_row = by * BM + as_row;
            int a_col = s + as_col;

            if (a_row < M && a_col < K) {
                a_smem[as_row][as_col] = A[a_row * K + a_col];
            }
            else {
                a_smem[as_row][as_col] = 0.0f;
            }
        }

#pragma unroll
        for (int n = 0; n < TN; n++) {
            int bs_row = ty;
            int bs_col = tx * TN + n;
            int b_row = s + bs_row;
            int b_col = bx * BN + bs_col;
            if (b_row < K && b_col < N) {
                b_smem[bs_row][bs_col] = B[b_row * N + b_col];
            }
            else {
                b_smem[bs_row][bs_col] = 0.0f;
            }
        }
        __syncthreads();

        // 在极低延迟的共享内存中完成当前 Tile 的乘加计算
#pragma unroll
        for (int k = 0; k < BK; k++) {
#pragma unroll
            for (int m = 0; m < TM; m++) {
                int as_row = ty * TM + m;
#pragma unroll
                for (int n = 0; n < TN; n++) {
                    int bs_col = tx * TN + n;
                    c_val[m][n] += a_smem[as_row][k] * b_smem[k][bs_col];
                }
            }
        }

        // 同步，防止进入下一个 step 时，有线程过早覆盖掉共享内存
        __syncthreads();
    }

#pragma unroll
    for (int m = 0; m < TM; m++) {
        int c_row = by * BM + ty * TM + m;
#pragma unroll
        for (int n = 0; n < TN; n++) {
            int c_col = bx * BN + tx * TN + n;
            if (c_row < M && c_col < N) {
                C[c_row * N + c_col] = c_val[m][n];
            }
        }
    }

}

// GPU 矩阵乘法 寄存器复用
template<const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void gemm_register_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int row = blockDim.y * by + ty;
    int col = blockDim.x * bx + tx;

    if (row >= M || col >= N) return;

    // 分配共享内存
    __shared__ float a_smem[BM][BK];
    __shared__ float b_smem[BK][BN];
    float c_val[TM][TN] = { {0.0f} };
    float reg_a[TM] = { 0.0f };
    float reg_b[TN] = { 0.0f };

    // 沿 K 维度步进加载分块
    for (int s = 0; s < K; s += BK) {
        // 协同加载：每个线程负责搬运一个元素到共享内存

#pragma unroll
        for (int m = 0; m < TM; m++) {
            int as_row = ty * TM + m;
            int as_col = tx;
            int a_row = by * BM + as_row;
            int a_col = s + as_col;

            if (a_row < M && a_col < K) {
                a_smem[as_row][as_col] = A[a_row * K + a_col];
            }
            else {
                a_smem[as_row][as_col] = 0.0f;
            }
        }

#pragma unroll
        for (int n = 0; n < TN; n++) {
            int bs_row = ty;
            int bs_col = tx * TN + n;
            int b_row = s + bs_row;
            int b_col = bx * BN + bs_col;
            if (b_row < K && b_col < N) {
                b_smem[bs_row][bs_col] = B[b_row * N + b_col];
            }
            else {
                b_smem[bs_row][bs_col] = 0.0f;
            }
        }
        __syncthreads();

        // 在极低延迟的共享内存中完成当前 Tile 的乘加计算
#pragma unroll
        for (int k = 0; k < BK; k++) {
#pragma unroll
            for (int m = 0; m < TM; m++) {
                reg_a[m] = a_smem[ty * TM + m][k];
            }
#pragma unroll
            for (int n = 0; n < TN; n++) {
                reg_b[n] = b_smem[k][tx * TN + n];
            }

#pragma unroll
            for (int m = 0; m < TM; m++) {
#pragma unroll
                for (int n = 0; n < TN; n++) {
                    c_val[m][n] += reg_a[m] * reg_b[n];
                }
            }
        }

        // 同步，防止进入下一个 step 时，有线程过早覆盖掉共享内存
        __syncthreads();
    }

#pragma unroll
    for (int m = 0; m < TM; m++) {
        int c_row = by * BM + ty * TM + m;
#pragma unroll
        for (int n = 0; n < TN; n++) {
            int c_col = bx * BN + tx * TN + n;
            if (c_row < M && c_col < N) {
                C[c_row * N + c_col] = c_val[m][n];
            }
        }
    }

}

// GPU 矩阵乘法 向量化访存
template<const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void gemm_float4_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tid = ty * blockDim.x + tx;

    int a_smem_row = tid >> 1;
    int a_smem_col = (tid & 1) << 2;
    int b_smem_row = tid >> 5;
    int b_smem_col = (tid & 31) << 2;

    int a_row = by * BM + a_smem_row;
    int b_col = bx * BN + b_smem_col;

    // 分配共享内存
    __shared__ float a_smem[BM][BK];
    __shared__ float b_smem[BK][BN];
    float c_val[TM][TN] = { {0.0f} };
    float reg_a[TM] = { 0.0f };
    float reg_b[TN] = { 0.0f };

    // 沿 K 维度步进加载分块
    for (int s = 0; s < K; s += BK) {
        // 协同加载：每个线程负责搬运一个元素到共享内存
        int a_col = s + a_smem_col;
        FLOAT4(a_smem[a_smem_row][a_smem_col]) = CFLOAT4(A[a_row * K + a_col]);

        int b_row = s + b_smem_row;
        FLOAT4(b_smem[b_smem_row][b_smem_col]) = CFLOAT4(B[b_row * N + b_col]);

        __syncthreads();

        // 在极低延迟的共享内存中完成当前 Tile 的乘加计算
#pragma unroll
        for (int k = 0; k < BK; k++) {
#pragma unroll
            for (int m = 0; m < TM; m++) {
                reg_a[m] = a_smem[ty * TM + m][k];
            }
#pragma unroll
            for (int n = 0; n < TN>>2; n++) {
                FLOAT4(reg_b[n << 2]) = FLOAT4(b_smem[k][tx * TN + (n << 2)]);
            }

#pragma unroll
            for (int m = 0; m < TM; m++) {
#pragma unroll
                for (int n = 0; n < TN; n++) {
                    c_val[m][n] += reg_a[m] * reg_b[n];
                }
            }
        }

        // 同步，防止进入下一个 step 时，有线程过早覆盖掉共享内存
        __syncthreads();
    }

#pragma unroll
    for (int m = 0; m < TM; m++) {
        int c_row = by * BM + ty * TM + m;
#pragma unroll
        for (int n = 0; n < TN>>2; n++) {
            int c_col = bx * BN + tx * TN + (n<<2);
            if (c_row < M && c_col < N) {
                FLOAT4(C[c_row * N + c_col]) = FLOAT4(c_val[m][n<<2]);
            }
        }
    }

}

// GPU 矩阵乘法 消除bank conflict
template<const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void gemm_without_bankconflict_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tid = ty * blockDim.x + tx;

    int a_smem_row = tid >> 1;
    int a_smem_col = (tid & 1) << 2;
    int b_smem_row = tid >> 5;
    int b_smem_col = (tid & 31) << 2;

    int a_row = by * BM + a_smem_row;
    int b_col = bx * BN + b_smem_col;

    int num_f4_tm = TM / 4;
    int num_f4_tn = TN / 4;

    // 分配共享内存
    __shared__ float a_smem[BK][BM];
    __shared__ float b_smem[BK][BN];
    float c_val[TM][TN] = { {0.0f} };
    float reg_a[TM] = { 0.0f };
    float reg_b[TN] = { 0.0f };

    float a_tmp[4];
    // 沿 K 维度步进加载分块
    for (int s = 0; s < K; s += BK) {
        // 协同加载：每个线程负责搬运一个元素到共享内存
        int a_col = s + a_smem_col;
        FLOAT4(a_tmp) = CFLOAT4(A[a_row * K + a_col]);
        a_smem[a_smem_col][a_smem_row] = a_tmp[0];
        a_smem[a_smem_col + 1][a_smem_row] = a_tmp[1];
        a_smem[a_smem_col + 2][a_smem_row] = a_tmp[2];
        a_smem[a_smem_col + 3][a_smem_row] = a_tmp[3];


        int b_row = s + b_smem_row;
        FLOAT4(b_smem[b_smem_row][b_smem_col]) = CFLOAT4(B[b_row * N + b_col]);

        __syncthreads();

        // 在极低延迟的共享内存中完成当前 Tile 的乘加计算
#pragma unroll
        for (int k = 0; k < BK; k++) {
#pragma unroll
            for (int m = 0; m < TM >> 2; m++) {
                FLOAT4(reg_a[m << 2]) = FLOAT4(a_smem[k][ty * TM / num_f4_tm + m * BM / num_f4_tm]);
            }
#pragma unroll
            for (int n = 0; n < TN >> 2; n++) {
                FLOAT4(reg_b[n << 2]) = FLOAT4(b_smem[k][tx * TN / num_f4_tn + n * BN / num_f4_tn]);
            }

#pragma unroll
            for (int m = 0; m < TM; m++) {
#pragma unroll
                for (int n = 0; n < TN; n++) {
                    c_val[m][n] += reg_a[m] * reg_b[n];
                }
            }
        }

        // 同步，防止进入下一个 step 时，有线程过早覆盖掉共享内存
        __syncthreads();
    }

    //将 128 * 128 的块拆分为 4 个 64 * 64 的子块进行写回
#pragma unroll
    for (int m = 0; m < TM / 2; m++) {
        int c_row = by * BM + ty * TM / 2 + m;
        int c_col = bx * BN + tx * TN / 2;
        // 1. 左上子块 (Sub-tile 0,0)
        if (c_row < M && c_col < N) {
            FLOAT4(C[c_row * N + c_col]) = FLOAT4(c_val[m][0]);
        }
        // 2. 右上子块 (Sub-tile 0,1)
        if (c_row < M && c_col + (BN / 2) < N) {
            FLOAT4(C[c_row * N + c_col + (BN / 2)]) = FLOAT4(c_val[m][4]);
        }
    }

#pragma unroll
    for (int m = 0; m < TM / 2; m++) {
        int c_row = by * BM + ty * TM / 2 + m + BM / 2;
        int c_col = bx * BN + tx * TN / 2;
        // 3. 左下子块 (Sub-tile 1,0)
        if (c_row < M && c_col < N) {
            FLOAT4(C[c_row * N + c_col]) = FLOAT4(c_val[m + 4][0]);
        }
        // 4. 右下子块 (Sub-tile 1,1)
        if (c_row < M && c_col + (BN / 2) < N) {
            FLOAT4(C[c_row * N + c_col + (BN / 2)]) = FLOAT4(c_val[m + 4][4]);
        }
    }
}


// GPU 矩阵乘法 双缓冲
template<const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void gemm_double_buffer_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tid = ty * blockDim.x + tx;

    int a_smem_row = tid >> 1;
    int a_smem_col = (tid & 1) << 2;
    int b_smem_row = tid >> 5;
    int b_smem_col = (tid & 31) << 2;

    int a_row = by * BM + a_smem_row;
    int b_col = bx * BN + b_smem_col;

    int num_f4_tm = TM / 4;
    int num_f4_tn = TN / 4;
    int write_idx = 0;

    // 分配共享内存
    __shared__ float a_smem[2][BK][BM];
    __shared__ float b_smem[2][BK][BN];
    float c_val[TM][TN] = { {0.0f} };
    float reg_a[2][TM] = { 0.0f };
    float reg_b[2][TN] = { 0.0f };

    float a_tmp[4];
    // 沿 K 维度步进加载分块
    for (int s = 0; s < K; s += BK) {
        // 协同加载：每个线程负责搬运一个元素到共享内存
        int a_col = s + a_smem_col;
        FLOAT4(a_tmp) = CFLOAT4(A[a_row * K + a_col]);
        a_smem[write_idx][a_smem_col][a_smem_row] = a_tmp[0];
        a_smem[write_idx][a_smem_col + 1][a_smem_row] = a_tmp[1];
        a_smem[write_idx][a_smem_col + 2][a_smem_row] = a_tmp[2];
        a_smem[write_idx][a_smem_col + 3][a_smem_row] = a_tmp[3];


        int b_row = s + b_smem_row;
        FLOAT4(b_smem[write_idx][b_smem_row][b_smem_col]) = CFLOAT4(B[b_row * N + b_col]);

        __syncthreads();

        // 在极低延迟的共享内存中完成当前 Tile 的乘加计算
#pragma unroll
        for (int k = 0; k < BK; k++) {
#pragma unroll
            for (int m = 0; m < TM >> 2; m++) {
                FLOAT4(reg_a[write_idx][m << 2]) = FLOAT4(a_smem[write_idx][k][ty * TM / num_f4_tm + m * BM / num_f4_tm]);
            }
#pragma unroll
            for (int n = 0; n < TN >> 2; n++) {
                FLOAT4(reg_b[write_idx][n << 2]) = FLOAT4(b_smem[write_idx][k][tx * TN / num_f4_tn + n * BN / num_f4_tn]);
            }

#pragma unroll
            for (int m = 0; m < TM; m++) {
#pragma unroll
                for (int n = 0; n < TN; n++) {
                    c_val[m][n] += reg_a[write_idx][m] * reg_b[write_idx][n];
                }
            }
        }

        write_idx ^= 1;
    }

    //将 128 * 128 的块拆分为 4 个 64 * 64 的子块进行写回
#pragma unroll
    for (int m = 0; m < TM / 2; m++) {
        int c_row = by * BM + ty * TM / 2 + m;
        int c_col = bx * BN + tx * TN / 2;
        // 1. 左上子块 (Sub-tile 0,0)
        if (c_row < M && c_col < N) {
            FLOAT4(C[c_row * N + c_col]) = FLOAT4(c_val[m][0]);
        }
        // 2. 右上子块 (Sub-tile 0,1)
        if (c_row < M && c_col + (BN / 2) < N) {
            FLOAT4(C[c_row * N + c_col + (BN / 2)]) = FLOAT4(c_val[m][4]);
        }
    }

#pragma unroll
    for (int m = 0; m < TM / 2; m++) {
        int c_row = by * BM + ty * TM / 2 + m + BM / 2;
        int c_col = bx * BN + tx * TN / 2;
        // 3. 左下子块 (Sub-tile 1,0)
        if (c_row < M && c_col < N) {
            FLOAT4(C[c_row * N + c_col]) = FLOAT4(c_val[m + 4][0]);
        }
        // 4. 右下子块 (Sub-tile 1,1)
        if (c_row < M && c_col + (BN / 2) < N) {
            FLOAT4(C[c_row * N + c_col + (BN / 2)]) = FLOAT4(c_val[m + 4][4]);
        }
    }

}

// PTX 16字节 (128-bit) 异步拷贝宏：直接从 Global 到 Shared Memory
__device__ __forceinline__ void cp_async_cg(void* smem_ptr, const void* glob_ptr) {
    unsigned int smem_addr = __cvta_generic_to_shared(smem_ptr);
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], 16;\n"
        :: "r"(smem_addr), "l"(glob_ptr)
        );
}

// 异步管道提交与等待宏
__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n");
}

template<int N_GROUPS>
__device__ __forceinline__ void cp_async_wait_group() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N_GROUPS));
}

// GPU 矩阵乘法 异步拷贝
template<const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void gemm_async_kernel(const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tid = ty * blockDim.x + tx;

    int a_smem_row = tid >> 1;
    int a_smem_col = (tid & 1) << 2;
    int b_smem_row = tid >> 5;
    int b_smem_col = (tid & 31) << 2;

    int a_row = by * BM + a_smem_row;
    int b_col = bx * BN + b_smem_col;

    int num_f4_tm = TM / 4;
    int num_f4_tn = TN / 4;

    // 共享内存布局: [缓冲层][K维度][M/N维度]
    __shared__ float a_smem[2][BK][BM];
    __shared__ float b_smem[2][BK][BN];

    float c_val[TM][TN] = { {0.0f} };
    float reg_a[TM] = { 0.0f };
    float reg_b[TN] = { 0.0f };
    float a_tmp[4];

    // -----------------------------------------------------------------
    // 1. Prologue: 预加载第 0 个 Tile
    // -----------------------------------------------------------------
    // A 矩阵: 采用寄存器中转向量化转置写入 smem
    int a_col_0 = 0 + a_smem_col;
    if (a_row < M && a_col_0 < K) {
        FLOAT4(a_tmp) = CFLOAT4(A[a_row * K + a_col_0]);
        a_smem[0][a_smem_col][a_smem_row] = a_tmp[0];
        a_smem[0][a_smem_col + 1][a_smem_row] = a_tmp[1];
        a_smem[0][a_smem_col + 2][a_smem_row] = a_tmp[2];
        a_smem[0][a_smem_col + 3][a_smem_row] = a_tmp[3];
    }

    // B 矩阵: 内存连续，使用硬件 cp.async 16字节零寄存器搬运
    int b_row_0 = 0 + b_smem_row;
    if (b_row_0 < K && b_col < N) {
        cp_async_cg(&b_smem[0][b_smem_row][b_smem_col], &B[b_row_0 * N + b_col]);
    }
    cp_async_commit();
    cp_async_wait_group<0>(); // 确保 第 0 个 Tile 到位
    __syncthreads();

    int write_idx = 1;
    int read_idx = 0;

    // -----------------------------------------------------------------
    // 2. Main Loop: 异步重叠计算与预取
    // -----------------------------------------------------------------
    for (int s = BK; s < K; s += BK) {
        // [步骤 A] 发起下一个 Tile (write_idx) 的预取
        int a_col = s + a_smem_col;
        if (a_row < M && a_col < K) {
            FLOAT4(a_tmp) = CFLOAT4(A[a_row * K + a_col]);
            a_smem[write_idx][a_smem_col][a_smem_row] = a_tmp[0];
            a_smem[write_idx][a_smem_col + 1][a_smem_row] = a_tmp[1];
            a_smem[write_idx][a_smem_col + 2][a_smem_row] = a_tmp[2];
            a_smem[write_idx][a_smem_col + 3][a_smem_row] = a_tmp[3];
        }

        int b_row = s + b_smem_row;
        if (b_row < K && b_col < N) {
            cp_async_cg(&b_smem[write_idx][b_smem_row][b_smem_col], &B[b_row * N + b_col]);
        }
        cp_async_commit(); // 提交异步队列

        // [步骤 B] 计算当前 Tile (read_idx)
#pragma unroll
        for (int k = 0; k < BK; k++) {
#pragma unroll
            for (int m = 0; m < TM >> 2; m++) {
                FLOAT4(reg_a[m << 2]) = FLOAT4(a_smem[read_idx][k][ty * TM / num_f4_tm + m * BM / num_f4_tm]);
            }
#pragma unroll
            for (int n = 0; n < TN >> 2; n++) {
                FLOAT4(reg_b[n << 2]) = FLOAT4(b_smem[read_idx][k][tx * TN / num_f4_tn + n * BN / num_f4_tn]);
            }

#pragma unroll
            for (int m = 0; m < TM; m++) {
#pragma unroll
                for (int n = 0; n < TN; n++) {
                    c_val[m][n] += reg_a[m] * reg_b[n];
                }
            }
        }

        // [步骤 C] 等待 write_idx 的 B 矩阵数据到达，准备下一次循环
        cp_async_wait_group<0>();
        __syncthreads();

        write_idx ^= 1;
        read_idx ^= 1;
    }

    // -----------------------------------------------------------------
    // 3. Epilogue: 计算最后一个 Tile
    // -----------------------------------------------------------------
#pragma unroll
    for (int k = 0; k < BK; k++) {
#pragma unroll
        for (int m = 0; m < TM >> 2; m++) {
            FLOAT4(reg_a[m << 2]) = FLOAT4(a_smem[read_idx][k][ty * TM / num_f4_tm + m * BM / num_f4_tm]);
        }
#pragma unroll
        for (int n = 0; n < TN >> 2; n++) {
            FLOAT4(reg_b[n << 2]) = FLOAT4(b_smem[read_idx][k][tx * TN / num_f4_tn + n * BN / num_f4_tn]);
        }

#pragma unroll
        for (int m = 0; m < TM; m++) {
#pragma unroll
            for (int n = 0; n < TN; n++) {
                c_val[m][n] += reg_a[m] * reg_b[n];
            }
        }
    }

    // -----------------------------------------------------------------
    // 4. 写回 C 矩阵
    // -----------------------------------------------------------------
#pragma unroll
    for (int m = 0; m < TM / 2; m++) {
        int c_row = by * BM + ty * TM / 2 + m;
        int c_col = bx * BN + tx * TN / 2;
        if (c_row < M && c_col < N) {
            FLOAT4(C[c_row * N + c_col]) = FLOAT4(c_val[m][0]);
        }
        if (c_row < M && c_col + (BN / 2) < N) {
            FLOAT4(C[c_row * N + c_col + (BN / 2)]) = FLOAT4(c_val[m][4]);
        }
    }

#pragma unroll
    for (int m = 0; m < TM / 2; m++) {
        int c_row = by * BM + ty * TM / 2 + m + BM / 2;
        int c_col = bx * BN + tx * TN / 2;
        if (c_row < M && c_col < N) {
            FLOAT4(C[c_row * N + c_col]) = FLOAT4(c_val[m + 4][0]);
        }
        if (c_row < M && c_col + (BN / 2) < N) {
            FLOAT4(C[c_row * N + c_col + (BN / 2)]) = FLOAT4(c_val[m + 4][4]);
        }
    }
}

#define BLOCK_SIZE 8
__global__ void transpose_kernel(const float* input, float* output, int rows, int cols) {
    __shared__ float scratch[BLOCK_SIZE][BLOCK_SIZE + 1];
    const int in_x = threadIdx.x + BLOCK_SIZE * blockIdx.x;
    const int in_y = threadIdx.y + BLOCK_SIZE * blockIdx.y;

    const int in_cols = cols;

    const int input_idx = in_x + in_cols * in_y;
    if (input_idx < rows * cols) {
        scratch[threadIdx.y][threadIdx.x] = input[input_idx];
    }
    __syncthreads();
    const int out_y = threadIdx.y + BLOCK_SIZE * blockIdx.x;
    const int out_x = threadIdx.x + BLOCK_SIZE * blockIdx.y;
    const int out_cols = rows;
    const int out_rows = cols;
    if (out_x < out_cols && out_y < out_rows) {
        output[out_x + out_cols * out_y] = scratch[threadIdx.x][threadIdx.y];
    }
}

template<const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void gemm_async_opt_kernel(
    const float* __restrict__ A_T, // 已经转置的 A (K * M)
    const float* __restrict__ B,   // 原始 B (K * N)
    float* __restrict__ C,
    int M, int N, int K) {

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tid = threadIdx.x;
    int tx = tid & 15;
    int ty = tid >> 4;

    int a_smem_row = tid >> 5;               // 0-7
    int a_smem_col = (tid & 31) << 2;         // 0-124
    int b_smem_row = tid >> 5;               // 0-7
    int b_smem_col = (tid & 31) << 2;        // 0-124

    // 转置后 A_T 的索引 (行是 K，列是 M)
    int a_global_m = by * BM + a_smem_col;
    int a_global_k = a_smem_row;

    int b_global_k = b_smem_row;
    int b_global_n = bx * BN + b_smem_col;

    int num_f4_tm = TM / 4;
    int num_f4_tn = TN / 4;

    // 增加 Padding 彻底消除 Bank Conflict
    __shared__ float a_smem[2][BK][BM];
    __shared__ float b_smem[2][BK][BN];

    float c_val[TM][TN] = { {0.0f} };
    float reg_a[TM] = { 0.0f };
    float reg_b[TN] = { 0.0f };

    // ---------------- 1. Prologue ----------------
    // 预加载第 0 块 (k=0)
    if (a_global_m < M && a_global_k < K) {
        cp_async_cg(&a_smem[0][a_smem_row][a_smem_col], &A_T[a_global_k * M + a_global_m]);
        cp_async_cg(&a_smem[0][a_smem_row + 8][a_smem_col], &A_T[(a_global_k + 8)*M + a_global_m]);
    }
    if (b_global_k < K && b_global_n < N) {
        cp_async_cg(&b_smem[0][b_smem_row][b_smem_col], &B[b_global_k * N + b_global_n]);
        cp_async_cg(&b_smem[0][b_smem_row + 8][b_smem_col], &B[(b_global_k + 8)*N + b_global_n]);
    }
    cp_async_commit();
    cp_async_wait_group<0>(); // 等待所有数据到位
    __syncthreads();

    int write_idx = 1;
    int read_idx = 0;

    // ---------------- 2. Main Loop ----------------
    for (int s = BK; s < K; s += BK) {
        // [步骤 A] 发起下一个 Tile 的预取 (A_T 和 B 均使用异步拷贝)
        if (a_global_m < M && (s + a_smem_row) < K) {
            cp_async_cg(&a_smem[write_idx][a_smem_row][a_smem_col], &A_T[(s + a_smem_row) * M + a_global_m]);
            cp_async_cg(&a_smem[write_idx][a_smem_row + 8][a_smem_col], &A_T[(s + a_smem_row + 8) * M + a_global_m]);
        }
        if ((s + b_smem_row) < K && b_global_n < N) {
            cp_async_cg(&b_smem[write_idx][b_smem_row][b_smem_col], &B[(s + b_smem_row) * N + b_global_n]);
            cp_async_cg(&b_smem[write_idx][b_smem_row + 8][b_smem_col], &B[(s + b_smem_row + 8) * N + b_global_n]);
        }
        cp_async_commit();

        // [步骤 B] 计算当前 Tile (read_idx)
#pragma unroll
        for (int k = 0; k < BK; k++) {
#pragma unroll
            for (int m = 0; m < TM >> 2; m++) {
                FLOAT4(reg_a[m << 2]) = FLOAT4(a_smem[read_idx][k][ty * TM / num_f4_tm + m * BM / num_f4_tm]);
            }
#pragma unroll
            for (int n = 0; n < TN >> 2; n++) {
                FLOAT4(reg_b[n << 2]) = FLOAT4(b_smem[read_idx][k][tx * TN / num_f4_tn + n * BN / num_f4_tn]);
            }

#pragma unroll
            for (int m = 0; m < TM; m++) {
#pragma unroll
                for (int n = 0; n < TN; n++) {
                    c_val[m][n] += reg_a[m] * reg_b[n];
                }
            }
        }

        // [步骤 C] 等待之前提交的数据安全到达
        cp_async_wait_group<1>(); // 等待除了最后提交之外的所有组
        __syncthreads();

        write_idx ^= 1;
        read_idx ^= 1;
    }

    // ---------------- 3. Epilogue: 计算最后一个 Tile ----------------
#pragma unroll
    for (int k = 0; k < BK; k++) {
#pragma unroll
        for (int m = 0; m < TM >> 2; m++) {
            FLOAT4(reg_a[m << 2]) = FLOAT4(a_smem[read_idx][k][ty * TM / num_f4_tm + m * BM / num_f4_tm]);
        }
#pragma unroll
        for (int n = 0; n < TN >> 2; n++) {
            FLOAT4(reg_b[n << 2]) = FLOAT4(b_smem[read_idx][k][tx * TN / num_f4_tn + n * BN / num_f4_tn]);
        }
#pragma unroll
        for (int m = 0; m < TM; m++) {
#pragma unroll
            for (int n = 0; n < TN; n++) {
                c_val[m][n] += reg_a[m] * reg_b[n];
            }
        }
    }

    // ---------------- 4. 写回 C 矩阵 (与原代码一致) ----------------
#pragma unroll
    for (int m = 0; m < TM / 2; m++) {
        int c_row = by * BM + ty * TM / 2 + m;
        int c_col = bx * BN + tx * TN / 2;
        if (c_row < M && c_col < N) {
            FLOAT4(C[c_row * N + c_col]) = FLOAT4(c_val[m][0]);
        }
        if (c_row < M && c_col + (BN / 2) < N) {
            FLOAT4(C[c_row * N + c_col + (BN / 2)]) = FLOAT4(c_val[m][4]);
        }
    }
#pragma unroll
    for (int m = 0; m < TM / 2; m++) {
        int c_row = by * BM + ty * TM / 2 + m + BM / 2;
        int c_col = bx * BN + tx * TN / 2;
        if (c_row < M && c_col < N) {
            FLOAT4(C[c_row * N + c_col]) = FLOAT4(c_val[m + 4][0]);
        }
        if (c_row < M && c_col + (BN / 2) < N) {
            FLOAT4(C[c_row * N + c_col + (BN / 2)]) = FLOAT4(c_val[m + 4][4]);
        }
    }
}


__global__ void max_error_kernel(const float* C, const float* C_ref, int n, float* max_err) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float err = fabsf(C[idx] - C_ref[idx]);
        // 将非负 float 按位视为 unsigned int，利用 atomicMax 获取最大值
        unsigned int err_bits = __float_as_uint(err);
        atomicMax((unsigned int*)max_err, err_bits);
    }
}

void gemm_naive(float* A, float* B, float* C, int M, int N, int K, cudaStream_t stream) {
    dim3 threads(16, 16);
    dim3 blocks(CEIL(N, threads.x), CEIL(M, threads.y));
    gemm_naive_kernel << <blocks, threads, 0, stream >> > (A, B, C, M, N, K);
}

void gemm_coalescing(float* A, float* B, float* C, int M, int N, int K, cudaStream_t stream) {
    const int TILE_SIZE = 16;
    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks(CEIL(N, TILE_SIZE), CEIL(M, TILE_SIZE));
    gemm_coalescing_kernel<< <blocks, threads, 0, stream >> > (A, B, C, M, N, K);
}

void gemm_smem(float* A, float* B, float* C, int M, int N, int K, cudaStream_t stream) {
    const int TILE_SIZE = 16;
    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks(CEIL(N, threads.x), CEIL(M, threads.y));
    gemm_smem_kernel<TILE_SIZE> << <blocks, threads, 0, stream >> > (A, B, C, M, N, K);
}

void gemm_tile1d(float* A, float* B, float* C, int M, int N, int K, cudaStream_t stream) {
    const int TILE_SIZE = 16;
    const int BM = 128; // A矩阵Tile行维度
    const int BN = 16; // B矩阵Tile列维度
    const int BK = 16; // A矩阵Tile列维度
    const int TM = 8; // M维度方向单线程处理元素数
    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks(CEIL(N, BN), CEIL(M, BM));
    gemm_tile1d_kernel <BM, BN, BK, TM> << <blocks, threads, 0, stream >> > (A, B, C, M, N, K);
}

void gemm_tile2d(float* A, float* B, float* C, int M, int N, int K, cudaStream_t stream) {
    const int TILE_SIZE = 16;
    const int BM = 128;  // A矩阵Tile行维度
    const int BN = 128;  // B矩阵Tile列维度
    const int BK = 16; // A矩阵Tile列维度
    const int TM = 8;  // M维度方向单线程处理元素数
    const int TN = 8;  // N维度方向单线程处理元素数
    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks(CEIL(N, BN), CEIL(M, BM));
    gemm_tile2d_kernel <BM, BN, BK, TM, TN> << <blocks, threads, 0, stream >> > (A, B, C, M, N, K);
}

void gemm_register(float* A, float* B, float* C, int M, int N, int K, cudaStream_t stream) {
    const int TILE_SIZE = 16;
    const int BM = 128;  // A矩阵Tile行维度
    const int BN = 128;  // B矩阵Tile列维度
    const int BK = 16; // A矩阵Tile列维度
    const int TM = 8;  // M维度方向单线程处理元素数
    const int TN = 8;  // N维度方向单线程处理元素数
    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks(CEIL(N, BN), CEIL(M, BM));
    gemm_register_kernel <BM, BN, BK, TM, TN> << <blocks, threads, 0, stream >> > (A, B, C, M, N, K);
}

void gemm_float4(float* A, float* B, float* C, int M, int N, int K, cudaStream_t stream) {
    const int TILE_SIZE = 16;
    const int BM = 128;
    const int BN = 128;
    const int BK = 8;
    const int TM = 8;
    const int TN = 8;
    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks(CEIL(N, BN), CEIL(M, BM));
    gemm_float4_kernel <BM, BN, BK, TM, TN> << <blocks, threads, 0, stream >> > (A, B, C, M, N, K);
}

void gemm_without_bankconflict(float* A, float* B, float* C, int M, int N, int K, cudaStream_t stream) {
    const int TILE_SIZE = 16;
    const int BM = 128;
    const int BN = 128;
    const int BK = 8;
    const int TM = 8;
    const int TN = 8;
    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks(CEIL(N, BN), CEIL(M, BM));
    gemm_without_bankconflict_kernel <BM, BN, BK, TM, TN> << <blocks, threads, 0, stream >> > (A, B, C, M, N, K);
}

void gemm_double_buffer(float* A, float* B, float* C, int M, int N, int K, cudaStream_t stream) {
    const int TILE_SIZE = 16;
    const int BM = 128;
    const int BN = 128;
    const int BK = 8;
    const int TM = 8;
    const int TN = 8;
    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks(CEIL(N, BN), CEIL(M, BM));
    gemm_double_buffer_kernel <BM, BN, BK, TM, TN> << <blocks, threads, 0, stream >> > (A, B, C, M, N, K);
}

void gemm_async(float* A, float* B, float* C, int M, int N, int K, cudaStream_t stream) {
    const int TILE_SIZE = 16;
    const int BM = 128;
    const int BN = 128;
    const int BK = 8;
    const int TM = 8;
    const int TN = 8;
    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks(CEIL(N, BN), CEIL(M, BM));
    gemm_async_kernel <BM, BN, BK, TM, TN> << <blocks, threads, 0, stream >> > (A, B, C, M, N, K);
}

void transpose(float* A, float* A_T, int M, int K, cudaStream_t stream) {
    dim3 trans_block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 trans_grid(CEIL(K, BLOCK_SIZE), CEIL(M, BLOCK_SIZE));
    transpose_kernel << <trans_grid, trans_block, 0, stream >> > (A, A_T, M, K);
}

void gemm_async_opt(float* A, float* A_T, float* B, float* C, int M, int N, int K, cudaStream_t stream) {
    const int TILE_SIZE = 16;
    const int BM = 128;
    const int BN = 128;
    const int BK = 16;
    const int TM = 8;
    const int TN = 8;
	
	transpose(A, A_T, M, K, stream);
	
    int threads = TILE_SIZE*TILE_SIZE;
    dim3 blocks(CEIL(N, BN), CEIL(M, BM));
    gemm_async_opt_kernel <BM, BN, BK, TM, TN> << <blocks, threads, 0, stream >> > (A_T, B, C, M, N, K);
}

int main() {
    // 测试参数
    const int start_size = 256;
    const int end_size = 6400;
    const int step = 256;
    const int num_runs = 10;  // 每个 kernel 重复次数（尺寸较大时建议设为 1 或 2）

    // 打开输出文件
    std::ofstream outfile("gemm_benchmark.txt");
    if (!outfile.is_open()) {
        std::cerr << "Failed to open output file." << std::endl;
    }

    // 写入 CSV 标题
    outfile << "M,N,K,kernel,time_ms,gflops\n";

    // 创建 CUDA 事件（可复用）
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 创建 cuBLAS 句柄和流（可复用）
    cudaStream_t stream;
    cublasHandle_t handle;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    cublasCreate(&handle);
    cublasSetStream(handle, stream);

    // 循环不同尺寸
    for (int size = start_size; size <= end_size; size += step) {
        int M = size, N = size, K = size;

        std::cout << "Testing size: " << size << std::endl;

        // 主机内存分配
        float* A = new float[M * K];
        float* B = new float[K * N];
        float* C = new float[M * N];   // 仅用于最终检查（可选）
        // 初始化随机数（使用固定种子确保不同尺寸间数据分布一致，也可每次随机）
        srand(42);
        for (int i = 0; i < M * K; i++) A[i] = rand() / (float)RAND_MAX;
        for (int j = 0; j < K * N; j++) B[j] = rand() / (float)RAND_MAX;

        // 设备内存分配
        float* dev_A, * dev_A_T, * dev_B, * dev_C;
        CUDA_CHECK(cudaMalloc((void**)&dev_A, M * K * sizeof(float)));
        CUDA_CHECK(cudaMalloc((void**)&dev_A_T, K * M * sizeof(float)));
        CUDA_CHECK(cudaMalloc((void**)&dev_B, K * N * sizeof(float)));
        CUDA_CHECK(cudaMalloc((void**)&dev_C, M * N * sizeof(float)));

        // 拷贝数据到设备
        CUDA_CHECK(cudaMemcpy(dev_A, A, M * K * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dev_B, B, K * N * sizeof(float), cudaMemcpyHostToDevice));

        float alpha = 1.0f, beta = 0.0f;

        // 定义 kernel 列表（捕获当前尺寸）
        std::vector<std::pair<std::string, std::function<void()>>> kernels = {
            {"gemm_naive",        [&]() { gemm_naive(dev_A, dev_B, dev_C, M, N, K, stream); }},
            {"gemm_coalescing",   [&]() { gemm_coalescing(dev_A, dev_B, dev_C, M, N, K, stream); }},
            {"gemm_smem",         [&]() { gemm_smem(dev_A, dev_B, dev_C, M, N, K, stream); }},
            {"gemm_tile1d",       [&]() { gemm_tile1d(dev_A, dev_B, dev_C, M, N, K, stream); }},
            {"gemm_tile2d",       [&]() { gemm_tile2d(dev_A, dev_B, dev_C, M, N, K, stream); }},
            {"gemm_register",     [&]() { gemm_register(dev_A, dev_B, dev_C, M, N, K, stream); }},
            {"gemm_float4",       [&]() { gemm_float4(dev_A, dev_B, dev_C, M, N, K, stream); }},
            {"gemm_without_bankconflict", [&]() { gemm_without_bankconflict(dev_A, dev_B, dev_C, M, N, K, stream); }},
            {"gemm_double_buffer",[&]() { gemm_double_buffer(dev_A, dev_B, dev_C, M, N, K, stream); }},
            {"gemm_async",        [&]() { gemm_async(dev_A, dev_B, dev_C, M, N, K, stream); }},
            {"gemm_async_opt",    [&]() { gemm_async_opt(dev_A, dev_A_T, dev_B, dev_C, M, N, K, stream); }},
            {"cublasSgemm",       [&]() {
                cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                            N, M, K, &alpha, dev_B, N, dev_A, K, &beta, dev_C, N);
            }},
        };

        // 预热：每个 kernel 运行一次（不记录时间）
        for (auto& k : kernels) {
            k.second();
            cudaStreamSynchronize(stream);
            cudaMemsetAsync(dev_C, 0, M * N * sizeof(float), stream);
        }
        cudaStreamSynchronize(stream);

        // 正式计时
        for (auto& k : kernels) {
            // 判断当前是否是 transpose kernel
            bool is_transpose = (k.first == "transpose");

            // 清零 dev_C（确保从零开始）
            cudaMemsetAsync(dev_C, 0, M * N * sizeof(float), stream);
            cudaStreamSynchronize(stream);

            float total_time = 0.0f;
            for (int run = 0; run < num_runs; ++run) {
                cudaEventRecord(start, stream);
                k.second();
                cudaEventRecord(stop, stream);
                cudaEventSynchronize(stop);
                float ms;
                cudaEventElapsedTime(&ms, start, stop);
                total_time += ms;
            }
            float avg_ms = total_time / num_runs;
            double gflops = 2.0 * M * N * K / (avg_ms * 1e-3) / 1e9;

            // 如果是 transpose，不需要计算误差，直接打印 N/A
            if (is_transpose) {
                outfile << M << "," << N << "," << K << ","
                    << k.first << "," << avg_ms << "," << 0.0f << "\n";
                continue; // 跳过后面的误差计算代码
            }

            // 写入文件
            outfile << M << "," << N << "," << K << ","
                << k.first << "," << avg_ms << "," << gflops << "\n";
        }

        // 释放当前尺寸的内存
        cudaFree(dev_A);
        cudaFree(dev_B);
        cudaFree(dev_C);
        delete[] A;
        delete[] B;
        delete[] C;
    }

    // 清理资源
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaStreamDestroy(stream);
    cublasDestroy(handle);
    outfile.close();

    std::cout << "Benchmark results saved to gemm_benchmark.txt" << std::endl;
	
	return 0;
}

