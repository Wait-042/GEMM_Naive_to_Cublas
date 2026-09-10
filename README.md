# GEMM_Naive_to_Cublas
在此记录自己学习GEMM的过程，尝试手写Float GEMM kernel从naive版本逐步优化使其性能接近cublas，当前手写kernel差不多有94%的cublas性能

## 前言
这里给出我们测试的矩阵乘法形式和维度符号，在后续的代码测试中，为了简化代码，我在代码中并没有做很严谨的边界判断，矩阵尺寸都是4的倍数

$$
C_{M*N} = A_{M*K} * B_{K*N}
$$

## 环境
```
Graphics Card: RTX 4070 Super

Python 3.11

CUDA Version 13.2

CUDAToolkit 12.8

g++(Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0
```

## quick start
- cuda
```
// 调节main函数中参数测试
const int start_size = 256; // 起始测试矩阵尺寸
const int end_size = 256; // 最终测试矩阵尺寸
const int step = 256; // 每次步进矩阵尺寸变大step
const int num_runs = 1;  // 每个 kernel 重复次数（尺寸较大时建议设为 1 或 2）

mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
./app
```

PS: 异步拷贝只在sm80以上架构可使用
CMakeLists.txt文件中需要删除这行，.cu文件也要注意注释掉异步拷贝kernel
```
set(CMAKE_CUDA_ARCHITECTURES 89)
```

- python
前面cuda跑出结果后用python画图
```
pip install -r requirements.txt
python result_plot.py
```

## Kernel优化步骤和结果对比
![gemm_result](https://github.com/Wait-042/GEMM_Naive_to_Cublas/blob/main/fig/gemm_result.png)
- 从Naive版本逐步引入合并访存、共享内存、一维分块、二维分块、寄存器、向量化、bank conflict消除、双缓冲、异步拷贝手段使得手写GEMM kernel
性能接近cublas 94%
- 这里注意到cublas性能波动较大，部分原因是尾部效应，尺寸在2048时，waves per SM = 6.86，而尺寸在3840时 waves per SM = 17.14，SM利用率不够
- 小尺寸时cublas性能远超手写算子是因为我们的kernel参数写死了，导致sm利用率不够，只有少数sm活跃，cublas会动态调整
- gemm_doubel_buffer、gemm_async、gemm_async_opt从尺寸2304开始，随着尺寸变大GFLOPS在降低，这同样有部分是尾部效应的影响，
同时还有L2 cache被击穿影响，因为我们的显卡L2缓存只有48MB、导致尺寸增大后L2 cache命中率会降低，其他手写kernel没有这个情况猜测是瓶颈不在这，
cublas内部有优化策略，不够这块没完全搞懂原因

=============== Relative to cuBLAS (GFLOPS %), M >= 2048 ===============

| Kernel                    | Mean % | Min %  | Max %  | Num Sizes |  
|---------------------------|--------|--------|--------|-----------|
| cublasSgemm               | 100.00 | 100.00 | 100.00 | 18        |
| gemm_async_opt            | 94.01  | 84.62  | 102.76 | 18        |
| gemm_async                | 87.36  | 78.75  | 96.12  | 18        |
| gemm_double_buffer        | 87.13  | 78.49  | 97.85  | 18        |
| gemm_without_bankconflict | 83.31  | 67.57  | 91.60  | 18        |
| gemm_float4               | 80.55  | 71.25  | 88.54  | 18        |
| gemm_register             | 60.96  | 53.76  | 65.12  | 18        |
| gemm_tile2d               | 59.78  | 53.66  | 62.90  | 18        |
| gemm_tile1d               | 35.59  | 33.33  | 38.43  | 18        |
| gemm_smem                 | 11.93  | 9.41   | 14.57  | 18        |
| gemm_coalescing           | 8.88   | 7.20   | 11.04  | 18        |
| gemm_naive                | 2.53   | 2.26   | 2.74   | 18        |
==========================================================================

### CEMM-Cublas

$$
C_{M*N} = \alpha * A_{M*K} * B_{K*N} + \beta * C_{M*N}
$$

我们以cublasSgemm来作为基准

$alpha=1.0, beta=0.0$.

```        
cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
            N, M, K, &alpha, 
            B, N, 
            A, K, 
            &beta, C, N);
```

![cublasSgemm](https://github.com/Wait-042/GEMM_Naive_to_Cublas/blob/main/fig/cublasSgemm.png)

### GEMM-Naive
每个线程去做K维度的内积，然后将结果输出到C矩阵
```
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

void gemm_naive(float* A, float* B, float* C, int M, int N, int K, cudaStream_t stream) {
    dim3 threads(16, 16);
    dim3 blocks(CEIL(N, threads.x), CEIL(M, threads.y));
    gemm_naive_kernel << <blocks, threads, 0, stream >> > (A, B, C, M, N, K);
}
```
- 我们先按一个warp分析，threadIdx.x范围是0-15，threadIdx.y范围是0-1，那么对应读取A矩阵和写入C矩阵就会出现跨行读写情况，
这样会产生更多的内存事务

- 现在内存读取量是 $(2 * M * N * K + M * N) * 4$ Bytes、浮点运算量是 $2 * M * N * K$ 
所以计算访存比 

$$
\frac{2 * M * N * K}{(2 * M * N * K + M * N) * 4} \approx 0.25 Flop/Byte
$$

- 这是一个比较低的值，我们的显卡理论上能达到 $70 Flop/Byte$ 

- 理论上我们需要做 $M * K + K * N$ 次读取，M * N次写入， $2 * K * M * M$ 次浮点运算
理论计算访存比 

$$
\frac{2 * M * N * K}{M * K + N * K} = \frac{2 * M * N}{M + N} Flop/Byte
$$

- SOL对比(绿色是cublas gemm)
从SOL对比来看Naive是memory bond，说明内存访问很频繁，需要优化内存访问
![gemm_naive_sol](https://github.com/Wait-042/GEMM_Naive_to_Cublas/blob/main/fig_note/gemm_naive_sol.png)

- warp state statistics
从warp stall能看出来“Stall LG Throttle”很长，说明有很多内存读取操作导致指令队列压力大，下面提示也说明了对global memory操作极其频繁
![gemm_naive_warp_state_statistics](https://github.com/Wait-042/GEMM_Naive_to_Cublas/blob/main/fig_note/gemm_naive_warp_state_statistics.png)

- GFLOPS对比
![gemm_naive](https://github.com/Wait-042/GEMM_Naive_to_Cublas/blob/main/fig/gemm_naive.png)

### GEMM-Coalescing
我们将naive kernel的row和col互换下，再按一个warp分析，threadIdx.x范围是0-15，threadIdx.y范围是0-1，这是一样的，但是现在row是只有0和1
col是0-15，那么对应读取A矩阵时，前16个线程读取的同一地址数据，后16个线程也是同一地址数据，这样触发广播机制，读取B矩阵是前16个线程读取的是连续的16个
float数据，后16个线程也是如此，写入C矩阵也是写入连续的地址，这样相比Naive kernel产生更少的内存事务，这就是合并访问，连续的线程访问连续的地址数据。
```
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

void gemm_coalescing(float* A, float* B, float* C, int M, int N, int K, cudaStream_t stream) {
    const int TILE_SIZE = 16;
    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks(CEIL(N, TILE_SIZE), CEIL(M, TILE_SIZE));
    gemm_coalescing_kernel<< <blocks, threads, 0, stream >> > (A, B, C, M, N, K);
}
```

- 计算内积时每次都需要从全局内存读取数据，延迟较高

- SOL对比(绿色是cublas gemm)
说明内存访问依然很频繁
![gemm_coalescing_sol](https://github.com/Wait-042/GEMM_Naive_to_Cublas/blob/main/fig_note/gemm_coalescing_sol.png)

- warp state statistics
从warp stall能看出来“Stall LG Throttle”相比Naive版本有明显改善
![gemm_coalescing_warp_state_statistics](https://github.com/Wait-042/GEMM_Naive_to_Cublas/blob/main/fig_note/gemm_coalescing_warp_state_statistics.png)

- GFLOPS对比
![gemm_coalescing](https://github.com/Wait-042/GEMM_Naive_to_Cublas/blob/main/fig/gemm_coalescing.png)

### GEMM-Smem
- 在CUDA中，内存一般有全局内存、共享内存、常数内存、纹理内存、寄存器内存，我们每次读数据都是从全局内存读取，效率比较低，因为矩阵乘法可以复用数据，
因此我们可以先把数据搬运到访问延迟比较低的共享内存，后续计算内积时可以从共享内存读取数据。
- 但是共享内存是有限的，通常是几十kb大小，当矩阵尺寸较大时，就无法把整个矩阵都搬运到共享内存，因此需要分块做，将矩阵切割成TILE_SIZE*TILE_SIZE大小
的块，然后线程在K维度上累加不同块的内积

| 内存类型    | 物理位置 | 访问权限  | 可见范围     | 生命周期     | 访问时钟延迟周期           |
|---------|------|-------|----------|----------|--------------------|    
| 全局内存    | 芯片外  | 可读可写  | 所有线程和主机端 | 由主机分配与释放 | 最高                 |
| 常量内存    | 芯片外  | 仅可读   | 所有线程和主机端 | 由主机分配与释放 | 中等 (低于全局内存，高于共享内存) |
| 纹理和表面内存 | 芯片外  | 一般仅可读 | 所有线程和主机端 | 由主机分配与释放 | 最高 (与全局内存相当)       |
| 局部内存    | 芯片外  | 可读可写  | 单个线程     | 所在线程     | 最高 (与全局内存相当)       |
| 共享内存    | 芯片内  | 可读可写  | 单个线程块    | 所在线程块    | 很低 (~20-30 个时钟周期)  |
| 寄存器内存   | 芯片内  | 可读可写  | 单个线程     | 所在线程     | 最低 (通常为 0 个额外时钟周期) |

```
TILE_SIZE = 16
\\ 利用共享内存减少全局内存访问
__shared__ float smem_a[TILE_SIZE][TILE_SIZE];
__shared__ float smem_b[TILE_SIZE][TILE_SIZE];

void gemm_smem(float* A, float* B, float* C, int M, int N, int K, cudaStream_t stream) {
    const int TILE_SIZE = 16;
    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks(CEIL(N, threads.x), CEIL(M, threads.y));
    gemm_smem_kernel<TILE_SIZE> << <blocks, threads, 0, stream >> > (A, B, C, M, N, K);
}
```

- 现在内存读取量是 $(\frac{2 * M * N * K}{TILE SIZE} + M * N) * 4$ Bytes、浮点运算量是 $2 * M * N * K$ 
所以计算访存比 

$$
\frac{2 * M * N * K}{(\frac{2 * M * N * K}{TILE SIZE} + M * N) * 4} \approx \frac{TILE SIZE}{4} Flop/Byte
$$

刚好是之前计算Naive版本的Tile_Size倍，这正是因为我们在K维度对数据进行了Tile_Size次复用，Tile_Size=16时访存比为4

- 如果我们考虑改变Tile的形状，对A取 $BM * BK$，B取 $BK * BM$ 
那么现在访存比为

$$
\frac{2*BM*BN*K + M*N}{(BM*K+K*BN)*4} \approx \frac{BM*BN}{2 * (BM+BN)}
$$ 

从这个公式我们可以发现访存比和K无关了，所以我们可以增大BM和BN的大小

- 让单个线程只计算C矩阵的一个值对算力有点浪费，我们可以考虑一个线程输出多个C矩阵的值

- warp state statistics
从warp stall能看出来“Stall MIO Throttle”成为了新的阻塞瓶颈，这是因为我们使用了共享内存，这里提示我们可以使用更宽的数据读取指令来减少指令降低pipeline压力
![gemm_smem_warp_state_statistics](https://github.com/Wait-042/GEMM_Naive_to_Cublas/blob/main/fig_note/gemm_smem_warp_state_statistics.png)

- GFLOPS对比
![gemm_smem](https://github.com/Wait-042/GEMM_Naive_to_Cublas/blob/main/fig/gemm_smem.png)

### GEMM_tile1d
我们设置如下参数：
```
// 分配共享内存和寄存器内存
__shared__ float a_smem[BM][BK];
__shared__ float b_smem[BK][BN];
float c_val[TM] = { 0.0f };
    
// 在极低延迟的共享内存中完成当前 Tile 的乘加计算
#pragma unroll
for (int k = 0; k < BK; k++) {
#pragma unroll
    for (int m = 0; m < TM; m++) {
        c_val[m] += a_smem[ty * TM + m][k] * b_smem[k][tx];
    }
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
```

- 按照之前的分析我们很容易得到现在的访存比为 

$$
\frac{BM*BN}{2 * (BM+BN)} = \frac{128*16}{2 * (128+16)} = 7.11
$$

- warp state statistics
“Stall MIO Throttle”相比GEMM_smem有明显改善，这是因为我们数据复用率更高
![gemm_tile1d_warp_state_statistics](https://github.com/Wait-042/GEMM_Naive_to_Cublas/blob/main/fig_note/gemm_tile1d_warp_state_statistics.png)

- memory workload analysis
可以看到数据指令明显降低
![gemm_tile1d_memory_workload](https://github.com/Wait-042/GEMM_Naive_to_Cublas/blob/main/fig_note/gemm_tile1d_memory_workload.png)

- GFLOPS对比
这里能看到性能提升比较多
![gemm_tile1d](https://github.com/Wait-042/GEMM_Naive_to_Cublas/blob/main/fig/gemm_tile1d.png)

### GEMM_tile2d
同样的我们也可以把BN也放大，这样单个线程就可以处理C矩阵TM*TN个元素

我们设置如下参数：
```
// 分配共享内存和寄存器内存
__shared__ float a_smem[BM][BK];
__shared__ float b_smem[BK][BN];
float c_val[TM][TN] = { {0.0f} };
    
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
```

- 现在的访存比为 

$$
\frac{BM*BN}{2 * (BM+BN)} = \frac{128*128}{2 * (128+128)} = 32
$$

- warp state statistics
现在阻塞周期降低了很多，“Stall Long Scoreboard”这个指标含义是L1Tex(Global, Local, Surface, Tex)结果依赖，后续操作强依赖前面的数据操作，因此需等待前面的数据操作完成
通常需要优化内存访问来降低阻塞时间
![gemm_tile2d_warp_state_statistics](https://github.com/Wait-042/GEMM_Naive_to_Cublas/blob/main/fig_note/gemm_tile2d_warp_state_statistics.png)

- memory workload analysis
数据指令进一步减少，同时这里提示我们访存可以优化，并且有bank conflicts
![gemm_tile2d_memory_workload](https://github.com/Wait-042/GEMM_Naive_to_Cublas/blob/main/fig_note/gemm_tile2d_memory_workload.png)

- GFLOPS对比
![gemm_tile2d](https://github.com/Wait-042/GEMM_Naive_to_Cublas/blob/main/fig/gemm_tile2d.png)

### GEMM_register
前面我们提到CUDA内存时，寄存器内存比共享内存更快，所以我们可以把共享内存的数据往寄存器搬运，然后再计算累乘，不过实测时发现收益不大
```
// 分配共享内存和寄存器内存
__shared__ float a_smem[BM][BK];
__shared__ float b_smem[BK][BN];
float c_val[TM][TN] = { {0.0f} };
float reg_a[TM] = { 0.0f };
float reg_b[TN] = { 0.0f };
    
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
    
```

- warp state statistics
和GEMM_tile2d相比没啥变化，所以前面提到性能没什么区别，猜测应该是寄存器复用的收益在当前不明显
![gemm_register_warp_state_statistics](https://github.com/Wait-042/GEMM_Naive_to_Cublas/blob/main/fig_note/gemm_register_warp_state_statistics.png)

- GFLOPS对比
![gemm_register](https://github.com/Wait-042/GEMM_Naive_to_Cublas/blob/main/fig/gemm_register.png)

### GEMM_FLOAT4
从 Global Memory 加载数据到 Shared Memory 时，如果每次只搬运一个 float（32 bit），需要执行大量 LDG.32/STS.32 指令。
GPU 的内存系统支持一次搬运 128 bit（即一个 float4），这能将指令数量减少为原来的 1/4，显著降低指令发射压力。
```
// 分配共享内存和寄存器内存
__shared__ float a_smem[BM][BK];
__shared__ float b_smem[BK][BN];
float c_val[TM][TN] = { {0.0f} };
float reg_a[TM] = { 0.0f };
float reg_b[TN] = { 0.0f };
    
// 全局内存向量化读取和共享内存向量化写入
FLOAT4(a_smem[a_smem_row][a_smem_col]) = CFLOAT4(A[a_row * K + a_col]);
FLOAT4(b_smem[b_smem_row][b_smem_col]) = CFLOAT4(B[b_row * N + b_col]);

// 共享内存向量化读取和寄存器向量化写入
// 注意这里a_smem没有向量化，因为这里要跨行读取，没办法向量化，后面kernel会先把A转置存到共享内存
for (int m = 0; m < TM; m++) {
    reg_a[m] = a_smem[ty * TM + m][k];
}
for (int n = 0; n < TN>>2; n++) {
    FLOAT4(reg_b[n << 2]) = FLOAT4(b_smem[k][tx * TN + (n << 2)]);
}

// 寄存器内存向量化读取和全局内存向量化写入
FLOAT4(C[c_row * N + c_col]) = FLOAT4(c_val[m][n<<2]);
```

- 共享内存的读写存在大量的bank conflicts，bank conflicts来源于一个warp内不同线程访问同一bank不同地址，这会增加额外的内存事务
```
\\ 假设k=0 n=0
b_smem[k][tx * TN + (n << 2)] = b_smem[0][tx * TN]
线程0~15: tx -> 0~15, 线程16~31: tx -> 0~15
线程0~15: tx*TN -> 0~120(step=8), 线程16~31: tx*TN -> 0~120(step=8)
````
- 这里的跨行读取就会产生bank conflict，因为0、32、64、96是同一bank(bank0)，但是不同现在在同一bank读取不同地址数据，因此这里就会产生冲突

- memory workload analysis
数据指令进一步减少，同时这里提示bank conflicts
![gemm_float4_memory_workload](https://github.com/Wait-042/GEMM_Naive_to_Cublas/blob/main/fig_note/gemm_float4_memory_workload.png)

- GFLOPS对比
![gemm_float4](https://github.com/Wait-042/GEMM_Naive_to_Cublas/blob/main/fig/gemm_float4.png)

### GEMM_without_bankconflict
根据前面kernel的结构，我们对共享内存的读写索引进行了重排，从而消除了bank conflicts，大大减少了内存事务，提高了访存效率

![A_Tile_data_layout](https://github.com/Wait-042/GEMM_Naive_to_Cublas/blob/main/fig_note/A_Tile_data_layout.png)
![B_Tile_data_layout](https://github.com/Wait-042/GEMM_Naive_to_Cublas/blob/main/fig_note/B_Tile_data_layout.png)
因为向量化读取一次读4个float，数据Tile大小是8*128=1024个float，所以刚好一个block可以通过float4读取一个Tile所有数据，我们把256个线程映射到
A和BTile取float4数据时的行列索引
```
int tid = ty * blockDim.x + tx;

int a_smem_row = tid >> 1;
int a_smem_col = (tid & 1) << 2;
int b_smem_row = tid >> 5;
int b_smem_col = (tid & 31) << 2;

int a_row = by * BM + a_smem_row;
int b_col = bx * BN + b_smem_col;

```

然后把数据搬运到共享内存，注意这里我们把A的Tile数据存到共享内存时是转置存的，这里的目的是为了后续从共享内存读数据是能够向量化读取，因为从全局内存搬运到
共享内存这里虽然有bank conflict，但是只需要做一次搬运就行，而后面从共享内存读取数据时因为会复用数据，所以在前面把转置就做掉更合适
```
__shared__ float a_smem[BK][BM];
__shared__ float b_smem[BK][BN];

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
```

从共享内存读取数据放入寄存器内存，之前的bank conflict大头就在这，现在a_smem和b_smem结构是一样的，都是8*128
```
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
```
这里我根据warp0的tx、ty线程索引给出了共享内存读取索引情况,先看k=0情况，可以看到我对分了8个颜色块，这代表内存事务发送的数据申请索引，这里其实我之前
也困惑了很久，我之前理解内存事务是在warp级别的，也就是说要从warp内32个线程的数据索引情况去分析bank冲突和额外的内存事务，但是在向量化读取时，就如下面这种情况
编译器会识别连续数据的申请从而把前几个线程的向量化读取合并成一个内存事务，这样就避免了bank冲突
这一块可参考<https://forums.developer.nvidia.com/t/problem-about-bank-conflict-test/285476/2>
![warp0_tx_data_idx](https://github.com/Wait-042/GEMM_Naive_to_Cublas/blob/main/fig_note/warp0_tx_data_idx.png)
![warp1_tx_data_idx](https://github.com/Wait-042/GEMM_Naive_to_Cublas/blob/main/fig_note/warp1_tx_data_idx.png)

- memory workload analysis
可以看到这里我们消除了Shared Load的bank conflicts，虽然带来了一定的Shared Store bank conflicts，不过相比于Load是小量
![gemm_without_bankconflict_memory_workload](https://github.com/Wait-042/GEMM_Naive_to_Cublas/blob/main/fig_note/gemm_without_bankconflict_memory_workload.png)

- GFLOPS对比
![gemm_without_bankconflict](https://github.com/Wait-042/GEMM_Naive_to_Cublas/blob/main/fig/gemm_without_bankconflict.png)

### GEMM_double_buffer
为了避免数据的读写冲突，我们对共享内存和寄存器内存额外分配了一倍的空间，使得当每次读写位置不在同一地址，避免冲突和串行等待，降低了数据同步延迟
之前的流程：数据写入到共享内存——同步——寄存器读取共享内存——累乘计算——同步——下一轮迭代

共享内存、寄存器内存额外分配一倍空间，我们可以理解成两个Tile
现在的流程：数据写入到共享内存Tile0——同步——寄存器读取共享内存Tile0——累乘计算——翻转Tile索引(0变1，1变0)——数据写入到共享内存Tile1——同步——寄存器读取共享内存Tile1——累乘计算——翻转Tile索引(0变1，1变0)
从这个流程上我们能看到比之前的代码在每次迭代里少了一次同步
```
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

```
- SOL
计算吞吐和内存吞吐都有一定提升
![gemm_double_buffer_sol](https://github.com/Wait-042/GEMM_Naive_to_Cublas/blob/main/fig_note/gemm_double_buffer_sol.png)

- GFLOPS对比
![gemm_double_buffer](https://github.com/Wait-042/GEMM_Naive_to_Cublas/blob/main/fig/gemm_double_buffer.png)

### GEMM_async
从 SM80（Ampere）开始，NVIDIA 引入了 cp.async 指令，支持从 Global Memory 到 Shared Memory 的异步直接拷贝，无需经过寄存器中转：
前面的共享内存需要从全局内存-L2缓存-共享内存，且必须等待数据读取完毕才能进行下一步操作，我们引入异步拷贝操作，直接从全局内存拷贝到共享内存，
让数据在计算tile块的时候，同时异步读取tile+1块，构建数据传递流水线从而进一步掩盖数据延迟，减少同步消耗
```
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

// B 矩阵: 内存连续，使用硬件 cp.async 16字节零寄存器搬运
int b_row_0 = 0 + b_smem_row;
if (b_row_0 < K && b_col < N) {
    cp_async_cg(&b_smem[0][b_smem_row][b_smem_col], &B[b_row_0 * N + b_col]);
}
cp_async_commit();
cp_async_wait_group<0>(); // 确保 第 0 个 Tile 到位
__syncthreads();
```
- memory workload analysis
数据指令减少
![gemm_async_memory_workload](https://github.com/Wait-042/GEMM_Naive_to_Cublas/blob/main/fig_note/gemm_async_memory_workload.png)

- GFLOPS对比
![gemm_async](https://github.com/Wait-042/GEMM_Naive_to_Cublas/blob/main/fig/gemm_async.png)

### GEMM_async_opt
由于A矩阵需要转置存储的原因，无法向量化读取，因为异步拷贝要求4/8/16字节对齐，我们异步拷贝只拷贝了B矩阵，还是要等A矩阵拷贝完才能进行下一步的计算。
因此，在这里我们提前将A矩阵转置，这样A矩阵的数据读取逻辑和B矩阵保持一致，能够同时使用异步拷贝预取A和B矩阵的Tile数据，进一步掩盖数据延迟
```
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
```

- SOL对比(绿色是cublas gemm)
计算吞吐稍微逊色cublas
![gemm_async_opt_sol](https://github.com/Wait-042/GEMM_Naive_to_Cublas/blob/main/fig_note/gemm_async_opt_sol.png)

- warp state statistics
“Issued Warp Per Scheduler”指标和cublas一致，说明warp指令发射没有被明显阻塞(理想值是1)
![gemm_async_opt_warp_state_statistics](https://github.com/Wait-042/GEMM_Naive_to_Cublas/blob/main/fig_note/gemm_async_opt_warp_state_statistics.png)

- GFLOPS对比
![gemm_async_opt](https://github.com/Wait-042/GEMM_Naive_to_Cublas/blob/main/fig/gemm_async_opt.png)

## 结语
每次学GEMM感觉都有很多新的东西，即便是现在感觉自己也还有很多没完全搞明白，有的似懂非懂，有的感觉像是朦胧的摸到了一点边，继续加油吧！

## 参考
- [CUDA（三）：通用矩阵乘法：从入门到熟练](https://zhuanlan.zhihu.com/p/657632577)
- [CUDA SGEMM优化笔记](https://linn-ylz.com/Computer-Science/CUDA/CUDA-SGEMM-optimization-notes/)
- [CUDA GEMM算子性能优化](https://caomaolufei.github.io/AIInfraGuide/guides/%E6%A8%A1%E5%9D%97%E4%BA%8C-cuda%E7%BC%96%E7%A8%8B%E4%B8%8E%E7%AE%97%E5%AD%90%E4%BC%98%E5%8C%96/41-cuda-gemm%E7%AE%97%E5%AD%90%E6%80%A7%E8%83%BD%E4%BC%98%E5%8C%96/)
- [异步拷贝与现代 Pipeline——从 cp.async 到 memcpy_async](https://forceinjection.github.io/02_gpu_programming/02_cuda/16_async_copy_pipeline.html)
- [CUDA Matrix Multiplication Optimization](https://leimao.github.io/article/CUDA-Matrix-Multiplication-Optimization/)
- [How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance: a Worklog](https://siboehm.com/articles/22/CUDA-MMM)
- [LeetGPU Matrix Multiplication](https://leetgpu.com/challenges/matrix-multiplication)
- [CUDA GEMM 算子详解](https://zhuanlan.zhihu.com/p/1910636263666610461)
- [搞懂 CUDA Shared Memory 上的 bank conflicts 和向量化指令（LDS.128 / float4）的访存特点](https://zhuanlan.zhihu.com/p/690052715)
- [Nsight Compute - Scheduler Statistics](https://zhuanlan.zhihu.com/p/673770855)
- [【CUDA进阶】深入理解 Nsight System 和 Nsight Compute](https://www.bilibili.com/video/BV13w411o7cu/?spm_id_from=333.337.search-card.all.click&vd_source=1c4ced049bd2298f31c95dae280fb833)
- [cutlass GEMM——sliced-K、split-K & stream-K 分析 （一）](https://zhuanlan.zhihu.com/p/713411778)
- [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html?highlight=shfl#throughput-of-native-arithmetic-instructions)
- [NVIDIA Nsight Systems user guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html#)
- [The User Guide for Nsight Compute](https://docs.nvidia.com/nsight-compute/NsightCompute/index.html#)
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/index.html)

## 下一步计划和代办项

- cutlass学习