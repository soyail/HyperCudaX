#include "hypercudax.h"
#include <cuda_runtime.h>
#include <iostream>

namespace hypercudax {
template <const int TILE_SIZE>
__global__ void matmul_kernel_fp16(
    const __half* A,
    const __half* B,
    __half* C,
    const int M,
    const int N,
    const int K
){
    // 共享内存声明，使用__half2类型
    __shared__ __half2 shareA[TILE_SIZE][TILE_SIZE/2];
    __shared__ __half2 shareB[TILE_SIZE/2][TILE_SIZE];

    // 计算线程索引
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;

    // 计算全局索引，每个线程处理两个元素
    const int row = by * TILE_SIZE + ty;
    const int col = bx * TILE_SIZE + tx*2;

    // 初始化结果，使用__half2
    __half2 sum = __float2half2_rn(0.0f);

    // 检查边界
    if (row >= M || col + 1 >= N) return;

    // 主循环
    for (int tile_idx = 0; tile_idx < K; tile_idx += TILE_SIZE) {
        // 加载数据到共享内存，使用__half2
        if (row < M && tile_idx + tx*2 < K) {
            shareA[ty][tx] = reinterpret_cast<const __half2*>(A + row * K + tile_idx + tx*2)[0];
        } else {
            shareA[ty][tx] = __float2half2_rn(0.0f);
        }

        if (tile_idx + ty*2 < K && col + 1 < N) {
            shareB[ty][tx] = reinterpret_cast<const __half2*>(B + (tile_idx + ty*2) * N + col)[0];
        } else {
            shareB[ty][tx] = __float2half2_rn(0.0f);
        }

        __syncthreads();

        // 计算矩阵乘法，使用__half2
        for (int k = 0; k < TILE_SIZE/2; ++k) {
            // 使用__half2的融合乘加
            sum = __hfma2(shareA[ty][k], shareB[k][tx], sum);
        }

        __syncthreads();
    }

    // 写入结果
    if (row < M && col + 1 < N) {
        *reinterpret_cast<__half2*>(C + row * N + col) = sum;
    }
}

void matmul_fp16(const __half* A, const __half* B, __half* C, const int M, const int N, const int K) {
    // 确保输入矩阵维度是2的倍数
    if (M % 2 != 0 || N % 2 != 0 || K % 2 != 0) {
        std::cerr << "Error: Matrix dimensions must be multiples of 2 for FP16 implementation" << std::endl;
        return;
    }

    const int TILE_SIZE = 16;  // 必须是2的倍数
    // 修改blockDim，x方向减半因为每个线程处理两个元素
    dim3 blockDim(TILE_SIZE/2, TILE_SIZE);
    dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
    matmul_kernel_fp16<TILE_SIZE><<<gridDim, blockDim>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}
}