#include "hypercudax.h"

namespace hypercudax{
    template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void matmul_kernel(
    const float* A,
    const float* B,
    float* C,
    const int M,
    const int N,
    const int K
){
    //share memory.
    __shared__ float shareA[BM*BK];
    __shared__ float shareB[BK*BN];

    // use register to store result.
    float threadResults[TM*TN] = {0.0};
    float regA[TM] = {0.0};
    float regB[TN] = {0.0};
   
   
    // calculating the indices that this thread will be responsible for the result C.
    const int threadRow = (threadIdx.x*TN / BN)*TM;
    const int threadCol = threadIdx.x*TN % BN;

    A += blockIdx.x * BM * K;
    B += blockIdx.y * BN;
    C += blockIdx.x * BM * N + blockIdx.y * BN;

    int threadNumsPerBlock = BM * BN / (TM*TN);
    int strideA = threadNumsPerBlock/BK;
    int strideB = threadNumsPerBlock/BN;

    // calculating the indices that this thread will load into SMEM
    const uint innerRowA = threadIdx.x / BK; // [0,1,2,...,BM]
    const uint innerColA = threadIdx.x % BK; 
    const uint innerRowB = threadIdx.x / BN; // [0,1,2,...,BK]
    const uint innerColB = threadIdx.x % BN;

    for(uint tile_idx = 0; tile_idx < K; tile_idx += BK){
        for(uint loadoffset = 0; loadoffset < BM; loadoffset+=strideA){
            shareA[(innerRowA+loadoffset)*BK+innerColA] = A[(innerRowA+loadoffset)*K+innerColA]; 
        }
        for(uint loadoffset = 0; loadoffset < BK; loadoffset+=strideB){
            shareB[(innerRowB+loadoffset)*BN+innerColB] = B[(innerRowB+loadoffset)*N+innerColB];
        }
        __syncthreads();

        A += BK;
        B += BK*N; 
        
        for(uint dotIdx = 0; dotIdx < BK; dotIdx++){                             
            // load into register.
            for(uint i=0; i<TM; i++){
                regA[i] = shareA[(threadRow+i)*BK+dotIdx];
            }
            for(uint i=0; i<TN; i++){
                regB[i] = shareB[dotIdx*BN+threadCol+i];
            }
            for(uint resIdxM=0; resIdxM < TM; resIdxM++){
                for(uint resIdxN=0; resIdxN<TN; resIdxN++){
                    threadResults[resIdxM*TN+resIdxN] += regA[resIdxM]*regB[resIdxN];
                }
            }
        }
        __syncthreads();
    }
    for(uint resIdxM=0; resIdxM < TM; resIdxM++){
        for(uint resIdxN=0; resIdxN<TN; resIdxN++){
            C[(threadRow+resIdxM)*N+threadCol+resIdxN] = threadResults[resIdxM*TN+resIdxN];
        }
    }
}

void matmul(
    const float* A,
    const float* B,
    float* C,
    const int M,
    const int N,
    const int K
){
    // dim对于性能的影响
    const int BM = 64;
    const int BN = 64;
    const int BK = 16;
    const int TM = 8;
    const int TN = 4;
    
    dim3 threads(BM*BN/(TM*TN));
    dim3 blocks((M+BM-1)/BM, (N+BN-1)/BN);
    matmul_kernel<BM,BN,BK,TM,TN><<<blocks, threads>>>(
        A, B, C, M, N, K
    );
}

}