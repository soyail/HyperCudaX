#ifndef HYPERCUDAX_H
#define HYPERCUDAX_H

#include <cuda_fp16.h>

namespace hypercudax {
    void matmul(const float *A, const float *B, float *C, const int M, const int N, const int K);
    void matmul_fp16(const __half *A, const __half *B, __half *C, const int M, const int N, const int K);
}

#endif