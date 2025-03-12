#ifndef HYPERCUDAX_H
#define HYPERCUDAX_H



namespace hypercudax {
    void matmul(const float *A, const float *B, float *C, const int M, const int N, const int K);
}

#endif