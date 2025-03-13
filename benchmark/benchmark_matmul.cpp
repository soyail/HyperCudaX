#include "hypercudax.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

void benchmark_hypermatmul(int M, int N, int K){
    float *A, *B, *C;
    cudaMallocManaged(&A, M*K*sizeof(float));
    cudaMallocManaged(&B, K*N*sizeof(float));  
    cudaMallocManaged(&C, M*N*sizeof(float));
    auto start = std::chrono::high_resolution_clock::now();
    hypercudax::matmul(A, B, C, M, N, K);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    double gflps = 2.0*M*N*K/((end-start).count());
    std::cout << "HyperCudaX GFLPS: " << gflps << std::endl;
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
}

void benchmark_cublas(int M, int N, int K){
    cublasHandle_t handle;
    cublasCreate(&handle);
    float *A, *B, *C;
    cudaMallocManaged(&A, M*K*sizeof(float));
    cudaMallocManaged(&B, K*N*sizeof(float));
    cudaMallocManaged(&C, M*N*sizeof(float));
    auto start = std::chrono::high_resolution_clock::now();
    float alpha=1.0f;
    float beta=0.0f;
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, N, A, K, &beta, C, N);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    double gflps = 2.0*M*N*K/((end-start).count());
    std::cout << "CUBLAS GFLPS: " << gflps << std::endl;
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    cublasDestroy(handle);
}

void benchmark_hypermatmul_fp16(int M, int N, int K){
    __half *A, *B, *C;
    cudaMallocManaged(&A, M*K*sizeof(__half));
    cudaMallocManaged(&B, K*N*sizeof(__half));  
    cudaMallocManaged(&C, M*N*sizeof(__half));
    
    // 初始化数据
    for(int i = 0; i < M*K; i++) {
        A[i] = __float2half(static_cast<float>(rand())/RAND_MAX);
    }
    for(int i = 0; i < K*N; i++) {
        B[i] = __float2half(static_cast<float>(rand())/RAND_MAX);
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    hypercudax::matmul_fp16(A, B, C, M, N, K);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    double gflps = 2.0*M*N*K/((end-start).count());
    std::cout << "HyperCudaX FP16 GFLPS: " << gflps << std::endl;
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
}

void benchmark_cublas_fp16(int M, int N, int K){
    cublasHandle_t handle;
    cublasCreate(&handle);
    __half *A, *B, *C;
    cudaMallocManaged(&A, M*K*sizeof(__half));
    cudaMallocManaged(&B, K*N*sizeof(__half));
    cudaMallocManaged(&C, M*N*sizeof(__half));
    
    // 初始化数据
    for(int i = 0; i < M*K; i++) {
        A[i] = __float2half(static_cast<float>(rand())/RAND_MAX);
    }
    for(int i = 0; i < K*N; i++) {
        B[i] = __float2half(static_cast<float>(rand())/RAND_MAX);
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    __half alpha = __float2half(1.0f);
    __half beta = __float2half(0.0f);
    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
                 N, M, K, 
                 &alpha, B, CUDA_R_16F, N,
                 A, CUDA_R_16F, K,
                 &beta, C, CUDA_R_16F, N,
                 CUDA_R_16F, CUBLAS_GEMM_DEFAULT);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    double gflps = 2.0*M*N*K/((end-start).count());
    std::cout << "CUBLAS FP16 GFLPS: " << gflps << std::endl;
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    cublasDestroy(handle);
}

void verify_result(const float* A, const float* B, int M, int N){
    for(int i=0; i<M*N; ++i){
        if(fabs(A[i]-B[i]) > 1e-3){
            std::cout << "Verification failed!" << std::endl;
            std::cout << "A[" << i << "] = " << A[i] << ", B[" << i << "] = " << B[i] << std::endl;
            return;
        }
    }
    std::cout << "Verification passed!" << std::endl;
}

void verify_result_fp16(const __half* A, const __half* B, int M, int N){
    for(int i=0; i<M*N; ++i){
        float diff = fabs(__half2float(A[i]) - __half2float(B[i]));
        if(diff > 1e-2){
            std::cout << "Verification failed!" << std::endl;
            std::cout << "A[" << i << "] = " << __half2float(A[i]) 
                      << ", B[" << i << "] = " << __half2float(B[i]) << std::endl;
            return;
        }
    }
    std::cout << "FP16 Verification passed!" << std::endl;
}

void initialize_matrix(float* matrix, int rows, int cols){
    for(int i = 0; i < rows; i++){
        for(int j = 0; j < cols; j++){
            matrix[i*cols+j] = static_cast<float>(rand())/RAND_MAX;
        }
    }
}

void test_correctness(){
    //TODO implement this function
    int M = 128, N = 128, K = 128;
    float* A = (float*)malloc(M*K*sizeof(float));
    float* B = (float*)malloc(K*N*sizeof(float));
    float* C = (float*)malloc(M*N*sizeof(float));
    float* C_cublas = (float*)malloc(M*N*sizeof(float));
    initialize_matrix(A, M, K);
    initialize_matrix(B, K, N);
    float *A_gpu, *B_gpu, *C_gpu, *C_cublas_gpu;
    cudaMalloc(&A_gpu, M*K*sizeof(float));
    cudaMalloc(&B_gpu, K*N*sizeof(float));
    cudaMalloc(&C_gpu, M*N*sizeof(float));
    cudaMalloc(&C_cublas_gpu, M*N*sizeof(float));

    cudaMemcpy(A_gpu, A, M*K*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B_gpu, B, K*N*sizeof(float), cudaMemcpyHostToDevice);
    
    // 初始化C_gpu和C_cublas_gpu
    cudaMemset(C_gpu, 0, M*N*sizeof(float));
    cudaMemset(C_cublas_gpu, 0, M*N*sizeof(float));
    hypercudax::matmul(A_gpu, B_gpu, C_gpu, M, N, K);

    cublasHandle_t handle;
    cublasCreate(&handle);
    float alpha = 1.0f;
    float beta = 0.0f;
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B_gpu, N, A_gpu, K, &beta, C_cublas_gpu, N);
    

    // 将结果从GPU拷贝到CPU
    cudaMemcpy(C, C_gpu, M*N*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(C_cublas, C_cublas_gpu, M*N*sizeof(float), cudaMemcpyDeviceToHost);
    
    verify_result(C, C_cublas, M, N);
    cudaFree(A_gpu);
    cudaFree(B_gpu);
    cudaFree(C_gpu);
    cudaFree(C_cublas_gpu);
    cublasDestroy(handle);
    free(A);
    free(B);
    free(C);
    free(C_cublas);
}

void test_correctness_fp16(){
    int M = 128, N = 128, K = 128;
    __half* A = (__half*)malloc(M*K*sizeof(__half));
    __half* B = (__half*)malloc(K*N*sizeof(__half));
    __half* C = (__half*)malloc(M*N*sizeof(__half));
    __half* C_cublas = (__half*)malloc(M*N*sizeof(__half));
    
    // 初始化数据
    for(int i = 0; i < M*K; i++) {
        A[i] = __float2half(static_cast<float>(rand())/RAND_MAX);
    }
    for(int i = 0; i < K*N; i++) {
        B[i] = __float2half(static_cast<float>(rand())/RAND_MAX);
    }
    
    __half *A_gpu, *B_gpu, *C_gpu, *C_cublas_gpu;
    cudaMalloc(&A_gpu, M*K*sizeof(__half));
    cudaMalloc(&B_gpu, K*N*sizeof(__half));
    cudaMalloc(&C_gpu, M*N*sizeof(__half));
    cudaMalloc(&C_cublas_gpu, M*N*sizeof(__half));

    cudaMemcpy(A_gpu, A, M*K*sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(B_gpu, B, K*N*sizeof(__half), cudaMemcpyHostToDevice);
    
    cudaMemset(C_gpu, 0, M*N*sizeof(__half));
    cudaMemset(C_cublas_gpu, 0, M*N*sizeof(__half));
    
    hypercudax::matmul_fp16(A_gpu, B_gpu, C_gpu, M, N, K);

    cublasHandle_t handle;
    cublasCreate(&handle);
    __half alpha = __float2half(1.0f);
    __half beta = __float2half(0.0f);
    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
                 N, M, K, 
                 &alpha, B_gpu, CUDA_R_16F, N,
                 A_gpu, CUDA_R_16F, K,
                 &beta, C_cublas_gpu, CUDA_R_16F, N,
                 CUDA_R_16F, CUBLAS_GEMM_DEFAULT);

    cudaMemcpy(C, C_gpu, M*N*sizeof(__half), cudaMemcpyDeviceToHost);
    cudaMemcpy(C_cublas, C_cublas_gpu, M*N*sizeof(__half), cudaMemcpyDeviceToHost);
    
    verify_result_fp16(C, C_cublas, M, N);
    
    cudaFree(A_gpu);
    cudaFree(B_gpu);
    cudaFree(C_gpu);
    cudaFree(C_cublas_gpu);
    cublasDestroy(handle);
    free(A);
    free(B);
    free(C);
    free(C_cublas);
}

void run_benchmarks(){
    std::vector<std::tuple<int,int,int>> test_sizes = {
        {256, 256, 256}, {512, 512, 512}, {1024, 1024, 1024}, {2048, 2048, 2048},
        {128, 8192, 128}, {8192, 128, 8192}, {2048, 512, 2048}, {4096, 4096, 4096}
    };
    
    std::cout << "\n=== FP32 Benchmarks ===" << std::endl;
    for(auto& [M, N, K] : test_sizes){
        std::cout << "\nBenchmarking M=" << M << ", N=" << N << ", K=" << K << std::endl;
        benchmark_hypermatmul(M, N, K);
        benchmark_cublas(M, N, K);
    }
    
    std::cout << "\n=== FP16 Benchmarks ===" << std::endl;
    for(auto& [M, N, K] : test_sizes){
        std::cout << "\nBenchmarking M=" << M << ", N=" << N << ", K=" << K << std::endl;
        benchmark_hypermatmul_fp16(M, N, K);
        benchmark_cublas_fp16(M, N, K);
    }
}



int main(){
    std::cout << "=== Testing FP32 Correctness ===" << std::endl;
    test_correctness();
    
    std::cout << "\n=== Testing FP16 Correctness ===" << std::endl;
    test_correctness_fp16();
    
    std::cout << "\n=== Running Benchmarks ===" << std::endl;
    run_benchmarks();
    
    return 0;
}