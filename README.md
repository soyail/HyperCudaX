# HyperCudaX
A high-performance CUDA acceleration library for large model inference, optimized for Matmul, MLP, and FlashAttention with cutting-edge operator fusion, quantization, and tensor parallelism. 


## 编译

1. 修改CMakeLists.txt中的CMAKE_CUDA_ARCHITECTURES为你显卡对应的版本。可通过https://developer.nvidia.com/cuda-gpus进行查询。
2.