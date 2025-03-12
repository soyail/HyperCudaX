nvprof --metrics achieved_occupancy,flop_count_sp ./benchmark_matmul
nsys profile --trace=cuda,nvtx --output=matmul_profile ./benchmark_matmul
