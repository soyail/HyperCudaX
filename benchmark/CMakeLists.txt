# Set the project name
project(benchmark_matmul)

#find_package(CUBLAS REQUIRED)

#include_directories(${PROJECT_SOURCE_DIR}/include)
# Add the executable
add_executable(benchmark_matmul benchmark_matmul.cpp)

target_include_directories(benchmark_matmul PRIVATE ${CUDA_INCLUDE_DIRS})

target_link_libraries(benchmark_matmul PRIVATE hypercudax cublas)

set_target_properties(benchmark_matmul PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

