#include "../ai/allocator/allocator.hpp"
#include "cuda_runtime_api.h"
#include <iostream>

int main(int argv, char** argc) {

    GPU::Device gpu;
    Allocator alloc(gpu, 32);

    const size_t matrix_dim = 2;
    const size_t matrix_size = matrix_dim * matrix_dim;

    const size_t kernel_dim = 1;
    const size_t kernel_size = kernel_dim * kernel_dim;

    const size_t out_dim = GPU::Device::calculate_conv_dims(kernel_dim, matrix_dim);
    const size_t out_size = out_dim*out_dim;

    std::cout << "out size = " << out_dim << std::endl;

    const size_t block_size = matrix_size + kernel_size + out_size;
    int block_res = alloc.alloc_new_block(block_size * 3);

    if (block_res != 0) {
        std::cerr << "allocating memory failed!" << std::endl;
        exit(-1);
    }

    float* cudaMat = alloc.alloc_space(matrix_size * 3);
    float* cudaKernel = alloc.alloc_space(kernel_size * 3);
    float* cudaOut = alloc.alloc_space(out_size * 3);

    std::cout << "Out size: " << out_dim << std::endl;

    float mat_dat[matrix_size * 3] = {
        1, 2, 
        3, 4, 
        5, 6, 
        7, 8, 
        9, 10, 
        11, 12,
    };

    float kernel_dat[kernel_size * 3] = {1, 2, 3};
    float out_dat[3];

    gpu.memcpy_host(mat_dat, cudaMat, sizeof(float) * matrix_size * 3); 
    gpu.memcpy_host(kernel_dat, cudaKernel, sizeof(float) * kernel_size * 3); 

    gpu.batched_conv_ver1(cudaKernel, cudaMat, cudaOut, kernel_dim, matrix_dim, 2, 
            GPU::ActivationFunction::ReLU, 3, 4, 3, 0);

    gpu.device_sync();
    gpu.memcpy_device(cudaOut, out_dat, sizeof(float) * out_size * 3);

    const cudaError_t last_error = cudaGetLastError();

    if (last_error != 0) {
        const char* error_name = cudaGetErrorName(last_error);
        const char* error_str = cudaGetErrorString(last_error);

        std::cout << "Error Name: " << error_name << "\nError desc: " << error_str << std::endl;
    } else {
        std::cout << "No error, nice! D: kringe" << std::endl;
    }

    for (int i = 0; i < out_size * 3; ++i) {
        std::cout << "Elm " << i << ". => " << out_dat[i] << std::endl;
    }

    return 0;
}
