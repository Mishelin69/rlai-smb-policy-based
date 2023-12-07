#include "../ai/allocator/allocator.hpp"
#include <iostream>

int main(int argv, char** argc) {

    GPU::Device gpu;
    Allocator alloc(gpu);

    const size_t matrix_dim = 2;
    const size_t matrix_size = matrix_dim * matrix_dim;

    const size_t kernel_dim = 1;
    const size_t kernel_size = kernel_dim * kernel_dim;

    const size_t out_dim = GPU::Device::calculate_conv_dims(kernel_dim, matrix_dim);
    const size_t out_size = out_dim*out_dim;

    const size_t block_size = matrix_size + kernel_size + out_size;
    int block_res = alloc.alloc_new_block(block_size);

    if (block_res != 0) {
        std::cerr << "allocating memory failed!" << std::endl;
        exit(-1);
    }

    float* cudaMat = alloc.alloc_space(matrix_size);
    float* cudaKernel = alloc.alloc_space(kernel_size);
    float* cudaOut = alloc.alloc_space(out_size);

    float mat_dat[matrix_size] = {1, 2, 3, 4};
    float kernel_dat[kernel_size] = {1};
    float out_dat[4];

    gpu.memcpy_host(mat_dat, cudaMat, sizeof(float) * matrix_size); 
    gpu.memcpy_host(kernel_dat, cudaKernel, sizeof(float) * kernel_size); 
    gpu.conv_ver1(cudaKernel, cudaMat, cudaOut, kernel_dim, matrix_dim, 2, GPU::ActivationFunction::None);
    gpu.device_sync();
    gpu.memcpy_device(cudaOut, out_dat, sizeof(float) * out_size);

    for (int i = 0; i < out_size; ++i) {
        std::cout << "Elm " << i << ". => " << out_dat[i] << std::endl;
    }

    return 0;
}
