#include "../max_pooling/max_pooling.hpp"
#include <iostream>

MaxPooling::MaxPooling(GPU::Device& gpu, int pool_size, uint32_t n_inputs, uint32_t in_size):
    gpu(gpu), n_inputs(n_inputs), in_size(in_size), pool_size(pool_size) {

    if (in_size % pool_size != 0) {
        std::cerr << "MaxPooling::MaxPooling() | Input size not compatible with pool size!" << std::endl;
    }

    uint32_t out_size_dim = in_size / pool_size;

    this->out_size = out_size_dim;

    auto result = cudaMalloc(&this->cuda_idx, sizeof(int) * out_size_dim * out_size_dim * n_inputs);

    if (result != cudaSuccess) {
        std::cerr << "MaxPooling::MaxPooling() | Could alloc mem for idx :(" << std::endl;
    }

}


void MaxPooling::init_self(GPU::Device& gpu, int pool_size, uint32_t n_inputs, uint32_t in_size) {

    this->gpu = gpu;
    this->n_inputs = n_inputs;
    this->in_size = in_size;
    this->pool_size = pool_size;

    if (in_size % pool_size != 0) {
        std::cerr << "MaxPooling::MaxPooling() | Input size not compatible with pool size!" << std::endl;
    }

    uint32_t out_size_dim = in_size / pool_size;

    this->out_size = out_size_dim;

    auto result = cudaMalloc(&this->cuda_idx, sizeof(int) * out_size_dim * out_size_dim * n_inputs);

    if (result != cudaSuccess) {
        std::cerr << "MaxPooling::MaxPooling() | Could alloc mem for idx :(" << std::endl;
    }
}


void MaxPooling::pool(GPU::Tensor in, GPU::Tensor out, cudaStream_t stream) {

    //I really hope this works D:
    for (uint32_t i = 0; i < this->n_inputs; ++i) {

        gpu.max_pool_ver2(in, out, cuda_idx, pool_size, stream);

        in.dat_pointer = in.dat_pointer + i*in_size*in_size;
        out.dat_pointer = out.dat_pointer + i*out_size*out_size;
    }
}
