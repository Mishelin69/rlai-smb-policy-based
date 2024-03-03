#pragma once
#include "../../../gpu/gpu.h"

class MaxPooling {

private:

    int pool_size;
    uint32_t n_inputs;
    uint32_t in_size;
    uint32_t out_size;

    int* cuda_idx;

    GPU::Device& gpu = DummyDevice;

public:

    void init_self(GPU::Device& gpu, int pool_size, uint32_t n_inputs, uint32_t in_size);

    MaxPooling();

    MaxPooling(GPU::Device& gpu, int pool_size, uint32_t n_inputs, uint32_t in_size);
    ~MaxPooling() = default;
    MaxPooling(MaxPooling& other) = default;
    MaxPooling(MaxPooling&& other);

    void pool(GPU::Tensor in, GPU::Tensor out, cudaStream_t stream);

    //DONT FORGET TO RESET MAX POOLING UNPOOLING DELTA STUFF PLS :sob:
    void unpool(GPU::Tensor out, cudaStream_t stream);

};
