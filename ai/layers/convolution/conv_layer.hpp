#pragma once
#include <cstdint>

class Allocator;

class ConvolutionalLayer {

private:

    //kernel
    float* cuda_kernel;
    //kernel size in bytes
    uint32_t cuda_kernel_size;
    uint32_t kernel_size;

    //number of feature maps
    uint32_t feature_maps;
    //number of feature maps before
    uint32_t maps_before;
    //how many cons back 
    uint32_t to_prev_cons;

    //randomized indexes
    uint32_t* con_ref;

    uint32_t kernel_x;
    uint32_t kernel_y;

    uint32_t input_x;
    uint32_t input_y;

    uint32_t output_x;
    uint32_t output_y;

public:

    ConvolutionalLayer(
            uint32_t inp_x, uint32_t inp_y, uint32_t maps_before, uint32_t feature_maps,
            uint32_t const_to_prev, uint32_t kernel_dim, Allocator& alloc);

    ~ConvolutionalLayer();

    ConvolutionalLayer(ConvolutionalLayer& other) = delete;

};
