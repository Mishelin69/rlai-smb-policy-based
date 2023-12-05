#pragma once
#include "../../../gpu/gpu.h"

class Allocator;

class ConvolutionalLayer {

private:

    //kernel
    float* cuda_kernel;
    //kernel size in byttes
    uint32_t cuda_kernel_size;
    //number of feature maps
    uint32_t feature_maps;
    //number of feature maps before
    uint32_t maps_before;

    uint32_t kernel_x;
    uint32_t kernel_y;
    uint32_t kernel_shift;

    GPU::Device& gpu;

public:

    ConvolutionalLayer(GPU::Device& gpu,
            const uint32_t maps_before, const uint32_t feature_maps, 
            const uint32_t cons_to_prev, const uint32_t kernel_dim, 
            const uint32_t kernel_shift, Allocator& alloc, const uint32_t* con_ref);

    ~ConvolutionalLayer() = default;

    ConvolutionalLayer(ConvolutionalLayer& other) = delete;

    static std::pair<uint32_t, uint32_t> calc_output_size(
            const uint32_t kernel_x, const uint32_t kernel_y,
            const uint32_t input_x, const uint32_t input_y, const uint32_t kernel_shift
        ) noexcept;

    void convolve(float* a, float* out, const uint32_t a_x,
            const uint32_t a_y, const uint32_t offset) const noexcept;

};
