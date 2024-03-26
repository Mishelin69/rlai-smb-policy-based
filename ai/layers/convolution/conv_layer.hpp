#pragma once
#include "../../../gpu/gpu.h"
#include "driver_types.h"
#include <vector>

class Allocator;

struct ConvFilter {

    float* cuda_kernels;
    const uint32_t depth;
};

class ConvolutionalLayer {

private:

    //kernel
    float* cuda_kernel;

    //biases gpu
    float* cuda_bias;

    //filters
    std::vector<ConvFilter> filters;

    //kernel size in bytes
    uint32_t cuda_kernel_size;

    //number of feature maps
    uint32_t feature_maps;
    uint32_t input_chanels;

    uint32_t kernel_x;
    uint32_t kernel_y;

    GPU::Device& gpu = DummyDevice;
    GPU::ActivationFunction actv_func;

public:

    void convolve(GPU::Tensor a, GPU::Tensor b, float* out, cudaStream_t stream) const noexcept;

    void init_self(GPU::Device& gpu, GPU::ActivationFunction func,
            const uint32_t feature_maps, 
            const uint32_t input_chanels, const uint32_t kernel_dim, 
            const uint32_t input, float* cuda_w, float* cuda_b);

    void deep_copy(const ConvolutionalLayer& other);

    ConvolutionalLayer();

    ConvolutionalLayer(GPU::Device& gpu, GPU::ActivationFunction func,
            const uint32_t feature_maps, 
            const uint32_t input_chanels, const uint32_t kernel_dim, 
            const uint32_t input, float* cuda_w, float* cuda_b);

    //no need for anything special since memory is on the gpu
    ~ConvolutionalLayer() = default;

    ConvolutionalLayer(ConvolutionalLayer& other) = default;
    //ConvolutionalLayer(const ConvolutionalLayer& other) = default;

    static std::pair<uint32_t, uint32_t> calc_output_size(
            const uint32_t kernel_x, const uint32_t kernel_y,
            const uint32_t input_x, const uint32_t input_y, const uint32_t kernel_shift
        ) noexcept;

};
