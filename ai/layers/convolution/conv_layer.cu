#include "./conv_layer.hpp"
#include "../../allocator/allocator.hpp"
#include <cuda_runtime_api.h>
#include <iostream>

std::pair<uint32_t, uint32_t> ConvolutionalLayer::calc_output_size(
        const uint32_t kernel_x, const uint32_t kernel_y,
        const uint32_t input_x, const uint32_t input_y, const uint32_t kernel_shift) noexcept {

    const uint32_t _shft = kernel_shift - 1;

    const uint32_t out_x = input_x + 1 - (kernel_x + _shft);
    const uint32_t out_y = input_x + 1 - (kernel_x + _shft);

    return std::pair<uint32_t, uint32_t> { out_x, out_y };
}

ConvolutionalLayer::ConvolutionalLayer(GPU::Device& gpu, GPU::ActivationFunction actv_func,
            const uint32_t maps_before, const uint32_t feature_maps, 
            const uint32_t cons_to_prev, const uint32_t kernel_dim, 
            const uint32_t kernel_shift, Allocator& alloc, cudaStream_t stream)
    : gpu(gpu), maps_before(maps_before), feature_maps(feature_maps), actv_func(actv_func),
      kernel_x(kernel_dim), kernel_y(kernel_dim), kernel_shift(kernel_shift), stream(stream) {

    //calculate the amount of memory needed (in bytes) to keep the stuff in
    const uint32_t kernel_size_bytes = sizeof(float) * kernel_dim * kernel_dim;
    const uint32_t mem_size = kernel_dim * kernel_dim * feature_maps;
    this->cuda_kernel_size = kernel_size_bytes;

    float* cuda_p = alloc.alloc_space(mem_size);

    if (!cuda_p) {
        std::cerr << "Error allocating memory on the gpu!! exiting !!!" << std::endl;
        exit(-1);
    }
    
    this->cuda_kernel = cuda_p;

    //this will be slow but whatever
    int res = gpu.random_numbers(this->cuda_kernel, feature_maps * kernel_x * kernel_y);

    if (res != 0) {
        std::cerr << "ConvolutionalLayer::ConvolutionalLayer() | Error while trying to initialize kernel data!" << std::endl;
        exit(-1);
    }
}

void ConvolutionalLayer::convolve(float* a, float* out, 
        const uint32_t a_x, const uint32_t a_y) const noexcept {

    const std::pair<uint32_t, uint32_t> out_dims = ConvolutionalLayer::calc_output_size(
            this->kernel_x, this->kernel_y, a_x, a_y, this->kernel_shift
    );

    const uint32_t dim_x = out_dims.first;
    const uint32_t dim_y = out_dims.second;

    //ik using this is stupid but I like it, it makes things more explicit
    //this makes convolution go in one go, calling them separately would make them
    //either wait or use multiple stream, assuming I want to process multiple pieces
    //of data at once (batch) this would most definitely max out if not go past
    //the stream limit (a necessity in this case)
    gpu.batched_conv_ver1( 
            this->cuda_kernel,
            a,
            out, 
            this->kernel_x,
            a_x,
            dim_x,
            this->actv_func,
            this->feature_maps,
            a_x * a_y,
            this->maps_before,
            this->stream
        );

    //wait for the GPU to finish it's job (stream)
    //keep the data on the GPU tho
    cudaStreamSynchronize(this->stream);
}
