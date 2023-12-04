#include "./conv_layer.hpp"
#include "../../allocator/allocator.hpp"
#include <iostream>

std::pair<uint32_t, uint32_t> ConvolutionalLayer::calc_output_size(
        const uint32_t kernel_x, const uint32_t kernel_y,
        const uint32_t input_x, const uint32_t input_y, const uint32_t kernel_shift) noexcept {

    const uint32_t _shft = kernel_shift - 1;
    const uint32_t half = kernel_x / 2;

    const uint32_t out_x = input_x + 1 - (kernel_x + _shft);
    const uint32_t out_y = input_x + 1 - (kernel_x + _shft);

    return std::pair<uint32_t, uint32_t> { out_x, out_y };
}

ConvolutionalLayer::ConvolutionalLayer(
            const uint32_t maps_before, const uint32_t feature_maps, 
            const uint32_t cons_to_prev, const uint32_t kernel_dim, 
            const uint32_t kernel_shift, Allocator& alloc, const uint32_t* con_ref) {

    this->maps_before = maps_before;
    this->feature_maps = feature_maps;
    this->to_prev_cons = cons_to_prev;
    this->kernel_x = kernel_dim;
    this->kernel_y = kernel_dim;
    this->con_ref = con_ref;
    this->kernel_shift = kernel_shift;

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
}

void ConvolutionalLayer::convolve(float* a, float* out, const uint32_t a_x,
        const uint32_t a_y, const uint32_t offset) const noexcept {

    const std::pair<uint32_t, uint32_t> out_dims = ConvolutionalLayer::calc_output_size(
            this->kernel_x, this->kernel_y, a_x, a_y, this->kernel_shift
    );

    const uint32_t dim_x = out_dims.first;
    const uint32_t dim_y = out_dims.second;


}
