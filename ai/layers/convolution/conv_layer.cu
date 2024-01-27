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
            const uint32_t depth, const uint32_t feature_maps, 
            const uint32_t input_chanels, const uint32_t kernel_dim, 
            const uint32_t input, Allocator& alloc)
    : gpu(gpu), feature_maps(feature_maps), actv_func(actv_func),
    kernel_x(kernel_dim), kernel_y(kernel_dim), input_chanels(input_chanels) {

        //calculate the amount of memory needed (in bytes) to keep the stuff in
        const uint32_t kernel_size_bytes = sizeof(float) * kernel_dim * kernel_dim;
        const uint32_t mem_size = kernel_dim * kernel_dim * feature_maps * input_chanels;
        this->cuda_kernel_size = kernel_size_bytes;

        float* cuda_p = alloc.alloc_space(mem_size);

        if (!cuda_p) {
            std::cerr << "Error allocating memory on the gpu!! exiting !!!" << std::endl;
            exit(-1);
        }

        this->cuda_kernel = cuda_p;

        auto out = calc_output_size(kernel_dim, kernel_dim, input, input, 1);
        auto out_x = out.first;

        cuda_p = alloc.alloc_space(feature_maps * out_x * out_x);

        if (!cuda_p) {
            std::cerr << "Error allocating memory on the gpu!! exiting !!!" << std::endl;
            exit(-1);
        }

        this->cuda_bias = cuda_p;

        for (uint32_t i = 0; i < feature_maps; ++i) {
            this->filters.push_back(ConvFilter {
                    this->cuda_kernel + input_chanels * kernel_x * kernel_y,
                    input_chanels
                    });
        }

        //this will be slow but whatever
        int res = gpu.random_numbers(this->cuda_kernel, input_chanels * feature_maps * kernel_x * kernel_y);

        if (res != 0) {
            std::cerr << "ConvolutionalLayer::ConvolutionalLayer() | Error while trying to initialize kernel data!" << std::endl;
            exit(-1);
        }

        res = gpu.random_numbers(this->cuda_bias, feature_maps * out_x * out_x);

        if (res != 0) {
            std::cerr << "ConvolutionalLayer::ConvolutionalLayer() | Error while trying to initialize bias data!" << std::endl;
            exit(-1);
        }
    }

//fix
//now correctly does the convolution plus adds the bias :)
//at least it should do this, wherever this will be done correctly idk
void ConvolutionalLayer::convolve(GPU::Tensor a, GPU::Tensor b, float* out, cudaStream_t stream) const noexcept {

    const std::pair<uint32_t, uint32_t> out_dims = ConvolutionalLayer::calc_output_size(
            b.dat_x, b.dat_y, a.dat_x, a.dat_y, 1
            );

    const uint32_t dim_x = out_dims.first;
    const uint32_t dim_y = out_dims.second;

    //queue up jobs and wait for them to finish
    for (size_t i = 0; i < this->feature_maps; ++i) {
        this->gpu.conv_ver2(
                a, 
                GPU::Tensor { 
                this->filters[i].cuda_kernels, 
                this->kernel_x,
                this->kernel_y, 
                this->filters[i].depth 
                },

                GPU::Tensor {
                out + i*dim_x*dim_y,
                dim_x,
                dim_y,
                1
                }, 0, stream);

        this->gpu.conv_add(
                GPU::Tensor { 
                this->cuda_bias + dim_x*dim_y*i, 
                dim_x,
                1,//1 because same bias for everything 
                1 
                },

                GPU::Tensor {
                out + i*dim_x*dim_y,
                dim_x,
                dim_y,
                1
                }, stream);
    }

    //wait for the GPU to finish it's job (stream)
    //keep the data on the GPU tho
    cudaStreamSynchronize(stream);
}
