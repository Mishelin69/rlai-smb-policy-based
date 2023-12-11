#include "./dense_layer.hpp"
#include "../../allocator/allocator.hpp"
#include <cuda_runtime_api.h>
#include <iostream>

DenseLayer::DenseLayer(GPU::Device& gpu, Allocator& alloc, const size_t neurons, 
        const size_t input, const GPU::ActivationFunction actv_func, const cudaStream_t stream)
    : gpu(gpu), input_shape(input), neurons(neurons), stream(stream) {

    this->mat_y = neurons;
    this->mat_x = input;
    this->biases = neurons;

    float* cudaMat = alloc.alloc_space(mat_y * mat_x);

    if (!cudaMat) {
        std::cerr << "DenseLayer::DenseLayer() | Error: Couldn't allocate memory for neurons!!" << std::endl;
        exit(-1);
    }

    int res = gpu.random_numbers(cudaMat, mat_y * mat_x);

    //!!res would be crazy but correct :D since only 0 evals as false (talking numbers ofc)
    //(negative numbers eval to true since they hold some value :| )
    if (res != 0) {
        std::cerr << "DenseLayer::DenseLayer() | Error: Error in initializing neurons!!" << std::endl; 
    }

    float* cudaBias = alloc.alloc_space(biases);

    if (!cudaBias) {
        std::cerr << "DenseLayer::DenseLayer() | Error: Couldn't allocate memory for biases!!" << std::endl;
        exit(-1);
    }

    res = gpu.random_numbers(cudaBias, biases);

    if (res != 0) {
        std::cerr << "DenseLayer::DenseLayer() | Error: Error in initializing biases!!" << std::endl; 
    }
}

void DenseLayer::passthrough(float* a, float* out) const noexcept {

    std::pair<size_t, size_t> out_shape = GPU::Device::calculate_new_mat_dims(mat_x, mat_y, input_shape, input_shape); 

    size_t out_y = out_shape.first;
    size_t out_x = out_shape.second;

    gpu.matmul_ver1_gpu(
            this->cudaMat,
            a,
            out,
            this->mat_y,
            this->mat_x,
            1,
            this->input_shape,
            out_y,
            out_x,
            this->actv_func,
            this->stream
    );

    gpu.matadd_ver1(
            this->cudaBias,
            a,
            out,
            this->biases,
            1,
            biases,
            1,
            out_y, //this should match but worst scenario I get an error :chomik_xmas:
            1,
            this->stream
    );

    cudaStreamSynchronize(stream);
}
