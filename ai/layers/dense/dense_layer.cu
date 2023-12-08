#include "./dense_layer.hpp"
#include "../../allocator/allocator.hpp"
#include <iostream>

DenseLayer::DenseLayer(GPU::Device& gpu, Allocator& alloc, const size_t neurons, 
        const size_t input, const GPU::ActivationFunction actv_func)
    : gpu(gpu), input_shape(input), neurons(neurons) {

    this->mat_y = neurons;
    this->mat_x = input;
    this->biases = neurons;

    float* cudaMat = alloc.alloc_space(mat_y * mat_x);

    if (!cudaMat) {
        std::cerr << "DenseLayer::DenseLayer() | Error: Couldn't allocate memory for neurons!!" << std::endl;
        exit(-1);
    }

    int res = gpu.random_numbers(cudaMat, mat_y * mat_x);

    //!!res would be crazy but correct :D since only 0 evals as false
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

void DenseLayer::passthrough(const float* a, float* out) const noexcept {
}
