#include "./dense_layer.hpp"
#include "../../allocator/allocator.hpp"
#include <iostream>

DenseLayer::DenseLayer(GPU::Device& gpu, Allocator& alloc, size_t neurons, 
        size_t input, GPU::ActivationFunction actv_func)
    : gpu(gpu), input_shape(input), neurons(neurons) {

    this->mat_y = neurons;
    this->mat_x = input;
    this->biases = neurons;

    float* cudaMat = alloc.alloc_space(mat_y * mat_x);

    if (!cudaMat) {
        std::cerr << "DenseLayer::DenseLayer() | Error: Couldn't allocate memory for neurons!!" << std::endl;
        exit(-1);
    }

    float* cudaBias = alloc.alloc_space(biases);

    if (!cudaMat) {
        std::cerr << "DenseLayer::DenseLayer() | Error: Couldn't allocate memory for biases!!" << std::endl;
        exit(-1);
    }


}
