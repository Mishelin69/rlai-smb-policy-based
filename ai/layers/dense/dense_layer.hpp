#pragma once
#include "../../../gpu/gpu.h"

class Allocator;

class DenseLayer {

private:

    float* cudaMat;
    float* cudaBias;

    size_t mat_x;
    size_t mat_y;
    size_t neurons;
    size_t input_shape;
    size_t biases;

    GPU::Device& gpu;
    GPU::ActivationFunction actv_func;

public:

    DenseLayer(GPU::Device& gpu, Allocator& alloc,
            size_t neurons, size_t input, GPU::ActivationFunction actv_func);
    DenseLayer(DenseLayer& other) = default;
    ~DenseLayer();
};
