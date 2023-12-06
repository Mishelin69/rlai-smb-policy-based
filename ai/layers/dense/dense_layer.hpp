#pragma once
#include "../../../gpu/gpu.h"
#include "../actv_func/actv_func.hpp"

class Allocator;

class DenseLayer {

private:

    float* cudaMat;
    float* cudaBias;

    size_t mat_x;
    size_t mat_y;
    size_t neurons;
    size_t inputs;
    size_t biases;

    GPU::Device& gpu;
    ActivationFunction actv_func;

public:

    DenseLayer(GPU::Device& gpu, Allocator& alloc,
            size_t neurons, size_t input);
    DenseLayer(DenseLayer& other) = default;
    ~DenseLayer();
};
