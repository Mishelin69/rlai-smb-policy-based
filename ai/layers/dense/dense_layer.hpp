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
            const size_t neurons, const size_t input, const GPU::ActivationFunction actv_func);
    DenseLayer(DenseLayer& other) = default;

    //actually no problem since memory is on the gpu
    ~DenseLayer() = default;

    void passthrough(const float* a, float* out) const noexcept;
};
