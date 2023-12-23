#pragma once
#include "../../../gpu/gpu.h"
#include "driver_types.h"

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
    GPU::ActivationFunction der_actv_func;
    cudaStream_t stream;

public:

    DenseLayer(GPU::Device& gpu, Allocator& alloc, const size_t neurons, 
            const size_t input, const GPU::ActivationFunction actv_func, const GPU::ActivationFunction der_actv_func);
    DenseLayer(DenseLayer& other) = default;

    //actually no problem since memory is on the gpu
    ~DenseLayer() = default;

    void passthrough(float* a, float* out , const cudaStream_t stream) const noexcept;

    void gradient_calculation(const GPU::Tensor w_layer_before, const GPU::Tensor activations, 
            const GPU::Tensor gradient, GPU::Tensor out, const cudaStream_t stream) const noexcept;
};
