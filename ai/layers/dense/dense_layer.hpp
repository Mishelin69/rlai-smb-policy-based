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

    void init_self(GPU::Device& gpu, float* cuda_w, float* cuda_b, const size_t neurons, 
            const size_t input, const GPU::ActivationFunction actv_func, const GPU::ActivationFunction der_actv_func);

    DenseLayer(GPU::Device& gpu, float* cuda_w, float* cuda_b, const size_t neurons, 
            const size_t input, const GPU::ActivationFunction actv_func, const GPU::ActivationFunction der_actv_func);
    DenseLayer(DenseLayer& other) = default;

    //actually no problem since memory is on the gpu
    ~DenseLayer() = default;

    void passthrough(float* a, float* out , const cudaStream_t stream) const noexcept;

    //THIS NEEDS TO BE REWORKED AND SO DOES EVERY GRADIENT CALCULATION (IF ANY IDK?)
    //TO WORK WITH MY CODE (OTHERWISE ITS FINE)
    //SEPARATE GwR TO THE OUTPUT
    void gradient_calculation(const GPU::Tensor activations, 
            const GPU::Tensor gradient, GPU::Tensor out, const cudaStream_t stream) const noexcept;
};
