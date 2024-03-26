#include "./dense_layer.hpp"
#include <cuda_runtime_api.h>
#include <iostream>

DenseLayer::DenseLayer():
    gpu(DummyDevice) {

}

DenseLayer::DenseLayer(GPU::Device& gpu, float* cuda_w, float* cuda_b, const size_t neurons, 
        const size_t input, const GPU::ActivationFunction actv_func, const GPU::ActivationFunction der_actv_func)
    : gpu(gpu), input_shape(input), neurons(neurons), actv_func(actv_func), der_actv_func(der_actv_func) {

        this->mat_y = neurons;
        this->mat_x = input;
        this->biases = neurons;

        float* cudaMat = cuda_w; 

        int res = gpu.random_numbers(cudaMat, mat_y * mat_x);

        //!!res would be crazy but correct :D since only 0 evals as false (talking numbers ofc)
        //(negative numbers eval to true since they hold some value :| )
        if (res != 0) {
            std::cerr << "DenseLayer::DenseLayer() | Error: Error in initializing neurons!!" << std::endl; 
        }

        float* cudaBias = cuda_b;
        res = gpu.random_numbers(cudaBias, biases);

        if (res != 0) {
            std::cerr << "DenseLayer::DenseLayer() | Error: Error in initializing biases!!" << std::endl; 
        }
    }

void DenseLayer::init_self(GPU::Device& gpu, float* cuda_w, float* cuda_b, const size_t neurons, 
        const size_t input, const GPU::ActivationFunction actv_func, 
        const GPU::ActivationFunction der_actv_func) {

    this->gpu = gpu;
    this->input_shape = input;
    this->neurons = neurons;
    this->actv_func = actv_func;
    this->der_actv_func = der_actv_func;

    this->mat_y = neurons;
    this->mat_x = input;
    this->biases = neurons;

    float* cudaMat = cuda_w; 

    int res = gpu.random_numbers(cudaMat, mat_y * mat_x);

    //!!res would be crazy but correct :D since only 0 evals as false (talking numbers ofc)
    //(negative numbers eval to true since they hold some value :| )
    if (res != 0) {
        std::cerr << "DenseLayer::DenseLayer() | Error: Error in initializing neurons!!" << std::endl; 
    }

    float* cudaBias = cuda_b;
    res = gpu.random_numbers(cudaBias, biases);

    if (res != 0) {
        std::cerr << "DenseLayer::DenseLayer() | Error: Error in initializing biases!!" << std::endl; 
    }

    this->cudaMat = cudaMat;
    this->cudaBias = cudaBias;

}

void DenseLayer::deep_copy(const DenseLayer& original) {

    cudaMemcpy(this->cudaMat, original.cudaMat, sizeof(float) * mat_x * mat_y, cudaMemcpyDeviceToDevice);
    cudaMemcpy(this->cudaMat, original.cudaMat, sizeof(float) * biases, cudaMemcpyDeviceToDevice);

}


void DenseLayer::passthrough(float* a, float* out, const cudaStream_t stream) const noexcept {

    std::pair<size_t, size_t> out_shape = GPU::Device::calculate_new_mat_dims(input_shape, 1, mat_y, mat_x); 

    size_t out_y = out_shape.first;
    size_t out_x = out_shape.second;

    //std::cout << "INP: " << input_shape << " " << 1 << std::endl;
    //std::cout << "MAT: " << mat_x << " " << mat_y << std::endl;
    //std::cout << "OUT: " << out_x << " " << out_y << std::endl;

    if (!a || !out) {
        std::cerr << "DenseLayer::passthrough | NULL POINTER!!" << std::endl;
        exit(-1);
    }

    if (!cudaMat || !cudaBias) {
        std::cerr << "DenseLayer::passthrough | NULL POINTER!! (w/b)" << std::endl;
        exit(-1);
    }

    //std::cout << "OUTPUT Y: " << out_y << " OUTPUT X: " << out_x << std::endl;

    gpu.matmul_ver1_gpu(
            a,
            this->cudaMat,
            out,
            this->input_shape,
            1,
            this->mat_y,
            this->mat_x,
            out_y,
            out_x,
            this->actv_func,
            stream
            );

    gpu.matadd_ver1(
            this->cudaBias,
            a,
            out,
            this->biases,
            1,
            this->biases,
            1,
            out_y, //this should match but worst scenario I get an error :chomik_xmas: yaaah indeed was incorrect
            1,
            stream
            );

    //exit(-1);
}

//fix this to do the correct thing :(
//yeah its not doing that buddy :)
//Ill need to adjust it, to be more modular or something else idk make either a diff 
//function or just do it differentlya (pain) Ill figure it out later (soon pain again)
void DenseLayer::gradient_calculation(const GPU::Tensor activations, 
        const GPU::Tensor gradient, GPU::Tensor out, const cudaStream_t stream) const noexcept {

    gpu.matmul_ver1_gpu(
            gradient.dat_pointer, 
            cudaMat, 
            out.dat_pointer, 
            gradient.dat_x, 
            1, 
            1, 
            mat_y, 
            out.dat_x, 
            out.dat_y, 
            this->actv_func, 
            stream
            );

    //the end thing where you multiply by funciton derivartive the original output !:)
    gpu.matmul_elementwise(out, activations, out, stream, der_actv_func);

    cudaStreamSynchronize(stream);
}
