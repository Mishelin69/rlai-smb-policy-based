#include <device_atomic_functions.h>
#ifndef WARP_SIZE
    #define WARP_SIZE 32
#endif

__global__
void vector_outer_v1(float* output, float* a, float* b, int dim_len, int right) {

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    const int nth_elm = y * WARP_SIZE + x;

    //TURN THIS INTO FOR LOOP :)
    if (nth_elm < dim_len) {

        for (int i = 0; i < right; ++i) {
            output[nth_elm+i] = a[nth_elm] * b[nth_elm + i];
        }
    }
};

__global__
void multiply_by_scalar(float* out, float* a, float scalar, int n_items) {

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    const int nth_elm = y * WARP_SIZE + x;

    if (nth_elm < n_items) {
        out[nth_elm] = a[nth_elm] * scalar;
    }
};

__global__ 
void sum_vector_one(float* out, float* a, int n_items) {

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    const int nth_elm = y * WARP_SIZE + x;

    if (nth_elm < n_items) {
        atomicAdd(&out[0], a[nth_elm]);
    }
}

__global__ 
void sum_bias_conv(float* out, float* a, int filter_size, int n_items) {

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    const int nth_elm = y * WARP_SIZE + x;

    if (nth_elm < n_items) {

        for (int i = 0; i < filter_size; ++i) {
            for (int j = 0; j < filter_size; ++j) {
                out[nth_elm] += a[nth_elm*filter_size*filter_size + i*filter_size + j];
            }
        }
    }
 

}

