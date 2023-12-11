#include "device_launch_parameters.h"

__global__
void max_pooling_ver1(const float* a, const float* out, const size_t inputs, 
            const size_t inp_shape, const size_t pool_size) {

    const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;


}
