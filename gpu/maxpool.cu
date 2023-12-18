#include "device_launch_parameters.h"

__global__
void max_pooling_ver1(const float* a, float* out, size_t* out_idx, const size_t pool_size,
        const size_t input_dim, const size_t output_dim, const size_t z_dim) {

    const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < output_dim && y < z_dim * output_dim) {

        float max = a[y*output_dim + x*pool_size];
        size_t max_index = y*output_dim + x;

        for (size_t i = 0; i < pool_size; ++i) {
            for (size_t j = 0; j < pool_size; ++j) {

                if (a[(y+j) * output_dim + (x+i)*pool_size] > max) {
                    max = a[(y+j) * output_dim + (x+i)*pool_size];
                }

            }
        }

        out[y*output_dim + x] = max;
        out_idx[y*output_dim + x] = max_index;
    }
}
