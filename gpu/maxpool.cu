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

__global__
void pool_ver2(const float* input, float* out, int* idx, int in_dim, int out_dim, int pool_size) {

    const int id_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int id_y = blockIdx.y * blockDim.y + threadIdx.y;

    const int x = id_x % out_dim;
    const int y = (32*id_y + id_x) / out_dim;

    if (x < out_dim && y < out_dim) {

        float max = input[y*in_dim*in_dim + x*pool_size];
        int max_index = y*in_dim*in_dim + x*pool_size;

        for (int i = 0; i < pool_size; ++i) {
            for (int j = 0; j < pool_size; ++j) {

                if (input[y*in_dim*in_dim + x*pool_size + i*in_dim + j] > max) {

                    max = input[y*in_dim*in_dim + x*pool_size + i*in_dim + j];
                    max_index = y*in_dim*in_dim + x*pool_size + i*in_dim + j;
                }
            }
        }

        out[y*out_dim + x] = max;
        idx[y*out_dim + x] = max_index;
    }

}
