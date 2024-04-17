#include "device_launch_parameters.h"
#include <stdio.h>

__global__
void max_pooling_ver1(const float* a, float* out, size_t* out_idx, const size_t pool_size,
        const size_t input_dim, const size_t output_dim, const size_t z_dim) {

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

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

        float max = input[y*in_dim*pool_size + x*pool_size];
        int max_index = y*in_dim*pool_size + x*pool_size;

        for (int i = 0; i < pool_size; ++i) {
            for (int j = 0; j < pool_size; ++j) {

                int index = y*in_dim*pool_size + x*pool_size + i*in_dim + j;
                //printf("%f < %f | %d < %d\n", input[index], input[max_index], index, max_index);

                if (input[index] > max) {
                    max = input[index];
                    max_index = index;
                }
            }
        }

        //printf("idx: %d n: %f\n", y*out_dim + x, max);
        //printf("x: %d y: %d | %p\n", x, y, input);
        out[y*out_dim + x] = max;
        idx[y*out_dim + x] = max_index;
    }

}

__global__
void unpooling_v1(float* out, size_t* indices, float* loss, int output_dim, int in_dim, int pool_size) {

    const int id_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int id_y = blockIdx.y * blockDim.y + threadIdx.y;

    const int x = id_x % output_dim;
    const int y = (32*id_y + id_x) / output_dim;

    if (x < output_dim && y < output_dim) {

        size_t index = indices[y*in_dim + x];

        //uuuuuuhhhh annoying indexing, not a fan :C
        //so it wasnt that annoying after all once you realize you already have the indexes
        //I mean I'm quite literraly storing THE indexes into an array so ???
        out[index] = loss[y*in_dim + x];
    }
}
