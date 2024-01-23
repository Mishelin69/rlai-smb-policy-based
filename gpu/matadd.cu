#include "device_launch_parameters.h"

#include <cstdint>
#include <stdio.h>

__global__
void matadd_v1(float* A, float* B, float* C,
	size_t a_col, size_t a_row, size_t b_col, size_t b_row, size_t c_col, size_t c_row) {

	const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
	const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < c_col && y < c_row) {
		C[y * c_row + x] = A[y * c_row + x] + B[y * c_row + x];
	}
}

__global__
void conv_add_gpu(float* A, float* B, int a_dim, int b_dim) {

    const int id_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int id_y = blockIdx.y * blockDim.y + threadIdx.y;

    const int x = id_x % b_dim;
    const int y = (32*id_y + id_x) / b_dim;

    if (x < b_dim && y < b_dim) {

        B[y*b_dim + x] = B[y*b_dim + x] + A[y*b_dim + x];
    }

}
