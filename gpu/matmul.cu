#include "device_launch_parameters.h"

#include <stdio.h>

__global__
void matmul_v1(float* A, float* B, float* C,
	size_t a_col, size_t a_row, size_t b_col, size_t b_row, size_t c_col, size_t c_row) {

	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < c_col && y < c_row) {

		float tmp = 0.0;

		for (size_t i = 0; i < a_col; ++i) {

			int a_index = x * a_col + i;
			int b_index = i * b_col + y;

			tmp += A[a_index] * B[b_index];
		}

		C[x*c_col + y] = tmp;
	}
}
