#include "device_launch_parameters.h"

#include <cmath>
#include <stdio.h>


__global__
void matmul_preactv_v1(float* A, float* B, float* C, 
        size_t a_col, size_t a_row, size_t b_col, size_t b_row, size_t c_col, size_t c_row) {

	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < c_col && y < c_row) {

		float tmp = 0.0;

		for (size_t i = 0; i < a_col; ++i) {

			int a_index = y * a_col + i;
			int b_index = i * b_col + x;

            //printf("%d %d %lld | x: %d y: %d | %f | %f\n", a_index, b_index, y*c_col + x, x, y, A[a_index], B[b_index]);
			tmp += A[a_index] * B[b_index];
		}
        //printf("Outputting to %lld %d %d\n", x*c_row + y, x, y);
		C[y*c_col + x] = (tmp+C[y*c_col + x] > 0) ? 1.0 : 0.0;
    }	
}

__global__
void matmul_v1(float* A, float* B, float* C,
	size_t a_col, size_t a_row, size_t b_col, size_t b_row, size_t c_col, size_t c_row) {

	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < c_col && y < c_row) {

		float tmp = 0.0;

		for (size_t i = 0; i < a_col; ++i) {

			int a_index = y * a_col + i;
			int b_index = i * b_col + x;

            //printf("%d %d %lld | x: %d y: %d | %f | %f\n", a_index, b_index, y*c_col + x, x, y, A[a_index], B[b_index]);
			tmp += A[a_index] * B[b_index];
		}
        //printf("Outputting to %lld %d %d\n", x*c_row + y, x, y);
		C[y*c_col + x] += tmp;
    }	
}

__global__
void matmul_v1_ReLU(float* A, float* B, float* C,
	size_t a_col, size_t a_row, size_t b_col, size_t b_row, size_t c_col, size_t c_row) {

	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < c_col && y < c_row) {

		float tmp = 0.0;

		for (size_t i = 0; i < a_col; ++i) {

			int a_index = y * a_col + i;
			int b_index = i * b_col + x;

            //printf("%d %d %lld | x: %d y: %d | %f | %f\n", a_index, b_index, y*c_col + x, x, y, A[a_index], B[b_index]);
			tmp += A[a_index] * B[b_index];
		}
        //printf("Outputting to %lld %d %d\n", x*c_row + y, x, y);
		C[y*c_col + x] = (tmp+C[y*c_col + x] >= 0) ? (tmp+C[y*c_col + x]) : 0;
    }	
}

__global__
void matmul_v1_Sigmoid(float* A, float* B, float* C,
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

		C[x*c_col + y] += 1.0 / (1.0 + exp(-tmp));
    }	
}

__global__
void matmul_v1_DerReLU(float* A, float* B, float* C,
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

		C[x*c_col + y] += (tmp > 0) ? 1 : 0;
    }	
}

__global__
void matmul_v1_DerSigmoid(float* A, float* B, float* C,
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

		C[x*c_col + y] += tmp * (1.0 - tmp);
    }	
}
