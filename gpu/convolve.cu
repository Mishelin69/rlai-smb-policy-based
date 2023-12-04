#include "device_launch_parameters.h"

#ifndef MAX_KERNEL_SIZE
#define MAX_KERNEL_SIZE 12*12
#endif

__global__
void convolve_v1(const float* k, const float* m, float* o, int kx, int mx, int iter) {

    __shared__ float sharedKernel[MAX_KERNEL_SIZE];
    const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
	const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;

    //load the kernel and check if x and y are in bound of kernel
    //then wait for this to be loaded and in sync 
    //to make sure that the kernel is properly loaded in
    if (x*kx + y*kx < kx*kx) {
        sharedKernel[x*kx + y*kx] = k[x*kx + y*kx];
    }
    __syncthreads();

    //we only expect square matrices
    if (x < kx && y < kx) {

        float sum = 0;

        for (size_t cols = 0; cols < kx; ++cols) {
            for (size_t rows = 0; rows < kx; ++rows) {

            }
        }
    }
}

