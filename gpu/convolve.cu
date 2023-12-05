#include "device_launch_parameters.h"
#include <stdio.h>

#ifndef MAX_KERNEL_SIZE
#define MAX_KERNEL_SIZE 12*12
#endif

__global__
void convolve_v1(const float* k, const float* m, float* o, int kx, int mx, int ox) {

    //using __shared__ because I saw it somewhere and made sense at the time idk if it'll actually be beneficial
    //but local cache is 48k so most likely it won't be a problem but still better than accessing global all the time
    __shared__ float sharedKernel[MAX_KERNEL_SIZE];
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

    //load the kernel and check if x and y are in bound of kernel
    //then wait for this to be loaded and in sync 
    //to make sure that the kernel is properly loaded in
    if (y*kx + x < kx*kx) {
        sharedKernel[y*kx + x] = k[y*kx + x];
    }
    __syncthreads();

    //we only expect square matrices
    if (x < mx && y < mx) {

        float sum = 0;

        for (size_t rows = 0; rows < kx; ++rows) {
            for (size_t cols = 0; cols < kx; ++cols) {
                sum += sharedKernel[kx*rows + cols] * m[y*mx + rows*mx + cols + x];
            }
        }
        
        o[y*ox + x] = sum;
   }
}

