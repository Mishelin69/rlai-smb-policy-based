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
    if (x < ox && y < ox) {

        float sum = 0;

        for (size_t rows = 0; rows < kx; ++rows) {
            for (size_t cols = 0; cols < kx; ++cols) {
                sum += sharedKernel[kx*rows + cols] * m[y*mx + rows*mx + cols + x];
            }
        }

        o[y*ox + x] += sum;
    }
}

__global__
void convolve_v1_ReLU(const float* k, const float* m, float* o, int kx, int mx, int ox) {

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
    if (x < ox && y < ox) {

        float sum = 0;

        for (size_t rows = 0; rows < kx; ++rows) {
            for (size_t cols = 0; cols < kx; ++cols) {
                sum += sharedKernel[kx*rows + cols] * m[y*mx + rows*mx + cols + x];
            }
        }

        o[y*ox + x] += (sum >= 0) ? sum : 0;
    }
}

__global__
void convolve_v1_Sigmoid(const float* k, const float* m, float* o, int kx, int mx, int ox) {

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
    if (x < ox && y < ox) {

        float sum = 0;

        for (size_t rows = 0; rows < kx; ++rows) {
            for (size_t cols = 0; cols < kx; ++cols) {
                sum += sharedKernel[kx*rows + cols] * m[y*mx + rows*mx + cols + x];
            }
        }

        o[y*ox + x] += 1.0 / (1.0 + exp(-sum));
    }
}

__global__
void batched_convolve_v1_ReLU(const float* k, const float* m, float* o, 
        int kx, int mx, int ox, const size_t b_size, const size_t n_elms) {

    //using __shared__ because I saw it somewhere and made sense at the time idk if it'll actually be beneficial
    //but local cache is 48k (actually around 100k on my GPU but the point still stands) 
    //so most likely it won't be a problem but still better than accessing global all the time
    extern __shared__ float sharedKernel[];
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    //load the kernel and check if x and y are in bound of kernel
    //then wait for this to be loaded and in sync 
    //to make sure that the kernel is properly loaded in
    if (x < kx && y < kx*n_elms) {
        sharedKernel[y*kx + x] = k[y*kx + x];
    }
    __syncthreads();

    //we only expect square matrices
    if (x < ox && y < ox * n_elms) {

        //Index of the element inside batch
        const size_t nth_elm = (y*ox + x) / b_size;
        float sum = 0;

        for (size_t rows = 0; rows < kx; ++rows) {
            for (size_t cols = 0; cols < kx; ++cols) {

                const size_t kernel_index = nth_elm*kx*kx + kx*rows + cols;
                const size_t mat_index = y*mx + rows*mx + cols + x;

                sum += sharedKernel[kernel_index] * m[mat_index];
            }
        }

        o[y*ox + x] += (sum >= 0) ? sum : 0;
    }
}

__global__
void batched_convolve_v1_Sigmoid(const float* k, const float* m, float* o, 
        int kx, int mx, int ox, const size_t b_size, const size_t n_elms) {

    //using __shared__ because I saw it somewhere and made sense at the time idk if it'll actually be beneficial
    //but local cache is 48k (actually around 100k on my GPU but the point still stands) 
    //so most likely it won't be a problem but still better than accessing global all the time
    extern __shared__ float sharedKernel[];
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    //load the kernel and check if x and y are in bound of kernel
    //then wait for this to be loaded and in sync 
    //to make sure that the kernel is properly loaded in
    if (x < kx && y < kx*n_elms) {
        sharedKernel[y*kx + x] = k[y*kx + x];
    }
    __syncthreads();

    //we only expect square matrices
    if (x < ox && y < ox * n_elms) {

        //Index of the element inside batch
        const size_t nth_elm = (y*ox + x) / b_size;
        float sum = 0;

        for (size_t rows = 0; rows < kx; ++rows) {
            for (size_t cols = 0; cols < kx; ++cols) {

                const size_t kernel_index = nth_elm*kx*kx + kx*rows + cols;
                const size_t mat_index = y*mx + rows*mx + cols + x;

                sum += sharedKernel[kernel_index] * m[mat_index];
            }

        }        

        o[y*ox + x] += 1.0 / (1.0 + exp(-sum));
    }
}

//supports x*y bacthes not only x*x ! :) pain
__global__
void batched_convolve_v2_ReLU(const float* k, const float* m, float* o, 
        int kx, int mx, int ox, const size_t b_size, const size_t n_elms, const size_t inputs) {

    //using __shared__ because I saw it somewhere and made sense at the time idk if it'll actually be beneficial
    //but local cache is 48k (actually around 100k on my GPU but the point still stands) 
    //so most likely it won't be a problem but still better than accessing global all the time
    extern __shared__ float sharedKernel[];
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    //load the kernel and check if x and y are in bound of kernel
    //then wait for this to be loaded and in sync 
    //to make sure that the kernel is properly loaded in
    if (x < kx && y < kx*n_elms) {
        sharedKernel[y*kx + x] = k[y*kx + x];
    }
    __syncthreads();

    //we only expect square matrices
    if (x < ox && y < ox * n_elms) {

        //Index of the element inside batch
        //(y*ox + x) / (ox*ox) => (y) / (ox) since ox cancels out, ignore x since integer 
        //division rounds towards zero anyways
        const size_t nth_elm = (y*ox) / (ox*ox);
        float sum = 0;

        for (size_t rows = 0; rows < kx; ++rows) {
            for (size_t cols = 0; cols < kx; ++cols) {
                for (size_t i = 0; i < inputs; ++i) {

                    const size_t kernel_index = nth_elm*kx*kx + kx*rows + cols;
                    const size_t mat_index = i*b_size + (y - nth_elm * ox)*mx + rows*mx + cols + x;

                    sum += sharedKernel[kernel_index] * m[mat_index];
                }
            }
        }

        o[y*ox + x] += (sum >= 0) ? sum : 0;
    }
}

//supports x*y bacthes not only x*x ! :) pain
__global__
void batched_convolve_v2_Sigmoid(const float* k, const float* m, float* o, 
        int kx, int mx, int ox, const size_t b_size, const size_t n_elms, const size_t inputs) {

    //using __shared__ because I saw it somewhere and made sense at the time idk if it'll actually be beneficial
    //but local cache is 48k (actually around 100k on my GPU but the point still stands) 
    //so most likely it won't be a problem but still better than accessing global all the time
    extern __shared__ float sharedKernel[];
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    //load the kernel and check if x and y are in bound of kernel
    //then wait for this to be loaded and in sync 
    //to make sure that the kernel is properly loaded in
    if (x < kx && y < kx*n_elms) {
        sharedKernel[y*kx + x] = k[y*kx + x];
    }
    __syncthreads();

    //we only expect square matrices
    if (x < ox && y < ox * n_elms) {

        //Index of the element inside batch
        //(y*ox + x) / (ox*ox) => (y) / (ox) since ox cancels out, ignore x since integer 
        //division rounds towards zero anyways
        const size_t nth_elm = (y*ox) / (ox*ox);
        float sum = 0;

        for (size_t rows = 0; rows < kx; ++rows) {
            for (size_t cols = 0; cols < kx; ++cols) {
                for (size_t i = 0; i < inputs; ++i) {

                    const size_t kernel_index = nth_elm*kx*kx + kx*rows + cols;
                    const size_t mat_index = i*b_size + (y - nth_elm * ox)*mx + rows*mx + cols + x;

                    sum += sharedKernel[kernel_index] * m[mat_index];
                }
            }
        }

        o[y*ox + x] += 1.0 / (1.0 + exp(-sum));
    }
}

#ifndef MAX_KERNEL_SIZE
    #define MAX_KERNEL_SIZE 3*3
#endif

//maybe later Ill try to compare using 
//__shared and comparing it to this stupid thing
//just to see the performance diff
//Ill use __shared input since thats the only thing that these kernels share
//each ith kernel in a nth filter convolves with ith input
__global__
void conv_ReLU2(float* k, float* a, float* out, int k_size, 
        int a_size,  int out_size, int n_elms) {

    const int id_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int id_y = blockIdx.y * blockDim.y + threadIdx.y;

    const int x = id_x % out_size;
    const int y = (32*id_y + id_x) / out_size;

    if (x < out_size && y < out_size) {

        float sum = 0.0f;

        for (int n = 0; n < n_elms; ++n) {
            for (int i = 0; i < k_size; ++i) {
                for (int j = 0; j < k_size; ++j) {

                    int k_idx = n*k_size*k_size + i*k_size + j;
                    int i_idx = n*a_size*a_size + y*a_size + i*a_size + x + j;

                    //printf("k: %f a: %f | %d %d | %d %d\n", k[k_idx], a[i_idx], k_idx, i_idx, x, y);
                    sum += k[k_idx] * a[i_idx];

                }
            }
        }

        //printf("x: %d y: %d | o_x: %d o_y: %d | out: %d| n: %f\n", x, y, id_x, id_y, y * out_size + x, sum);
        out[y * out_size + x] = (sum+out[y * out_size + x] > 0) ? sum+out[y * out_size + x] : 0.0;
    }
}

__global__
void conv_pre(float* k, float* a, float* out, int k_size, 
        int a_size,  int out_size, int n_elms) {

    const int id_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int id_y = blockIdx.y * blockDim.y + threadIdx.y;

    const int x = id_x % out_size;
    const int y = (32*id_y + id_x) / out_size;

    if (x < out_size && y < out_size) {

        float sum = 0.0f;

        for (int n = 0; n < n_elms; ++n) {
            for (int i = 0; i < k_size; ++i) {
                for (int j = 0; j < k_size; ++j) {

                    int k_idx = n*k_size*k_size + i*k_size + j;
                    int i_idx = n*a_size*a_size + y*a_size + i*a_size + x + j;

                    //printf("k: %f a: %f | %d %d | %d %d\n", k[k_idx], a[i_idx], k_idx, i_idx, x, y);
                    sum += k[k_idx] * a[i_idx];

                }
            }
        }

        //printf("x: %d y: %d | o_x: %d o_y: %d | out: %d| n: %f\n", x, y, id_x, id_y, y * out_size + x, sum);
        out[y * out_size + x] = (sum+out[y * out_size + x] > 0) ? 1 : 0.0;
    }
}

__global__
void full_conv_v1(float* k, float* a, float* out, int k_size, int a_size,  int out_size, int n_elms) {

    const int id_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int id_y = blockIdx.y * blockDim.y + threadIdx.y;

    const int x = id_x % out_size;
    const int y = (32*id_y + id_x) / out_size;

    if (x < out_size && y < out_size) {

        float sum = 0.0f;

        for (int n = 0; n < n_elms; ++n) {
            for (int i = 0; i < k_size; ++i) {
                for (int j = 0; j < k_size; ++j) {

                    //reverse convolution (cross corellation)
                    int k_idx = n*k_size*k_size + k_size*k_size - (i*k_size + j);
                    int i_idx = 
                        (n*a_size*a_size + y*a_size + i*a_size + x + j) 
                        - (k_size - 1 - j) //horizontal offset
                        - (a_size * (k_size - 1 - j));

                    //printf("k: %f a: %f | %d %d | %d %d\n", k[k_idx], a[i_idx], k_idx, i_idx, x, y);
                    
                    if (i_idx >= n*a_size*a_size && i_idx < (n+1)*a_size*a_size) {
                        sum += k[k_idx] * a[i_idx];
                    }

                }
            }
        }

        //printf("x: %d y: %d | o_x: %d o_y: %d | out: %d| n: %f\n", x, y, id_x, id_y, y * out_size + x, sum);
        out[y * out_size + x] = sum + out[y * out_size + x];
    }
}
