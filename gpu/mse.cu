#ifndef WARP_SIZE
    #define WARP_SIZE 32
#endif

__global__
void mse_v1(float* out, float* a, float* b, int n_items) {

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    const int nth_elm = y * WARP_SIZE + x;

    if (nth_elm < n_items) {
        out[nth_elm] = powf(a[nth_elm] - b[nth_elm], 2) / ((float)n_items);
    }

}

__global__
void mse_der_v1(float* out, float* a, float* b, int n_items) {

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    const int nth_elm = y * WARP_SIZE + x;

    if (nth_elm < n_items) {
        out[nth_elm] = powf(a[nth_elm] - b[nth_elm], 2) * 0.5;
    }

}
