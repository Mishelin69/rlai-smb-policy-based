#ifndef WARP_SIZE
    #define WARP_SIZE 32
#endif

__global__
void matsub_v1(float* out, float* a, float* b, int n_elms) {


    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    const int nth_elm = y * WARP_SIZE + x;

    if (nth_elm < n_elms) {
        out[nth_elm] = a[nth_elm] - b[nth_elm];
    }
}
