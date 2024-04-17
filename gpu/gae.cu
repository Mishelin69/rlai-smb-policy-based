#include <cmath>

__global__
void gae_delta_v1(float* out, float* r, float* v, float gamma, int items, int x_dim) {

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

    const int nth_elm = y * x_dim + x;

    if (nth_elm < items) {
        const float r1 = r[nth_elm];
        const float vp1 = (nth_elm == items) ? 0 : v[nth_elm+1];
        const float v1 = v[nth_elm];

        out[nth_elm] = r1 + gamma*vp1 - v1;
    }
}

__global__
void gae_full_v1(float* out, float* td, float gamma, float lambda, int n_elms) {

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    const int nth_elm = y * 32 + x; //I've put flat 32 because thats the warp size

    if (nth_elm < n_elms) {

        float sum = 0;
        
        for (int i = nth_elm; i < n_elms; ++i) {
            int exponent = i - nth_elm;
            sum += powf(gamma*lambda, exponent) * td[nth_elm];
        }

        out[nth_elm] = sum;
    }
}
