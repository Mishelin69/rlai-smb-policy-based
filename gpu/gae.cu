__global__
void gae_delta_v1(float* out, float* r, float* v, float gamma, int items, int x_dim) {

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

    const int nth_elm = y * x_dim + x;
    const float r1 = r[nth_elm];
    const float vp1 = (nth_elm == items) ? 0 : v[nth_elm+1];
    const float v1 = v[nth_elm];

    out[nth_elm] = r1 + gamma*vp1 - v1;
}
