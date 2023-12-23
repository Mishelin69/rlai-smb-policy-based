__device__
float der_sigmoid(const float x) {
    return x * (1.0 - x);
}

__device__
float der_relu(const float x) {
    return (x > 0) ? 1.0 : 0.0;
}

__global__
void matmul_elementwise_DerSigmoid(float* a, float *b, float* c, const size_t x_max, const size_t y_max) {

    const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < x_max && y < y_max) {

        const float tmp = a[y * x_max + x] * der_sigmoid(b[y * x_max + x]);
        c[y * x_max + x] = tmp;
    }
}

__global__
void matmul_elementwise_DerReLU(float* a, float *b, float* c, const size_t x_max, const size_t y_max) {

    const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < x_max && y < y_max) {

        const float tmp = a[y * x_max + x] * der_relu(b[y * x_max + x]);
        c[y * x_max + x] = tmp;
    }
}
