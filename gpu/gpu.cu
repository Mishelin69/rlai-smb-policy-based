#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "gpu.h"

#include <iostream>
#include <algorithm>
#include <random>

int GPU::print_mem(float* p, size_t elms) {

    float* mem = new (std::nothrow) float[elms];

    if (!mem) {
        std::cerr << "GPU::Device::print_mem | Couldn't allocate cpu memory to print!" << std::endl;

        return -1;
    }

    auto res = cudaMemcpy(mem, p, sizeof(float) * elms, cudaMemcpyDeviceToHost);

    if (res != cudaSuccess) {
        std::cerr << "GPU::Device::print_mem | Couldn't copy mem from GPU over to CPU!" << std::endl;

        delete[] mem;
        return -1;
    }

    for (size_t i = 0; i < elms; ++i) {
        std::cout << "Element[" << i << "] = " << mem[i] << std::endl;
    }

    delete[] mem;
    return 1;
}

float* GPU::align_mem(float* ptr, const uint32_t bytes, const uint32_t align) {
    //make sure the block size is divisible by align and if not then make it mathch that size
    size_t mem_to_alloc = bytes;
    size_t extra = 0;

    if (mem_to_alloc % align != 0) {
        const size_t upper = (size_t) std::ceil(mem_to_alloc / align);
        mem_to_alloc = align * upper;
        extra = mem_to_alloc - bytes;
    }

    return ptr + mem_to_alloc;
}


size_t GPU::mem_needed_align(const uint32_t bytes, const uint32_t align) {

    const uint32_t upper = (uint32_t) std::ceil(bytes / align);
    const uint32_t mem_to_alloc = align * upper;

    return mem_to_alloc;
}

GPU::Device::Device(uint64_t gpu_id, int threads): 
    gpu_id(gpu_id), threads(threads) {

        if (cudaDeviceGetAttribute(&this->mps, 
                    cudaDevAttrMultiProcessorCount, 
                    this->gpu_id) != cudaSuccess) {

            std::cerr << "Error while getting maxMPs attribute!!" << std::endl;

            exit(-1);

        }

        if (cudaDeviceGetAttribute(&this->max_threads, 
                    cudaDevAttrMaxThreadsPerBlock, 
                    this->gpu_id) != cudaSuccess) {

            std::cerr << "Error while getting maxThreadsPerBlock attribute!!" << std::endl;

            exit(-1);

        }

        if (cudaDeviceGetAttribute(&this->threads_per_mp, 
                    cudaDevAttrMaxThreadsPerMultiProcessor, 
                    this->gpu_id) != cudaSuccess) {

            std::cerr << "Error while getting maxThreadsPerMP attribute!!" << std::endl;

            exit(-1);

        }

        if (cudaDeviceGetAttribute(&this->blocks_per_mp, 
                    cudaDevAttrMaxBlocksPerMultiprocessor, 
                    this->gpu_id) != cudaSuccess) {

            std::cerr << "Error while getting maxBlocksPerMP attribute!!" << std::endl;

            exit(-1);

        }

        if (cudaDeviceGetAttribute(&this->ecc, 
                    cudaDevAttrEccEnabled, 
                    this->gpu_id) != cudaSuccess) {

            std::cerr << "Error while getting eccEnabled attribute!!" << std::endl;

            exit(-1);

        }

        std::cout << "GPU CONSTRUCTOR ?? " << std::endl;
        /*
           std::cout << "MPS: " << this->mps << std::endl;
           std::cout << "threads: " << this->threads << std::endl;
           std::cout << "maxThreadsPerBlock: " << this->max_threads << std::endl;
           std::cout << "maxThreadsPerMP: " << this->threads_per_mp << std::endl;
           std::cout << "maxBlocksPerMP: " << this->blocks_per_mp << std::endl;
           std::cout << "eccEnabled: " << this->ecc << std::endl;
         */

    }

GPU::Device::~Device() {

    if (this->entries.size() < 1) {
        return;
    }

    for (auto& e : this->entries) {
        if (cudaFree(e.p) != cudaSuccess) {
            std::cerr << "Error while cleaning up memory!\nInvalid pointer(size: " << e.p_size << "): " << e.p << std::endl;
        }
    }

}

void GPU::Device::device_sync() {
    cudaDeviceSynchronize();
}

int GPU::Device::print_mem(float* p, size_t elms) {

    if (!validate_pointer(p, elms)) {
        std::cerr << "GPU::Device::print_mem | Either invalid pointer/size!" << std::endl;

        return -1;
    }

    float* mem = new (std::nothrow) float[elms];

    if (!mem) {
        std::cerr << "GPU::Device::print_mem | Couldn't allocate cpu memory to print!" << std::endl;

        return -1;
    }

    auto res = memcpy_device(p, mem, sizeof(float) * elms);

    if (res != cudaSuccess) {
        std::cerr << "GPU::Device::print_mem | Couldn't copy mem from GPU over to CPU!" << std::endl;

        delete[] mem;
        return -1;
    }

    for (size_t i = 0; i < elms; ++i) {
        std::cout << "Element[" << i << "] = " << mem[i] << std::endl;
    }

    delete[] mem;
    return 1;
}

int GPU::Device::get_mps() const noexcept {
    return this->mps;
}

int GPU::Device::get_threads() const noexcept {
    return this->threads;
}

int GPU::Device::get_max_threads() const noexcept {
    return this->max_threads;
}

int GPU::Device::get_threads_per_mp() const noexcept {
    return this->threads_per_mp;
}

int GPU::Device::get_blocks_per_mp() const noexcept {
    return this->blocks_per_mp;
}

int GPU::Device::get_ecc() const noexcept {
    return this->ecc;
}

int GPU::Device::update_threads(int _threads) noexcept {

    if (_threads < 0 || _threads > this->max_threads) {
        return 0;
    }

    this->threads = _threads;

    return 1;
}

float* GPU::Device::allocate_memory(size_t size) noexcept {

    float* cuda_p = NULL;

    if (cudaMalloc(&cuda_p, size) != cudaSuccess) {
        std::cerr << "Error while allocating memory on the GPU!" << std::endl;

        return nullptr;
    }

    this->entries.push_back(DeviceMemoryEntry{ cuda_p, size });

    return cuda_p;
}

int GPU::Device::validate_pointer(float* p, size_t p_size) const noexcept {

    for (auto& e : this->entries) {

        if (e.p == p && p_size == e.p_size) {
            return 1;
        }

        else if (e.p < p && (e.p + e.p_size > p)) {

            if ((e.p + e.p_size - p) >= p_size) {
                return 1;
            }

        }

    }

    return -1;
}

int GPU::Device::memcpy_host(float* src, float* dst, size_t size) noexcept {

    if (!this->validate_pointer(dst, size)) {
        return -1;
    }

    cudaError_t res = cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);

    if (res != cudaSuccess) {
        std::cerr << "Error while copying data from host to device!" << std::endl;
        return res;
    }

    return cudaSuccess;
};

int GPU::Device::memcpy_device(float* src, float* dst, size_t size) noexcept {

    if (!this->validate_pointer(src, size)) {
        return -1;
    }

    cudaError_t res = cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);

    if (res != cudaSuccess) {
        /*
           std::cerr << "Accessing: " << src << std::endl;
           std::cerr << "Memcpy error: " << cudaGetErrorString(res) << std::endl;
           std::cerr << "Error while copying data from host to device!" << std::endl;
         */
        return res;
    }

    return cudaSuccess;
};

int GPU::Device::memcpy_host_unsafe(float* src, float* dst, size_t size) noexcept {

    cudaError_t res = cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);

    if (res != cudaSuccess) {
        std::cerr << "Error while copying data from host to device!" << std::endl;
        return res;
    }

    return cudaSuccess;
};

int GPU::Device::memcpy_device_unsafe(float* src, float* dst, size_t size) noexcept {

    cudaError_t res = cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);

    if (res != cudaSuccess) {
        std::cerr << "Error while copying data from host to device!" << std::endl;
        return res;
    }

    return cudaSuccess;
};

int GPU::Device::free_memory(float* p) noexcept {

    size_t i = 0;

    for (auto & e : this->entries) {

        if (e.p == p) {

            //swap with the last since we don't care about order
            //and then pop the last element so we don't have to
            //move the whole vector when removing from ith position
            std::iter_swap(this->entries.begin() + i, this->entries.begin() + this->entries.size());

            cudaError_t res = cudaFree(p);

            if (res != cudaSuccess) {

                std::cerr << "Error while freeing memory on the GPU!" << std::endl;
                this->entries.pop_back();

                return res;
            }

            this->entries.pop_back();

            return cudaSuccess;
        }

        i += 1;
    }

    return -1;
}

int GPU::Device::memset(float* p, int value, const size_t n_elems) noexcept {

    if (!validate_pointer(p, n_elems)) {
        return -1;
    }

    cudaError_t res = cudaMemset(p, value, sizeof(float) * n_elems);

    if (res != cudaSuccess) {
        return res;
    }

    device_sync();
    return cudaSuccess;
}



int GPU::Device::random_numbers(float* p, const size_t n_elems) noexcept {

    if (!validate_pointer(p, n_elems)) {
        std::cerr << "GPU::Device::random_numbers | Pointer not valid!" << std::endl;
        return -1;
    }

    //Explanation: (std::nothrow) makes it work like malloc no throwing
    //I hate when languages throw worst feature, error as value always!!
    float* rnd_mem = new (std::nothrow) float[n_elems];

    if (!rnd_mem) {

        std::cerr << "GPU::Device::random_numbers | Error while allocating memory!" << std::endl;
        return -1;
    }

    //random stuff setup
    std::random_device rd;
    std::mt19937 gen(rd()); 

    std::uniform_real_distribution<float> dist(-1.0, 1.0);

    //fill in values hopefully not repeating D:
    for (int i = 0; i < n_elems; ++i) {
        rnd_mem[i] = dist(gen);
        //std::cout << "Rand number: " << rnd_mem[i] << std::endl;
    }

    //try to memcpy from cpu to gpu 
    int res = memcpy_host(rnd_mem, p, sizeof(float) * n_elems);

    if (res != 0) {
        std::cerr << "GPU::Device::random_numbers | Error while calling memcpy!" << std::endl;
        return res;
    }

    //don't forget to free the memory
    //def didn't just happen 
    delete[] rnd_mem;

    return 0;
}

inline bool GPU::Device::validate_matmul(size_t a_col, size_t b_row) noexcept {
    return a_col == b_row;
}

inline bool GPU::Device::validate_matadd(size_t a_col, size_t a_row, size_t b_col, size_t b_row) noexcept {
    return a_col == b_col && a_row == b_row;
}

bool GPU::Device::validate_convolution(const int kernel_dim, const int dat_dim, const int out_dim) noexcept {

    size_t real_out_dim = GPU::Device::calculate_conv_dims(kernel_dim, dat_dim);

    return real_out_dim == out_dim;
}

std::pair<size_t, size_t> GPU::Device::calculate_new_mat_dims(
        size_t a_col, size_t a_row, size_t b_col, size_t b_row) noexcept {

    return std::pair<size_t, size_t> { b_col, a_row };
}

size_t GPU::Device::calculate_conv_dims(const int kx, const int mx) noexcept {
    return mx + 1 - kx + 0; 
}


void GPU::Device::matmul_ver1_cpu(float* a, float* b, float* c, 
        size_t a_col, size_t a_row, size_t b_col, size_t b_row, size_t c_col, size_t c_row) const noexcept {

    if (!GPU::Device::validate_matmul(a_col, b_row)) {
        std::cerr << "Error: matmul invalid matrix dimensions!" << std::endl;
        return;
    }

    for (size_t i = 0; i < b_col; ++i) {
        for (size_t j = 0; j < a_row; ++j) {

            float tmp = 0;

            for (size_t k = 0; k < b_row; ++k) {
                tmp += a[j*a_col + k] * b[k*b_col + i];
            }

            c[j*c_col + i] = tmp;
        }
    }

}

__global__
void matmul_v1(float* A, float* B, float* C, 
        size_t a_col, size_t a_row, size_t b_col, size_t b_row, size_t c_col, size_t c_row);
__global__
void matmul_v1_ReLU(float* A, float* B, float* C, 
        size_t a_col, size_t a_row, size_t b_col, size_t b_row, size_t c_col, size_t c_row);
__global__
void matmul_v1_Sigmoid(float* A, float* B, float* C, 
        size_t a_col, size_t a_row, size_t b_col, size_t b_row, size_t c_col, size_t c_row);

__global__
void matmul_v1_DerReLU(float* A, float* B, float* C, 
        size_t a_col, size_t a_row, size_t b_col, size_t b_row, size_t c_col, size_t c_row);
__global__
void matmul_v1_DerSigmoid(float* A, float* B, float* C, 
        size_t a_col, size_t a_row, size_t b_col, size_t b_row, size_t c_col, size_t c_row);

void GPU::Device::matmul_ver1_gpu(float* a, float* b, float* c,
        size_t a_col, size_t a_row, size_t b_col, size_t b_row, size_t c_col, 
        size_t c_row, ActivationFunction actv_fn, const cudaStream_t stream) const noexcept {

    if (!GPU::Device::validate_matmul(a_col, b_row)) {
        std::cerr << "Error: matmul invalid matrix dimensions!" << std::endl;
        std::cerr 
            << "a_col: " << a_col << "\n" 
            << "a_row: " << a_row << "\n"
            << "b_col: " << b_col << "\n"
            << "b_row: " << b_row << "\n" << std::endl;

        exit(-1);

        return;
    }

    if (!a || !b || !c) {

        if (!a) {
            std::cerr << "A POINTER IS NULL" << std::endl;
        }

        if (!b) {
            std::cerr << "B POINTER IS NULL" << std::endl;
        }

        if (!c) {
            std::cerr << "C POINTER IS NULL" << std::endl;
        }

        std::cerr << "GPU::Device::matmul_ver1_gpu | NULL POINTER!!!!" << std::endl;
        exit(-1);
    }

    int elm_total = c_col * c_row;

    dim3 grid_dimensions(1, 1, 1);
    dim3 block_dimensions(32, ceilf(elm_total/32.0), 1);

    //std::cout << "Grid dims: " << c_row << " " << c_col << " | " << ceilf(c_row / 32.0) << " " <<  ceilf(c_col / 32.0) << std::endl;

    switch (actv_fn) {

        case ReLU:
            matmul_v1_ReLU<<<grid_dimensions, block_dimensions, 0, stream>>>(a, b, c, a_col, a_row, b_col, b_row, c_col, c_row);
            break;
        case Sigmoid:
            matmul_v1_Sigmoid<<<grid_dimensions, block_dimensions, 0, stream>>>(a, b, c, a_col, a_row, b_col, b_row, c_col, c_row);
            break;
        case None:
            matmul_v1<<<grid_dimensions, block_dimensions, 0, stream>>>(a, b, c, a_col, a_row, b_col, b_row, c_col, c_row);
            break;
        case DerReLU:
            matmul_v1_DerReLU<<<grid_dimensions, block_dimensions, 0, stream>>>(a, b, c, a_col, a_row, b_col, b_row, c_col, c_row);
            break;
        case DerSigmoid:
            matmul_v1_DerSigmoid<<<grid_dimensions, block_dimensions, 0, stream>>>(a, b, c, a_col, a_row, b_col, b_row, c_col, c_row);
            break;
    }
}

__global__
void matadd_v1(float* A, float* B, float* C,
        size_t a_col, size_t a_row, size_t b_col, size_t b_row, size_t c_col, size_t c_row);

void GPU::Device::matadd_ver1(float* a, float* b, float* c, 
        size_t a_col, size_t a_row, size_t b_col, size_t b_row, 
        size_t c_col, size_t c_row, const cudaStream_t stream) const noexcept {

    if (!GPU::Device::validate_matadd(a_col, a_row, b_col, b_row) 
            || !GPU::Device::validate_matadd(a_col, a_row, c_col, c_row)) {

        std::cerr << "Error: matadd invalid matrix dimensions!" << std::endl;
        std::cerr 
            << "a_col: " << a_col << "\n" 
            << "a_row: " << a_row << "\n"
            << "b_col: " << b_col << "\n"
            << "b_row: " << b_row << "\n" 
            << "c_col: " << c_col << "\n"
            << "c_row: " << c_row << "\n" << std::endl;

        return;
    }

    dim3 grid_dimensions(1, 1, 1);
    dim3 block_dimensions(32, ceilf((c_row * c_col) / 32.0), 1);

    matadd_v1<<<grid_dimensions, block_dimensions, 0, stream>>>(a, b, c, a_col, a_row, b_col, b_row, c_col, c_row);
}

__global__
void convolve_v1(const float* k, const float* m, float* o, int kx, int mx, int ox);
__global__
void convolve_v1_ReLU(const float* k, const float* m, float* o, int kx, int mx, int ox);
__global__
void convolve_v1_Sigmoid(const float* k, const float* m, float* o, int kx, int mx, int ox);

void GPU::Device::conv_ver1(const float* kernel, const float* dat, float* output, 
        const size_t kernel_dim, const size_t dat_dim, 
        const size_t out_dim, ActivationFunction actv_fn) const noexcept {

    if (!GPU::Device::validate_convolution(kernel_dim, dat_dim, out_dim)) {

        std::cerr << "Error: invalid function parameters! hint: out_dim" << std::endl;
        return;
    }

    dim3 grid_dimensions(ceilf(out_dim / 32.0), ceilf(out_dim / 32.0), 1);
    dim3 block_dimensions(32, 32, 1);

    switch (actv_fn) {

        case ReLU:
            convolve_v1_ReLU<<<grid_dimensions, block_dimensions>>>(kernel, dat, output, kernel_dim, dat_dim, out_dim);
            break;
        case Sigmoid:
            convolve_v1_Sigmoid<<<grid_dimensions, block_dimensions>>>(kernel, dat, output, kernel_dim, dat_dim, out_dim);
            break;
        case None:
            convolve_v1<<<grid_dimensions, block_dimensions>>>(kernel, dat, output, kernel_dim, dat_dim, out_dim);
            break;
    }
}

__global__
void batched_convolve_v2_ReLU(const float* k, const float* m, float* o, 
        int kx, int mx, int ox, const size_t b_size, const size_t n_elms, const size_t inputs);
__global__
void batched_convolve_v2_Sigmoid(const float* k, const float* m, float* o, 
        int kx, int mx, int ox, const size_t b_size, const size_t n_elms, const size_t inputs);

void GPU::Device::batched_conv_ver1(const float* kernel, const float* dat, float* output, 
        const size_t kernel_dim, const size_t dat_dim, const size_t out_dim, ActivationFunction actv_fn, 
        const size_t n_elms, const size_t batch_size, const size_t inputs, const cudaStream_t stream) const noexcept {

    if (!GPU::Device::validate_convolution(kernel_dim, dat_dim, out_dim)) {

        std::cerr << "Error: invalid function parameters! hint: out_dim" << std::endl;
        return;
    }

    dim3 grid_dimensions(ceilf(out_dim / 32.0), ceilf((n_elms * out_dim) / 32.0), 1);
    dim3 block_dimensions(32, 32, 1);

    const size_t cache_size = sizeof(float) * kernel_dim*kernel_dim * n_elms;
    std::cout << "Batched elm: " << cache_size / sizeof(float) << " byte size: " << cache_size << std::endl;

    switch (actv_fn) {

        case ReLU:
            batched_convolve_v2_ReLU<<<grid_dimensions, block_dimensions, cache_size, stream>>>(
                    kernel, dat, output, kernel_dim, dat_dim, out_dim, batch_size, n_elms, inputs
                    );
            break;
        case Sigmoid:
            batched_convolve_v2_Sigmoid<<<grid_dimensions, block_dimensions, cache_size, stream>>>(
                    kernel, dat, output, kernel_dim, dat_dim, out_dim, batch_size, n_elms, inputs
                    );
            break;
            //compiler ignores this idk why this there (no c++20 flag but I'm too lazy to put it there
            //I mean, would it really matter that much if it gets ignored? I don't think so :| )
            [[unlikely]] case None:
                std::cerr << "GPU::Device::batched_conv_ver1(...) | Option None not supported!!" << std::endl;  
            break;
    }
}

__global__
void conv_ReLU2(float* k, float* a, float* out, int k_size, 
        int a_size,  int out_size, int n_elms);

void GPU::Device::conv_ver2(Tensor a, Tensor b, Tensor out, uint32_t skip, cudaStream_t stream) {

    if (!GPU::Device::validate_convolution(b.dat_x, a.dat_x, out.dat_x)) {

        std::cerr << "CR2: Error: invalid function parameters! hint: out_dim" << std::endl;
        return;
    }

    const size_t elm_total = out.dat_y * out.dat_x;
    const size_t elm_y = ceilf(elm_total / 32.0);

    dim3 grid_dimensions(1, 1, 1);
    dim3 block_dimensions(32, elm_y, 1);

    conv_ReLU2<<<grid_dimensions, block_dimensions, 0, stream>>>(
            b.dat_pointer, a.dat_pointer, out.dat_pointer, 
            b.dat_x, a.dat_x, out.dat_x, b.dat_z 
            );
}

__global__
void conv_pre(float* k, float* a, float* out, int k_size, 
        int a_size,  int out_size, int n_elms);

void GPU::Device::conv_ver2_preactivations(Tensor out, Tensor inp, Tensor weights, 
        Tensor bias, cudaStream_t stream) {

    if (!GPU::Device::validate_convolution(weights.dat_x, inp.dat_x, out.dat_x)) {

        std::cerr << "CV2PA: Error: invalid function parameters! hint: out_dim" << std::endl;
        return;
    }

    const size_t elm_total = out.dat_y * out.dat_x;
    const size_t elm_y = ceilf(elm_total / 32.0);

    dim3 grid_dimensions(1, 1, 1);
    dim3 block_dimensions(32, elm_y, 1);

    //=======//
    //  ++i  //
    // i+=1  //
    //=======//
    //  i++  //
    //   i   //
    // i+=1  //
    //=======//

    for (int i = 0; i < out.dat_z; ++i) {
        conv_add(
            GPU::Tensor { 
            bias.dat_pointer, 
            out.dat_x, //actually if this wasnt uint32_t even -1 could be here and it wouldnt matter lol
            1,//1 because same bias for everything 
              //also bias is just 1D vector where its shape is (n,) 
              //where n => number of output feature maps
            1,
            },

            GPU::Tensor {
            out.dat_pointer + out.dat_x*out.dat_y*i,
            out.dat_x,
            out.dat_y,
            1
            }, 
            i, 
            stream
        );
    }


    for (int i = 0; i < out.dat_z; ++i) {
        conv_pre<<<grid_dimensions, block_dimensions, 0, stream>>>(
            weights.dat_pointer+i*weights.dat_x*weights.dat_y, 
            inp.dat_pointer, 
            out.dat_pointer, 
            weights.dat_x, 
            inp.dat_x, 
            out.dat_x, 
            weights.dat_z 
        );
    }
}

void GPU::Device::conv_ver2_all(Tensor out, Tensor inp, 
        Tensor weights, cudaStream_t stream) {

    if (!GPU::Device::validate_convolution(weights.dat_x, inp.dat_x, out.dat_x)) {

        std::cerr << "CV2A: Error: invalid function parameters! hint: out_dim" << std::endl;
        std::cout << "w: " << weights.dat_x << std::endl;
        std::cout << "i: " << inp.dat_x << std::endl;
        std::cout << "o: " << out.dat_x << std::endl;
        return;
    }

    const size_t elm_total = out.dat_y * out.dat_x;
    const size_t elm_y = ceilf(elm_total / 32.0);

    dim3 grid_dimensions(1, 1, 1);
    dim3 block_dimensions(32, elm_y, 1);

    //=======//
    //  ++i  //
    // i+=1  //
    //=======//
    //  i++  //
    //   i   //
    // i+=1  //
    //=======//

    for (int i = 0; i < out.dat_z; ++i) {
        conv_ReLU2<<<grid_dimensions, block_dimensions, 0, stream>>>(
            weights.dat_pointer+i*weights.dat_x, 
            inp.dat_pointer, 
            out.dat_pointer, 
            weights.dat_x, 
            inp.dat_x, 
            out.dat_x, 
            weights.dat_z 
        );
    }
}

__global__
void full_conv_v1(float* k, float* a, float* out, int k_size, 
        int a_size,  int out_size, int n_elms);

void GPU::Device::full_convolution(Tensor out, Tensor inp, Tensor weights, cudaStream_t stream) {

    //Safe uncomment I hope D:
    /*
    if (!GPU::Device::validate_convolution(weights.dat_x, inp.dat_x, out.dat_x)) {

        std::cerr << "FC1: Error: invalid function parameters! hint: out_dim" << std::endl;
        return;
    }*/

    const size_t elm_total = out.dat_y * out.dat_x;
    const size_t elm_y = ceilf(elm_total / 32.0);

    dim3 grid_dimensions(1, 1, 1);
    dim3 block_dimensions(32, elm_y, 1);


    //weights are actually kernels !
    for (int i = 0; i < out.dat_z; ++i) {
        conv_pre<<<grid_dimensions, block_dimensions, 0, stream>>>(
            weights.dat_pointer+i*weights.dat_x*weights.dat_y, 
            inp.dat_pointer, 
            out.dat_pointer, 
            weights.dat_x, 
            inp.dat_x, 
            out.dat_x, 
            weights.dat_z 
        );
    }

}

__global__
void conv_add_gpu(float* A, float* B, int a_dim, int b_dim, int bias_index);

void GPU::Device::conv_add(Tensor a, Tensor b, int bias_index, cudaStream_t stream) {

    const size_t elm_total = b.dat_y * b.dat_x;
    const size_t elm_y = ceilf(elm_total / 32.0);

    dim3 grid_dimensions(1, 1, 1);
    dim3 block_dimensions(32, elm_y, 1);

    conv_add_gpu<<<grid_dimensions, block_dimensions, 0, stream>>>(
            a.dat_pointer, b.dat_pointer, a.dat_x, b.dat_x, bias_index);
}

__global__
void max_pooling_ver1(const float* a, float* out, size_t* out_idx, const size_t pool_size,
        const size_t input_dim, const size_t output_dim, const size_t z_dim);

void GPU::Device::batched_max_pool_ver1(const Tensor input, Tensor out, size_t* idx_ptr,
        const size_t pool_size, const cudaStream_t stream) const noexcept {

    dim3 grid_dimensions(ceilf(out.dat_x / 32.0), ceilf((out.dat_z * out.dat_y) / 32.0), 1);
    dim3 block_dimensions(32, 32, 1);

    max_pooling_ver1<<<grid_dimensions, block_dimensions, 0, stream>>>(
            input.dat_pointer, 
            out.dat_pointer, 
            idx_ptr, 
            pool_size, 
            input.dat_x, 
            out.dat_x, 
            input.dat_z
            );

}

__global__
void pool_ver2(const float* input, float* out, int* idx, int in_dim, int out_dim, int pool_size);

void GPU::Device::max_pool_ver2(const GPU::Tensor input, GPU::Tensor out, int* idx, const int pool_size, cudaStream_t stream) {

    const size_t elm_total = out.dat_y * out.dat_x;
    const size_t elm_y = ceilf(elm_total / 32.0);

    dim3 grid_dimensions(1, 1, 1);
    dim3 block_dimensions(32, elm_y, 1);

    pool_ver2<<<grid_dimensions, block_dimensions, 0, stream>>>(
            input.dat_pointer, out.dat_pointer, idx, 
            input.dat_x, out.dat_x, pool_size
            );

}

__global__
void unpooling_v1(float* out, int* indices, float* loss, int output_dim, int in_dim, int pool_size); 

void GPU::Device::max_pool_der(Tensor out, Tensor loss, int* indices, cudaStream_t stream) {

    const size_t elm_total = out.dat_y * out.dat_x;
    const size_t elm_y = ceilf(elm_total / 32.0);

    dim3 grid_dimensions(1, 1, 1);
    dim3 block_dimensions(32, elm_y, 1);

    for (size_t i = 0; i < out.dat_z; ++i) {
        unpooling_v1<<<grid_dimensions, block_dimensions, 0, stream>>>(
            out.dat_pointer+i*elm_total, indices+i*(loss.dat_x*loss.dat_y), 
            loss.dat_pointer+i*(loss.dat_x*loss.dat_y), out.dat_x, loss.dat_x, 2
            );
    }
}

__global__
void matmul_elementwise_DerReLU(float* a, float *b, float* c, const size_t x_max, const size_t y_max);

__global__
void matmul_elementwise_DerSigmoid(float* a, float *b, float* c, const size_t x_max, const size_t y_max);

__global__
void matmul_elementwise_v1(float* a, float *b, float* c, const size_t x_max, const size_t y_max); 

void GPU::Device::matmul_elementwise(Tensor a, Tensor b, Tensor out, 
        const cudaStream_t stream, const ActivationFunction actv_fn) const noexcept {

    /*
    if (out.dat_x % 32 != 0) {

        std::cerr << "GPU::Device::matmul_elementwise | Dims not a multiple of 32!!!" << std::endl;
        return;
    }*/

    size_t elm1 = a.dat_y*a.dat_x*a.dat_z;
    size_t elm2 = b.dat_y*b.dat_x*b.dat_z;
    size_t elm3 = out.dat_y*out.dat_x*out.dat_z;

    if (elm1 != elm2 || elm2 != elm1) {
        std::cerr << "GPU::Device::matmul_elementwise | Sizes do not match!" << std::endl;
        std::cerr << "elm1: " << elm1 << " elm2: " << elm2 << " elm3: " << elm3 << std::endl;
    }

    dim3 grid_dimensions(ceilf(out.dat_y / 32.0), ceilf(out.dat_x / 32.0), 1);
    dim3 block_dimensions(32, 32, 1);

    switch (actv_fn) {

        case DerReLU:
            matmul_elementwise_DerReLU<<<grid_dimensions, block_dimensions, 0, stream>>>(a.dat_pointer, b.dat_pointer, out.dat_pointer, out.dat_x, out.dat_y);
            break;
        case DerSigmoid:
            matmul_elementwise_DerSigmoid<<<grid_dimensions, block_dimensions, 0, stream>>>(a.dat_pointer, b.dat_pointer, out.dat_pointer, out.dat_x, out.dat_y);
            break;

        case None:
            matmul_elementwise_v1<<<grid_dimensions, block_dimensions, 0, stream>>>(a.dat_pointer, b.dat_pointer, out.dat_pointer, out.dat_x, out.dat_y);
            break;
        case ReLU:
        case Sigmoid:
        default:
            std::cerr << "GPU::Device::matmul_elementwise | Invalid enum option" << std::endl;
            return;    
    }

}

__global__
void gae_delta_v1(float* out, float* r, float* v, float gamma, int items, int x_dim);

void GPU::Device::gae_delta(GPU::Tensor out, const GPU::Tensor rewards, 
        const GPU::Tensor values, const float gamma, const cudaStream_t stream) {

    if (!validate_pointer(rewards.dat_pointer, rewards.dat_x)) {
        std::cerr << "GPU::Device::gae_delta | Couldn't validate pointer (rewards tested)!" << std::endl;
    }

    const size_t elm_total = rewards.dat_y * rewards.dat_x;
    const size_t elm_y = ceilf(elm_total / 32.0);

    dim3 grid_dimensions(1, 1, 1);
    dim3 block_dimensions(32, elm_y, 1);

    gae_delta_v1<<<grid_dimensions, block_dimensions, 0, stream>>>(out.dat_pointer, rewards.dat_pointer, values.dat_pointer, gamma, rewards.dat_x, 32);
}

__global__
void gae_full_v1(float* out, float* td, float gamma, float lambda, int n_elms);

void GPU::Device::gae_full(GPU::Tensor out, GPU::Tensor td_delta, float gamma, float lambda, cudaStream_t stream) {

    if (!validate_pointer(out.dat_pointer, out.dat_x)) {
        std::cerr << "GPU::Device::gae_full | Couldn't validate pointer (output tested)!" << std::endl;
    }

    const size_t elm_total = td_delta.dat_y * td_delta.dat_x;
    const size_t elm_y = ceilf(elm_total / 32.0);

    dim3 grid_dimensions(1, 1, 1);
    dim3 block_dimensions(32, elm_y, 1);


    gae_full_v1<<<grid_dimensions, block_dimensions, 0, stream>>>(out.dat_pointer, td_delta.dat_pointer, gamma, lambda, td_delta.dat_x);
}

__global__
void ppo_v1(float* out, float* prob, float* prob_old, float* adv, float* idx_cur, 
        float* idx_prev, float epsilon, int n_elms);


void GPU::Device::ppo(Tensor out, Tensor prob_cur, Tensor prob_prev, Tensor adv, Tensor idx_cur, Tensor idx_prev, float epsilon, cudaStream_t stream) {

    if (!validate_pointer(out.dat_pointer, out.dat_x)) {
        std::cerr << "GPU::Device::ppo | Couldn't validate pointer (output tested)!" << std::endl;
    }

    const size_t elm_total = prob_cur.dat_y * prob_cur.dat_x;
    const size_t elm_y = ceilf(elm_total / 32.0);

    dim3 grid_dimensions(1, 1, 1);
    dim3 block_dimensions(32, elm_y, 1);


    ppo_v1<<<grid_dimensions, block_dimensions, 0, stream>>>(
            out.dat_pointer,
            prob_cur.dat_pointer, 
            prob_prev.dat_pointer,
            adv.dat_pointer, 
            idx_cur.dat_pointer,
            idx_prev.dat_pointer, 
            epsilon, elm_total);

}

__global__
void mse_v1(float* out, float* a, float* b, int n_items);

void GPU::Device::mse(Tensor out, Tensor a, Tensor b, cudaStream_t stream) {

    if (!validate_pointer(out.dat_pointer, out.dat_x)) {
        std::cerr << "GPU::Device::mse | Couldn't validate pointer (output tested)!" << std::endl;
    }

    const size_t elm_total = a.dat_y * a.dat_x;
    const size_t elm_y = ceilf(elm_total / 32.0);

    dim3 grid_dimensions(1, 1, 1);
    dim3 block_dimensions(32, elm_y, 1);


    mse_v1<<<grid_dimensions, block_dimensions, 0, stream>>>(
            out.dat_pointer,
            a.dat_pointer,
            b.dat_pointer,
            elm_total);

}

__global__
void mse_der_v1(float* out, float* a, float* b, int n_items);

void GPU::Device::mse_der(Tensor out, Tensor a, Tensor b, cudaStream_t stream) {

    if (!validate_pointer(out.dat_pointer, out.dat_x)) {
        std::cerr << "GPU::Device::mse | Couldn't validate pointer (output tested)!" << std::endl;
    }

    const size_t elm_total = a.dat_y * a.dat_x;
    const size_t elm_y = ceilf(elm_total / 32.0);

    dim3 grid_dimensions(1, 1, 1);
    dim3 block_dimensions(32, elm_y, 1);


    mse_der_v1<<<grid_dimensions, block_dimensions, 0, stream>>>(
            out.dat_pointer,
            a.dat_pointer,
            b.dat_pointer,
            elm_total
            );

}

__global__
void matmul_preactv_v1(float* A, float* B, float* C, 
        size_t a_col, size_t a_row, size_t b_col, size_t b_row, size_t c_col, size_t c_row);

//This might be the first function that makes calls to two diffetent
//kernels :0
void GPU::Device::preactivations_dense_relu(Tensor out, Tensor inp, Tensor weights, 
        Tensor bias, cudaStream_t stream) {

    if (!GPU::Device::validate_matmul(inp.dat_x, weights.dat_y)) {
        std::cerr << "Error: preactivations_dense_relu invalid matrix dimensions!" << std::endl;
        std::cerr 
            << "a_col: " << inp.dat_x << "\n" 
            << "a_row: " << inp.dat_y << "\n"
            << "b_col: " << weights.dat_x << "\n"
            << "b_row: " << weights.dat_y << "\n" << std::endl;

        exit(-1);

        return;
    }


    int elm_total = out.dat_x * out.dat_y;

    dim3 grid_dimensions(1, 1, 1);
    dim3 block_dimensions(32, ceilf(elm_total/32.0), 1);

    matadd_v1<<<grid_dimensions, block_dimensions, 0, stream>>>(
            bias.dat_pointer, 
            out.dat_pointer, 
            out.dat_pointer, 
            bias.dat_x,
            1,
            bias.dat_x,
            1,
            bias.dat_x,
            1
            );

    matmul_preactv_v1<<<grid_dimensions, block_dimensions, 0, stream>>>(
            inp.dat_pointer, weights.dat_pointer, out.dat_pointer, 
            inp.dat_x, inp.dat_y, 
            weights.dat_y, weights.dat_x, out.dat_x, out.dat_y
            );
}

__global__
void matsub_v1(float* out, float* a, float* b, int n_elms);

void GPU::Device::matsub(Tensor out, Tensor a, Tensor b, cudaStream_t stream) {

    if (!validate_pointer(out.dat_pointer, out.dat_x)) {
        std::cerr << "GPU::Device::mse | Couldn't validate pointer (output tested)!" << std::endl;
    }

    const size_t elm_total = out.dat_y * out.dat_x;
    const size_t elm_y = ceilf(elm_total / 32.0);

    dim3 grid_dimensions(1, 1, 1);
    dim3 block_dimensions(32, elm_y, 1);


    matsub_v1<<<grid_dimensions, block_dimensions, 0, stream>>>(
            out.dat_pointer,
            a.dat_pointer,
            b.dat_pointer,
            elm_total
            );

}

__global__
void vector_outer_v1(float* output, float* a, float* b, int dim_len, int right);

void GPU::Device::vector_outer(Tensor out, Tensor a, Tensor b, cudaStream_t stream) {

    if (!validate_pointer(out.dat_pointer, out.dat_x)) {
        std::cerr << "GPU::Device::vector_outer | Couldn't validate pointer (output tested)!" << std::endl;
    }

    const size_t elm_total = out.dat_y * out.dat_x; //one big problem, I dont have 8k threads :(
    const size_t elm_y = ceilf(a.dat_x / 32.0);

    int dim_a = a.dat_x * a.dat_y;
    int dim_b = b.dat_x * b.dat_y;

    dim3 grid_dimensions(1, 1, 1);
    dim3 block_dimensions(32, elm_y, 1);

    vector_outer_v1<<<grid_dimensions, block_dimensions, 0, stream>>>(
        out.dat_pointer,
        a.dat_pointer,
        b.dat_pointer,
        dim_a,
        dim_b
    );
}

__global__
void multiply_by_scalar(float* out, float* a, float scalar, int n_items);

void GPU::Device::vector_scalar(Tensor out, Tensor a, float scalar, cudaStream_t stream) {

    const size_t elm_total = out.dat_y * out.dat_x;
    const size_t elm_y = ceilf(elm_total / 32.0);

    dim3 grid_dimensions(1, 1, 1);
    dim3 block_dimensions(32, elm_y, 1);


    multiply_by_scalar<<<grid_dimensions, block_dimensions, 0, stream>>>(
            out.dat_pointer,
            a.dat_pointer,
            scalar,
            elm_total
            );
}

void GPU::Device::subs_number(Tensor a, float scalar, cudaStream_t stream) {

    float n = 0.0;

    auto status = memcpy_device(a.dat_pointer, &n, sizeof(float));

    if (status != cudaSuccess) {
        std::cerr << "GPU::Device::subs_number | Error copying data GPU -> CPU" << std::endl;
    }

    float res = n - scalar;

    status = memcpy_host(&res, a.dat_pointer, sizeof(float));

    if (status != cudaSuccess) {
        std::cerr << "GPU::Device::subs_number | Error copying data GPU -> CPU" << std::endl;
    }
}

__global__ 
void sum_vector_one(float* out, float* a, int n_items);

__global__ 
void sum_bias_conv(float* out, float* a, int filter_size, int n_items);


void GPU::Device::sum_bias_cnn(Tensor out, Tensor a, int filter_size, cudaStream_t stream) {

    const size_t elm_total = out.dat_y * out.dat_x;
    const size_t elm_y = ceilf(elm_total / 32.0);

    dim3 grid_dimensions(1, 1, 1);
    dim3 block_dimensions(32, elm_y, 1);


    sum_bias_conv<<<grid_dimensions, block_dimensions, 0, stream>>>(
            out.dat_pointer,
            a.dat_pointer,
            filter_size,
            elm_total
            );

}
