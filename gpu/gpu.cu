#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "gpu.h"

#include <iostream>
#include <algorithm>
#include <random>

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

	std::cout << "MPS: " << this->mps << std::endl;
	std::cout << "threads: " << this->threads << std::endl;
	std::cout << "maxThreadsPerBlock: " << this->max_threads << std::endl;
	std::cout << "maxThreadsPerMP: " << this->threads_per_mp << std::endl;
	std::cout << "maxBlocksPerMP: " << this->blocks_per_mp << std::endl;
	std::cout << "eccEnabled: " << this->ecc << std::endl;

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
		std::cerr << "Error while copying data from host to device!" << std::endl;
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

    return cudaSuccess;
}



int GPU::Device::random_numbers(float* p, const size_t n_elems) noexcept {

    if (!validate_pointer(p, n_elems)) {
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

    std::uniform_real_distribution<float> dist(0.0, 1.0);

    //fill in values hopefully not repeating D:
    for (int i = 0; i < n_elems; ++i) {
        rnd_mem[i] = dist(gen);
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


void GPU::Device::matmul_ver1_gpu(float* a, float* b, float* c,
	size_t a_col, size_t a_row, size_t b_col, size_t b_row, size_t c_col, 
    size_t c_row, ActivationFunction actv_fn, const cudaStream_t stream) const noexcept {

	if (!GPU::Device::validate_matmul(a_col, b_row)) {
		std::cerr << "Error: matmul invalid matrix dimensions!" << std::endl;
		return;
	}

	dim3 grid_dimensions(ceilf(c_row / 32.0), ceilf(c_col / 32.0), 1);
	dim3 block_dimensions(32, 32, 1);

	std::cout << "Grid dims: " << grid_dimensions.x << " " << grid_dimensions.y << std::endl;

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
    }
}

__global__
void matadd_v1(float* A, float* B, float* C,
	size_t a_col, size_t a_row, size_t b_col, size_t b_row, size_t c_col, size_t c_row);

void GPU::Device::matadd_ver1(float* a, float* b, float* c, size_t a_col, size_t a_row, 
	size_t b_col, size_t b_row, size_t c_col, size_t c_row, const cudaStream_t stream) const noexcept {

	if (!GPU::Device::validate_matadd(a_col, a_row, b_col, b_row) 
		|| !GPU::Device::validate_matadd(a_col, a_row, c_col, c_row)) {

		std::cerr << "Error: matadd invalid matrix dimensions!" << std::endl;
		return;
	}

	dim3 grid_dimensions(ceilf(c_row / 32.0), ceilf(c_col / 32.0), 1);
	dim3 block_dimensions(32, 32, 1);

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

void GPU::Device::batched_max_pool_ver1(const Tensor input, 
        Tensor out, const size_t pool_size, const cudaStream_t stream) const noexcept {

    dim3 grid_dimensions(ceilf(out.dat_x / 32.0), ceilf((out.dat_z * out.dat_y) / 32.0), 1);
    dim3 block_dimensions(32, 32, 1);

}
