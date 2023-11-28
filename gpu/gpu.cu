#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "gpu.h"

#include <iostream>
#include <algorithm>

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

		i =+ 1;
	}

	return -1;
}

inline bool GPU::Device::validate_matmul(size_t a_col, size_t b_row) noexcept {
	return a_col == b_row;
}

inline bool GPU::Device::validate_matadd(size_t a_col, size_t a_row, size_t b_col, size_t b_row) noexcept {
	return a_col == b_col && a_row == b_row;
}

std::pair<size_t, size_t> GPU::Device::calculate_new_mat_dims(
	size_t a_col, size_t a_row, size_t b_col, size_t b_row) noexcept {

	return std::pair<size_t, size_t> { b_col, a_row };
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

void GPU::Device::matmul_ver1_gpu(float* a, float* b, float* c,
	size_t a_col, size_t a_row, size_t b_col, size_t b_row, size_t c_col, size_t c_row) const noexcept {

	if (!GPU::Device::validate_matmul(a_col, b_row)) {
		std::cerr << "Error: matmul invalid matrix dimensions!" << std::endl;
		return;
	}

	dim3 grid_dimensions(ceilf(c_row / 32.0), ceilf(c_col / 32.0), 1);
	dim3 block_dimensions(32, 32, 1);

	std::cout << "Grid dims: " << grid_dimensions.x << " " << grid_dimensions.y << std::endl;

	matmul_v1<<<grid_dimensions, block_dimensions >>>(a, b, c, a_col, a_row, b_col, b_row, c_col, c_row);

	cudaDeviceSynchronize();
}

__global__
void matadd_v1(float* A, float* B, float* C,
	size_t a_col, size_t a_row, size_t b_col, size_t b_row, size_t c_col, size_t c_row);

void GPU::Device::matadd_ver1(float* a, float* b, float* c, size_t a_col, size_t a_row, 
	size_t b_col, size_t b_row, size_t c_col, size_t c_row) const noexcept {

	if (!GPU::Device::validate_matadd(a_col, a_row, b_col, b_row) 
		|| !GPU::Device::validate_matadd(a_col, a_row, c_col, c_row)) {

		std::cerr << "Error: matadd invalid matrix dimensions!" << std::endl;
		return;
	}

	dim3 grid_dimensions(ceilf(c_row / 32.0), ceilf(c_col / 32.0), 1);
	dim3 block_dimensions(32, 32, 1);

	matadd_v1 << <grid_dimensions, block_dimensions >> > (a, b, c, a_col, a_row, b_col, b_row, c_col, c_row);

	cudaDeviceSynchronize();
}
