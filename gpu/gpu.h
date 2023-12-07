#pragma once

#include <cstdint>
#include <vector>

namespace GPU {

enum ActivationFunction {
    ReLU = 0,
    Sigmoid,
    None
};

struct DeviceMemoryEntry {
	float* p;
	size_t p_size;
};

class Device {

private:

	//GPU info
	uint64_t gpu_id = 0;

	int mps;
	int threads;

	int max_threads;
	int threads_per_mp;
	int blocks_per_mp;
	int ecc;

	std::vector<DeviceMemoryEntry> entries;


private:

	int validate_pointer(float* p, size_t p_size) const noexcept;

	inline static bool validate_matmul(size_t a_col, size_t b_row) noexcept;

	inline static bool validate_matadd(size_t a_col, size_t a_row, size_t b_col, size_t b_row) noexcept;

	static bool validate_convolution(const int kernel_dim, const int dat_dim, const int out_dim) noexcept;

public:

	Device(uint64_t gpu_id = 0, int threads = 512);
	Device(Device& other) = default;
	~Device();

    //calls cuda_device_synchronize, yeah
    void device_sync();

	static std::pair<size_t, size_t> calculate_new_mat_dims(size_t a_col, size_t a_row, size_t b_col, size_t b_row) noexcept;

    //We are assuming m*m matrices for this problem
    //this is fitted for this project so it's fine to do it like this
    //we know for a fact that it'll only be m*m matrices
	static size_t calculate_conv_dims(const int kx, const int mx) noexcept;

	/*
	* Returns the amount of multiprocessors (mps) inside this device.
	*/
	int get_mps() const noexcept;

	/*
	* Returns the mount of threads preferred by user to use.
	* This can be changed using the "update_threads()" function.
	*/
	int get_threads() const noexcept;

	/*
	* Returns maximum of threads that can be utilized by device.
	* On most nvidia gpus this value is 1024.
	*/
	int get_max_threads() const noexcept;

	/*
	* Returns the amount of threads per multriprocessor on this device.
	*/
	int get_threads_per_mp() const noexcept;

	/*
	* Returns the amount of blocks per multriprocessors on this device.
	*/
	int get_blocks_per_mp() const noexcept;

	/*
	* Returns whether ecc is enabled on this device.
	*/
	int get_ecc() const noexcept;

	/*
	* Tries to update the interal threads value.
	* Returns:
		a) -1 if new the new number is invalid
		b)	1 if the value is valid but not a multiple of 32
		c)	0 if successful
	*/
	int update_threads(int _threads) noexcept;

	/*
	* 
	* Allocates size amount of bytes on the gpu and returns the pointer and idex in internal state
	* Do not attempt to free the pointer on your own call the appropriate mjethod for that
	* 
	* This method saves the pointer to internal memory which frees it in case of any crash!
	* Crashes on fail !!!
	*/
	float* allocate_memory(size_t size) noexcept;
	
	/*
	* 
	* Host to device
	* 
	* Works just like memcpy 
	* Copies to given memory
	* 
	* Also works with pointer offseting but it's better to use the memcpy_offset method for this
	*/
	int memcpy_host(float* src, float* dst, size_t size) noexcept;

	/*
	*
	* Device to host
	*
	* Works just like memcpy
	* Copies to given memory
	*
	* Also works with pointer offseting but it's better to use the memcpy_offset method for this
	*/
	int memcpy_device(float* src, float* dst, size_t size) noexcept;

	/*
	*
	* Host to device
	*
	* Works just like memcpy
	* Copies to given memory
	* 
	* Does not validate pointer given and does not ensure any error handling at all
	* Only cuda errors are handled 
	* Anything is possible with this :)
	*
	* Also works with pointer offseting but it's better to use the memcpy_offset method for this
	*/
	int memcpy_host_unsafe(float* src, float* dst, size_t size) noexcept;

	/*
	*
	* Device to host
	*
	* Works just like memcpy
	* Copies to given memory
	* 
	* Does not validate pointer given and does not ensure any error handling at all
	* Anything is possible with this :)
	*
	* Also works with pointer offseting but it's better to use the memcpy_offset method for this
	*/
	int memcpy_device_unsafe(float* src, float* dst, size_t size) noexcept;

	/*
	* Works just like free() would except on the gpu
	*/
	int free_memory(float* p) noexcept;

    /*
     * Works just like memset, sets n_elems to value
     */
    int memset(float* p, int value, const size_t n_elems) noexcept;

    /*
     * sets first n_elmens on the p pointer to random numbers
     */
    int random_numbers(float* p, const size_t n_elems) noexcept;

	//-----=====================-----

	void matmul_ver1_cpu(float* a, float* b, float* c, size_t a_col, 
            size_t a_row, size_t b_col, size_t b_row, 
            size_t c_col, size_t c_row) const noexcept;

	void matmul_ver1_gpu(float* a, float* b, float* c, size_t a_col, 
            size_t a_row, size_t b_col, size_t b_row, 
            size_t c_col, size_t c_row, ActivationFunction actv_fn) const noexcept;

	void matadd_ver1(float* a, float* b, float* c, size_t a_col, 
            size_t a_row, size_t b_col, size_t b_row, 
            size_t c_col, size_t c_row) const noexcept;

    void conv_ver1(const float* kernel, const float* dat, float* output, 
            const size_t kernel_dim, const size_t dat_dim, 
            const size_t out_dim, ActivationFunction actv_fn) const noexcept;

};

}
