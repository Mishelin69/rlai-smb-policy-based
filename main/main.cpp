#include "../gpu/gpu.h"

#include <iostream>
#include <chrono>

void fill_array_with_random_stuff(float* arr, size_t arr_len) {


	for (size_t i = 0; i < arr_len; ++i) {
		arr[i] = ((float)(rand() % 10))/10.0;
		//printf("Rand number: %f\n", arr[i]);
	}
	//std::cout << "-----------------------------------------" << std::endl;
}

int main(int argc, char** argv) {

	GPU::Device dev;

	/*float A[4] = { 1, 3, 5, 6};
	float B[4] = { 1, 3, 4, 5};

	auto shape_pair = GPU::Device::calculate_new_mat_dims(1, 4, 4, 1);

	size_t c_size = shape_pair.first * shape_pair.second;

	float* C = new (std::nothrow) float[c_size];

	dev.matmul_ver1_cpu(A, B, C, 1, 4, 4, 1, shape_pair.first, shape_pair.second);

	for (size_t i = 0; i < c_size; ++i) {
		std::cout << "Elm " << i << ". = " << C[i] << std::endl;
	}*/

	const size_t ARR_A_X = 1024;
	const size_t ARR_A_Y = 1024;

	const size_t ARR_B_X = 1024;
	const size_t ARR_B_Y = 1024;

	const size_t ARR_A_SIZE = ARR_A_X * ARR_A_Y;
	const size_t ARR_B_SIZE = ARR_B_X * ARR_B_Y;

	auto out_shape = GPU::Device::calculate_new_mat_dims(ARR_A_X, ARR_A_Y, ARR_B_X, ARR_B_Y);

	printf("Out dims: %zu %zu\n", out_shape.first, out_shape.second);

	const size_t ARR_SIZE = out_shape.first * out_shape.second;

	float* A = new (std::nothrow) float[ARR_A_SIZE];
	float* B = new (std::nothrow) float[ARR_B_SIZE];
	float* C = new (std::nothrow) float[ARR_SIZE];
	float* C_cpu = new (std::nothrow) float[ARR_SIZE];

	if (!A || !B || !C) {
		exit(-1);
	}

	float* cudaA = dev.allocate_memory(sizeof(float) * ARR_A_SIZE);
	float* cudaB = dev.allocate_memory(sizeof(float) * ARR_B_SIZE);
	float* cudaC = dev.allocate_memory(sizeof(float) * ARR_SIZE);

	srand(time(NULL));


	fill_array_with_random_stuff(A, ARR_A_SIZE);
	fill_array_with_random_stuff(B, ARR_B_SIZE);

	auto t1 = std::chrono::high_resolution_clock::now();

	dev.memcpy_host(A, cudaA, sizeof(float) * ARR_A_SIZE);
	dev.memcpy_host(B, cudaB, sizeof(float) * ARR_B_SIZE);

	dev.matmul_ver1_gpu(cudaA, cudaB, cudaC, ARR_A_X, ARR_A_Y, ARR_B_X, ARR_B_Y, out_shape.first, out_shape.second);
	
	auto t2 = std::chrono::high_resolution_clock::now();
	auto ms_int = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);
	std::cout << "GPU time: " << ms_int.count() << "ms" << std::endl;

	t1 = std::chrono::high_resolution_clock::now();

	dev.matmul_ver1_cpu(A, B, C_cpu, ARR_A_X, ARR_A_Y, ARR_B_X, ARR_B_Y, out_shape.first, out_shape.second);

	t2 = std::chrono::high_resolution_clock::now();
	ms_int = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);
	std::cout << "CPU time: " << ms_int.count() << "ms" << std::endl;

	dev.memcpy_device(cudaC, C, sizeof(float) * ARR_SIZE);

	return 0;
}
