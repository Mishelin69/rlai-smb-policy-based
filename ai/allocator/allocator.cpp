#include "./allocator.hpp"
#include <cmath>
#include <iostream>

//ugly ass code :( but it works! <3
Allocator::Allocator(GPU::Device& gpu, const size_t align):
    entry_n(0), blocks(std::vector<AllocatorBlock>()), entries(std::vector<AllocatorEntry>()), 
    gpu(gpu), alignment(align) { }

Allocator::~Allocator() {

    for (auto& b : this->blocks) {

        if (!b.block_root) {
            std::cerr << "Memory corruption!" << std::endl;
            return;
        }

        int ret = this->gpu.free_memory(b.block_root);

        if (ret == -1) {
            std::cerr << "Error while freeing memory!" << std::endl;

            exit(-1);
        }
    }

}

int Allocator::alloc_new_block(const size_t bytes) {

    //make sure the block size is divisible by align and if not then make it mathch that size
    size_t mem_to_alloc = bytes;
    size_t extra = 0;

    if (mem_to_alloc % alignment != 0) {
        const size_t upper = (size_t) std::ceil(mem_to_alloc / alignment);
        mem_to_alloc = sizeof(float) * upper;
        extra = mem_to_alloc - bytes;
    }

    float* cuda_p = this->gpu.allocate_memory(mem_to_alloc);

    if (!cuda_p) {
        std::cerr << "Couln't allocate memory on the gpu!" << std::endl;
        return -1;
    }

    this->blocks.push_back( AllocatorBlock { cuda_p, bytes, blocks.size(), bytes, extra } );

    return 0;
}

float* Allocator::alloc_space(const size_t bytes) {

    //basically make sure the memory is aligned correctly
    size_t block_id = 0;
    size_t mem_to_alloc = bytes;
    size_t extra = 0;

    if (mem_to_alloc % alignment != 0) {
        const size_t upper = (size_t) std::ceil(mem_to_alloc / alignment);
        mem_to_alloc = sizeof(float) * upper;
        extra = mem_to_alloc - bytes;
    }

    for (auto& b : blocks) {

        if (b.block_free < mem_to_alloc) {

            block_id += 1;
            continue;
        }

        b.block_free -= mem_to_alloc;
        float* cuda_p = b.block_root + b.block_free;

        this->entries.push_back(AllocatorEntry {cuda_p, mem_to_alloc, block_id} );

        return cuda_p;
    }

    return nullptr;
}
