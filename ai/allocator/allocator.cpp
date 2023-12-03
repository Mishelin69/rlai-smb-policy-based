#include "./allocator.hpp"
#include <iostream>

Allocator::Allocator():
    entry_n(0), blocks(std::vector<AllocatorBlock>()), entries(std::vector<AllocatorEntry>()) {

        this->gpu = GPU::Device(0);

    }

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

int Allocator::alloc_new_block(const size_t block_size) {

    float* cuda_p = this->gpu.allocate_memory(sizeof(float) * block_size);

    if (!cuda_p) {
        std::cerr << "Couln't allocate memory on the gpu!" << std::endl;
        return -1;
    }

    this->blocks.push_back( AllocatorBlock { cuda_p, block_size, blocks.size(), block_size } );

    return 0;
}

float* Allocator::alloc_space(const size_t block_size) {

    size_t blck_id = 0;

    for (auto& b : this->blocks) {

        if (b.block_free >= block_size) {

            b.block_free -= block_size;
            float* cuda_p = b.block_root + b.block_free;

            this->entries.push_back( AllocatorEntry { cuda_p, block_size, blck_id });

            return cuda_p;
        }

        blck_id += 1;
    }

    return NULL;
}
