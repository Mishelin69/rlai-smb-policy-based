#pragma once

#include "../../gpu/gpu.h"

struct AllocatorEntry {
    float* p;
    size_t p_size;
    size_t block_id;
};

struct AllocatorBlock {

    float* block_root;
    size_t block_size;
    size_t block_id;
    size_t block_free;

};

class Allocator {

private:

    GPU::Device gpu;
    size_t entry_n;
    
    std::vector<AllocatorBlock> blocks;
    std::vector<AllocatorEntry> entries;

public:

    Allocator();
    ~Allocator();
    Allocator(Allocator& other) = default;

    int alloc_new_block(const size_t block_size);
    float* alloc_space(const size_t block_size);

};
