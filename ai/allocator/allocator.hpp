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
    size_t padding;

};

class Allocator {

private:

    GPU::Device& gpu;
    size_t entry_n;
    
    std::vector<AllocatorBlock> blocks;
    std::vector<AllocatorEntry> entries;

    const size_t alignment;

public:

    Allocator(GPU::Device& gpu, const size_t align);
    ~Allocator();
    Allocator(Allocator& other) = default;

    int alloc_new_block(const size_t bytes);

    //return pointer to newly allocated memory if allocated otherwise null
    //block_size is number of elements you want to allocate
    float* alloc_space(const size_t n_elems);

};
