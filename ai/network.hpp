#pragma once
#include "./allocator/allocator.hpp"

#include <cstdint>

class ConvNeuralNet {

    //soon to come layer definitions and stuff, will be implemented
    //not now tho a bit too lazy rn
    
    const uint32_t num_of_layers;
    const uint32_t in_shape;
    const uint32_t out_shape;

    //this will be on the gpu most likely
    //or allocated by allocator
    const uint32_t* conv_layer_random_indexes;
    const uint32_t num_of_random_indexes;

    Allocator alloc;
};
