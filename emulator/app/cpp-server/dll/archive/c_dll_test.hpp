#pragma once

#include "dll_test.hpp"

extern "C" {

    __declspec(dllexport) void destroyStack(Stack* stack);
    __declspec(dllexport) Stack* createStack(size_t size);

    __declspec(dllexport) bool stack_push(Stack* stack, int val);
    __declspec(dllexport) bool stack_pop(Stack* stack);
    __declspec(dllexport) size_t stack_size(Stack* stack);
    __declspec(dllexport) size_t stack_current(Stack* stack);
}