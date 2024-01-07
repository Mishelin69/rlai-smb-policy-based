#include "dll_test.hpp"
#include <new>

Stack::Stack(size_t stack_size):
    stack_at(0), _stack_size(stack_size) {

    int* dat_ptr = new (std::nothrow) int[stack_size];

    if (!dat_ptr) {
        exit(-1);
    }

    this->_stack = dat_ptr;
}

Stack::~Stack() {

    if (!_stack) {
        return;
    }

    delete[] _stack;
}

bool Stack::push(int val) {

    if (stack_at == _stack_size) {
        return false;
    }

    _stack[stack_at] = val;
    stack_at += 1;

    return true;
}

bool Stack::pop() {

    if (stack_at < 1) {
        return false;
    }

    stack_at -= 1;

    return _stack[stack_at];
}

size_t Stack::stack_size() {
    return _stack_size;
}

size_t Stack::stack_current() {
    return stack_at;
}