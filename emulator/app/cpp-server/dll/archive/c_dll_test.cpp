#include "c_dll_test.hpp"

Stack* createStack(size_t size) {
    return new Stack(size);
}

void destroyStack(Stack* stack) {
    delete stack;
}

bool stack_push(Stack* stack, int val) {
    return stack->push(val);
}

bool stack_pop(Stack* stack) {
    return stack->pop();
}

size_t stack_size(Stack* stack) {
    return stack->stack_size();
}

size_t stack_current(Stack* stack) {
    return stack->stack_current();
}
