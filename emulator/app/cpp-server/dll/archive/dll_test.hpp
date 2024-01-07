#pragma once

class Stack {

private:

    size_t _stack_size;
    int* _stack;

    size_t stack_at;

public:

    bool push(int val);
    bool pop();
    size_t stack_size();
    size_t stack_current();

    ~Stack();
    Stack(size_t _stack_size);


};