#include <napi.h> //ignore
#include <Windows.h>  // Include Windows API headers

#include "../dll/dll_test.hpp"

HMODULE dll_handle = nullptr;

Stack* (*createStack)(size_t);
void (*destroyStack)(Stack *);

bool (*stack_push)(Stack*, int);
bool (*stack_pop)(Stack*);
size_t (*stack_size)(Stack*);
size_t (*stack_current)(Stack*);

void load_dll() {
    
    if (dll_handle) {
        return;
    }

    dll_handle = LoadLibrary("../dll/stack_dll.dll");

    if (!dll_handle) {
        return;
    }

    createStack = (Stack* (*)(size_t))GetProcAddress(dll_handle, "createStack");
    destroyStack = (void (*)(Stack*))GetProcAddress(dll_handle, "destroyStack");

    stack_push = (bool (*)(Stack*, int))GetProcAddress(dll_handle, "stack_push");
    stack_pop = (bool (*)(Stack*))GetProcAddress(dll_handle, "stack_pop");
    stack_size = (size_t (*)(Stack*))GetProcAddress(dll_handle, "stack_size");
    stack_current = (size_t (*)(Stack*))GetProcAddress(dll_handle, "stack_current");
}

void unload_dll() {

    if (dll_handle) {
        FreeLibrary(dll_handle);
        dll_handle = nullptr;
    }
}

class WrappedStack: public Napi::ObjectWrap<WrappedStack> {

public: 
    static Napi::Object Init(Napi::Env env, Napi::Object exports);
    WrappedStack(const Napi::CallbackInfo& info);

    // ~WrappedStack();

    Napi::Value push(const Napi::CallbackInfo& info);
    Napi::Value pop(const Napi::CallbackInfo& info);

    Napi::Value size(const Napi::CallbackInfo& info);
    Napi::Value current(const Napi::CallbackInfo& info);

    static Napi::Value NewInstance(const Napi::CallbackInfo& info);

    static Napi::FunctionReference constructor;
    Stack* _stack;
};

Napi::Object WrappedStack::Init(Napi::Env env, Napi::Object exports) {

    load_dll();

    Napi::HandleScope scope(env);

    Napi::Function func = DefineClass(env, "WrappedStack", {
        StaticMethod("NewInstance", &WrappedStack::NewInstance, napi_default),
        InstanceMethod("push", &WrappedStack::push, napi_default),
        InstanceMethod("pop", &WrappedStack::pop, napi_default),
        InstanceMethod("size", &WrappedStack::size, napi_default),
        InstanceMethod("current", &WrappedStack::current, napi_default),
    });

    Napi::FunctionReference* constructor = new Napi::FunctionReference();

    *constructor = Napi::Persistent(func);
    exports.Set("WrappedStack", func);

    env.SetInstanceData<Napi::FunctionReference>(constructor);

    return exports;
}

WrappedStack::WrappedStack(const Napi::CallbackInfo& info) 
    : Napi::ObjectWrap<WrappedStack>(info) {
    
    Napi::Env env = info.Env();
    
    // Extract arguments, if any, and create the underlying Stack instance
    size_t size = info[0].IsNumber() ? info[0].As<Napi::Number>().Uint32Value() : 0;
    this->_stack = createStack(size);
}

Napi::Value WrappedStack::push(const Napi::CallbackInfo& info) {

    Napi::Env env = info.Env();
    size_t val = info[0].IsNumber() ? info[0].As<Napi::Number>().Uint32Value() : 0;
    
    // Implement logic to call stack_->push() and handle return value
    bool result = stack_push(this->_stack, val);

    return Napi::Boolean::New(env, result);
}

Napi::Value WrappedStack::pop(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    // Implement logic to call stack_->pop() and handle return value
    bool result = stack_pop(this->_stack);

    return Napi::Boolean::New(env, result);
}

Napi::Value WrappedStack::size(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    // Implement logic to call stack_->size() and return the value
    size_t result = stack_size(_stack);

    return Napi::Number::New(env, static_cast<uint32_t>(result));
}

Napi::Value WrappedStack::current(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    // Implement logic to call stack_->current() and return the value
    size_t result = stack_current(_stack);

    return Napi::Number::New(env, static_cast<uint32_t>(result));
}

Napi::Value WrappedStack::NewInstance(const Napi::CallbackInfo& info) {

    Napi::FunctionReference* constructor =
      info.Env().GetInstanceData<Napi::FunctionReference>();

  return constructor->New({ Napi::Number::New(info.Env(), 42) });
}

Napi::Object InitAll(Napi::Env env, Napi::Object exports) {

    WrappedStack::Init(env, exports);

    return exports;
}

NODE_API_MODULE(testaddon, InitAll)