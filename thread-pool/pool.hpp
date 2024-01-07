#pragma once

#include <cstdint>
#include <functional>
#include <future>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

namespace ThreadPool {


class Pool {

public:
    friend class Worker;

    template<typename F, typename... Args>
    auto add_task(F&& f, Args&&... args) -> std::future<decltype(f(args...))>;

    Pool(const uint32_t size);
    ~Pool();

    Pool(const Pool& othrer) = delete;
    Pool(const Pool&& othrer) = delete;

    void Shutdown();

    uint32_t queue_size() const noexcept;

private:

    mutable std::mutex mutex;
    std::condition_variable conditional_variable;

    std::vector<std::thread> threads;
    bool shutdown_condition;

    std::queue<std::function<void()>> queue;

};

class Worker {

public:

    Worker(Pool* pool): 
        thread_pool(pool) { }

    void operator()(); 

private: 
    
    Pool* thread_pool;
};

}
