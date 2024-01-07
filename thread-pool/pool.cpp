#include "pool.hpp"

template<typename F, typename... Args>
auto ThreadPool::Pool::add_task(F&& f, Args&&... args) -> std::future<decltype(f(args...))> {

    //Create a function with bounded parameters ready to exec
    auto func = std::bind(std::forward<F>(f), std::forward<Args>(args)...);

    //Encapsulate it into a shared pointer in order to be able to copy construct / assign
    auto task_ptr = std::make_shared<std::packaged_task<decltype(f(args...))()>>(func);

    //Wrap the task pointer into a void lambda
    auto wrapper_func = [task_ptr]() {(*task_ptr)(); };

    {
        std::lock_guard<std::mutex> lock(mutex);
        this->queue.push(wrapper_func);

        //Wake up threads if waiting;
        conditional_variable.notify_one();
    }

    return task_ptr->get_future();
}

ThreadPool::Pool::Pool(const uint32_t size): 
        threads(std::vector<std::thread>(size)),
        shutdown_condition(false) {

    for (size_t i = 0; i < size; ++i) {
        threads[i] = std::thread(ThreadPool::Worker(this));
    } 
}

ThreadPool::Pool::~Pool() {
    Shutdown();
}

void ThreadPool::Pool::Shutdown() {

    {
        //this will wait for lock to be free to get (lock.get())
        //then well set the shutdown_condition to true
        //then well notify all the threads which will make them exit
        std::lock_guard<std::mutex> lock(this->mutex);
        shutdown_condition = true;
        conditional_variable.notify_all();
    }
}

uint32_t ThreadPool::Pool::queue_size() const noexcept {
    //Acquire lock because of atomics
    return queue.size();
}

//==========Worker==========

void ThreadPool::Worker::operator()() {

    std::unique_lock<std::mutex> lock(thread_pool->mutex);

    while (!thread_pool->shutdown_condition || 
            (thread_pool->shutdown_condition && !thread_pool->queue.empty())) {

        thread_pool->conditional_variable.wait(lock, [this] {
                return this->thread_pool->shutdown_condition || !this->thread_pool->queue.empty();
                });

        if (!this->thread_pool->queue.empty()) {

            auto func = thread_pool->queue.front();
            thread_pool->queue.pop();

            lock.unlock();
            func();
            lock.lock();

        }
    }
}
