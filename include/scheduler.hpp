#pragma once

#include "lockfree_queue.hpp"
#include "memory_pool.hpp"
#include "timing.hpp"
#include <functional>
#include <thread>
#include <atomic>
#include <vector>

namespace realtime_ml {

/**
 * Task structure for the real-time scheduler
 */
struct Task {
    std::function<void()> work;
    uint64_t priority;
    uint64_t deadline_us;  // Deadline in microseconds from epoch
    
    Task() = default;
    Task(std::function<void()> w, uint64_t p, uint64_t d) 
        : work(std::move(w)), priority(p), deadline_us(d) {}
};

/**
 * Real-time task scheduler with deterministic execution guarantees
 * Uses lock-free queues and priority-based scheduling
 */
class RealtimeScheduler {
private:
    static constexpr size_t QUEUE_SIZE = 1024;
    static constexpr size_t NUM_PRIORITY_LEVELS = 8;
    
    // Priority queues (higher index = higher priority)
    std::array<LockFreeQueue<Task, QUEUE_SIZE>, NUM_PRIORITY_LEVELS> priority_queues_;
    
    // Worker threads
    std::vector<std::thread> worker_threads_;
    std::atomic<bool> running_{false};
    
    // Performance monitoring
    LatencyStats scheduling_stats_;
    LatencyStats execution_stats_;
    
    // Thread configuration
    std::vector<int> cpu_affinity_;
    int scheduler_priority_;
    
public:
    explicit RealtimeScheduler(const std::vector<int>& cpu_affinity = {}, 
                              int priority = 99);
    ~RealtimeScheduler();
    
    // Non-copyable, non-movable
    RealtimeScheduler(const RealtimeScheduler&) = delete;
    RealtimeScheduler& operator=(const RealtimeScheduler&) = delete;
    RealtimeScheduler(RealtimeScheduler&&) = delete;
    RealtimeScheduler& operator=(RealtimeScheduler&&) = delete;
    
    /**
     * Start the scheduler
     */
    void start();
    
    /**
     * Stop the scheduler
     */
    void stop();
    
    /**
     * Submit a task for execution
     * priority: 0-7 (7 is highest priority)
     * deadline_us: deadline in microseconds from now (0 = no deadline)
     */
    bool submit_task(std::function<void()> work, uint8_t priority = 4, 
                    uint64_t deadline_us = 0) noexcept;
    
    /**
     * Submit a high-priority inference task
     */
    bool submit_inference_task(std::function<void()> work, 
                              uint64_t deadline_us = 100) noexcept {
        return submit_task(std::move(work), 7, deadline_us);
    }
    
    /**
     * Get scheduling statistics
     */
    const LatencyStats& get_scheduling_stats() const noexcept {
        return scheduling_stats_;
    }
    
    /**
     * Get execution statistics
     */
    const LatencyStats& get_execution_stats() const noexcept {
        return execution_stats_;
    }
    
    /**
     * Get number of pending tasks across all queues
     */
    size_t pending_tasks() const noexcept;

private:
    /**
     * Worker thread main loop
     */
    void worker_thread_main(int cpu_id);
    
    /**
     * Find and execute the next highest priority task
     */
    bool execute_next_task();
    
    /**
     * Get current time in microseconds since epoch
     */
    uint64_t current_time_us() const noexcept;
};

} // namespace realtime_ml