#include "scheduler.hpp"
#include <chrono>
#include <algorithm>

namespace realtime_ml {

RealtimeScheduler::RealtimeScheduler(const std::vector<int>& cpu_affinity, int priority)
    : cpu_affinity_(cpu_affinity), scheduler_priority_(priority) {
    
    if (cpu_affinity_.empty()) {
        // Default to using first 4 CPUs
        cpu_affinity_ = {0, 1, 2, 3};
    }
}

RealtimeScheduler::~RealtimeScheduler() {
    stop();
}

void RealtimeScheduler::start() {
    if (running_.load()) {
        return; // Already running
    }
    
    running_.store(true);
    
    // Create worker threads
    worker_threads_.reserve(cpu_affinity_.size());
    for (size_t i = 0; i < cpu_affinity_.size(); ++i) {
        worker_threads_.emplace_back(&RealtimeScheduler::worker_thread_main, 
                                   this, cpu_affinity_[i]);
    }
}

void RealtimeScheduler::stop() {
    if (!running_.load()) {
        return; // Already stopped
    }
    
    running_.store(false);
    
    // Join all worker threads
    for (auto& thread : worker_threads_) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    worker_threads_.clear();
}

bool RealtimeScheduler::submit_task(std::function<void()> work, uint8_t priority, 
                                   uint64_t deadline_us) noexcept {
    if (!running_.load() || priority >= NUM_PRIORITY_LEVELS) {
        return false;
    }
    
    const uint64_t absolute_deadline = deadline_us > 0 ? 
        current_time_us() + deadline_us : 0;
    
    Task task(std::move(work), priority, absolute_deadline);
    
    return priority_queues_[priority].try_enqueue(std::move(task));
}

size_t RealtimeScheduler::pending_tasks() const noexcept {
    size_t total = 0;
    for (const auto& queue : priority_queues_) {
        total += queue.size();
    }
    return total;
}

void RealtimeScheduler::worker_thread_main(int cpu_id) {
    // Configure thread for real-time execution
    RealtimeThreadConfig::configure_realtime_thread(cpu_id, scheduler_priority_);
    RealtimeThreadConfig::warmup_caches();
    
    while (running_.load()) {
        if (!execute_next_task()) {
            // No tasks available, yield CPU briefly
            std::this_thread::yield();
        }
    }
}

bool RealtimeScheduler::execute_next_task() {
    // Check queues from highest to lowest priority
    for (int priority = NUM_PRIORITY_LEVELS - 1; priority >= 0; --priority) {
        Task task;
        if (priority_queues_[priority].try_dequeue(task)) {
            const uint64_t start_time = current_time_us();
            
            // Check if task has missed its deadline
            if (task.deadline_us > 0 && start_time > task.deadline_us) {
                // Task missed deadline, skip execution but record latency
                const uint64_t scheduling_latency = start_time - 
                    (task.deadline_us - 100); // Assume 100μs target deadline
                scheduling_stats_.record_latency(scheduling_latency);
                continue;
            }
            
            // Execute the task
            {
                ScopedLatencyMeasure exec_measure(execution_stats_);
                if (task.work) {
                    task.work();
                }
            }
            
            const uint64_t end_time = current_time_us();
            const uint64_t total_latency = end_time - start_time;
            scheduling_stats_.record_latency(total_latency);
            
            return true;
        }
    }
    
    return false; // No tasks found
}

uint64_t RealtimeScheduler::current_time_us() const noexcept {
    const auto now = std::chrono::high_resolution_clock::now();
    const auto duration = now.time_since_epoch();
    return std::chrono::duration_cast<std::chrono::microseconds>(duration).count();
}

} // namespace realtime_ml