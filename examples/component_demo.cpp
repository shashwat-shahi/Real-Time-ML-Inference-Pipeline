#include "lockfree_queue.hpp"
#include "memory_pool.hpp"
#include "scheduler.hpp"
#include "timing.hpp"
#include <iostream>
#include <random>
#include <iomanip>

using namespace realtime_ml;

void demonstrate_lockfree_queue() {
    std::cout << "\n=== Lock-Free Queue Demo ===\n";
    
    LockFreeQueue<int, 16> queue;
    
    // Producer
    std::cout << "Enqueuing items: ";
    for (int i = 0; i < 10; ++i) {
        if (queue.try_enqueue(i)) {
            std::cout << i << " ";
        }
    }
    std::cout << "\nQueue size: " << queue.size() << "\n";
    
    // Consumer
    std::cout << "Dequeuing items: ";
    int value;
    while (queue.try_dequeue(value)) {
        std::cout << value << " ";
    }
    std::cout << "\nQueue size: " << queue.size() << "\n";
}

void demonstrate_memory_pool() {
    std::cout << "\n=== Memory Pool Demo ===\n";
    
    MemoryPool<1024, 8> pool;
    
    std::cout << "Initial allocated count: " << pool.allocated_count() << "\n";
    
    // Allocate some blocks
    std::vector<void*> ptrs;
    for (int i = 0; i < 5; ++i) {
        void* ptr = pool.allocate();
        if (ptr) {
            ptrs.push_back(ptr);
            std::cout << "Allocated block " << i << " at " << ptr << "\n";
        }
    }
    
    std::cout << "Allocated count: " << pool.allocated_count() << "\n";
    
    // Deallocate blocks
    for (void* ptr : ptrs) {
        pool.deallocate(ptr);
    }
    
    std::cout << "Final allocated count: " << pool.allocated_count() << "\n";
}

void demonstrate_scheduler() {
    std::cout << "\n=== Real-Time Scheduler Demo ===\n";
    
    RealtimeScheduler scheduler({0}); // Single CPU
    scheduler.start();
    
    std::atomic<int> task_counter{0};
    
    // Submit tasks with different priorities
    for (int priority = 0; priority < 8; ++priority) {
        scheduler.submit_task([&task_counter, priority]() {
            task_counter.fetch_add(1);
            std::cout << "Executed task with priority " << priority << "\n";
        }, priority, 1000); // 1ms deadline
    }
    
    // Wait for tasks to complete
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    std::cout << "Total tasks completed: " << task_counter.load() << "\n";
    
    // Get statistics
    auto stats = scheduler.get_scheduling_stats();
    std::cout << "Scheduler P99 latency: " << stats.p99_latency() << " μs\n";
    
    scheduler.stop();
}

void demonstrate_timing() {
    std::cout << "\n=== High-Resolution Timing Demo ===\n";
    
    LatencyStats stats;
    
    // Simulate some operations with varying latencies
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(10, 200);
    
    for (int i = 0; i < 100; ++i) {
        HighResTimer timer;
        
        // Simulate work
        auto delay = std::chrono::microseconds(dis(gen));
        std::this_thread::sleep_for(delay);
        
        stats.record_latency(timer.elapsed_us());
    }
    
    std::cout << "Statistics from 100 measurements:\n";
    std::cout << "  Average: " << std::fixed << std::setprecision(1) 
              << stats.average_latency() << " μs\n";
    std::cout << "  P95:     " << stats.p95_latency() << " μs\n";
    std::cout << "  P99:     " << stats.p99_latency() << " μs\n";
    std::cout << "  Min:     " << stats.min_latency() << " μs\n";
    std::cout << "  Max:     " << stats.max_latency() << " μs\n";
}

int main() {
    std::cout << "Real-Time ML Inference Pipeline - Component Demonstration\n";
    std::cout << "=========================================================\n";
    
    try {
        demonstrate_lockfree_queue();
        demonstrate_memory_pool();
        demonstrate_scheduler();
        demonstrate_timing();
        
        std::cout << "\n🎉 All components demonstrated successfully!\n";
        std::cout << "\nKey Features Demonstrated:\n";
        std::cout << "✓ Lock-free data structures with wait-free operations\n";
        std::cout << "✓ Real-time memory management with deterministic allocation\n";
        std::cout << "✓ Priority-based task scheduling with microsecond precision\n";
        std::cout << "✓ High-resolution timing and performance monitoring\n";
        std::cout << "\nThese components form the foundation for achieving\n";
        std::cout << "<100μs P99 latency in ML inference workloads.\n";
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}