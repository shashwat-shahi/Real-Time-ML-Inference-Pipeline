#include "memory_pool.hpp"
#include <iostream>
#include <vector>
#include <thread>
#include <chrono>
#include <atomic>
#include <random>

using namespace realtime_ml;

bool test_allocation_performance() {
    std::cout << "Testing memory allocation performance...\n";
    
    constexpr int NUM_ALLOCATIONS = 100000;
    MemoryPool<1024, 1024> pool;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    std::vector<void*> ptrs;
    ptrs.reserve(1024);
    
    // Allocation benchmark
    for (int i = 0; i < NUM_ALLOCATIONS; ++i) {
        void* ptr = pool.allocate();
        if (ptr) {
            ptrs.push_back(ptr);
        }
        
        // Deallocate when pool is full
        if (ptrs.size() >= 1024 || !ptr) {
            for (void* p : ptrs) {
                pool.deallocate(p);
            }
            ptrs.clear();
        }
    }
    
    // Clean up remaining allocations
    for (void* p : ptrs) {
        pool.deallocate(p);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    double ops_per_second = static_cast<double>(NUM_ALLOCATIONS * 2) / duration.count() * 1000000.0;
    
    std::cout << "Performed " << (NUM_ALLOCATIONS * 2) << " operations in " 
              << duration.count() << " μs\n";
    std::cout << "Performance: " << ops_per_second << " ops/second\n";
    
    std::cout << "PASS: Allocation performance\n";
    return true;
}

bool test_concurrent_access() {
    std::cout << "Testing concurrent memory access...\n";
    
    constexpr int NUM_THREADS = 4;
    constexpr int OPERATIONS_PER_THREAD = 10000;
    
    MemoryPool<512, 256> pool;
    std::atomic<int> successful_operations{0};
    std::atomic<int> failed_operations{0};
    
    std::vector<std::thread> threads;
    
    for (int i = 0; i < NUM_THREADS; ++i) {
        threads.emplace_back([&pool, &successful_operations, &failed_operations]() {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<> dis(1, 10);
            
            std::vector<void*> local_ptrs;
            
            for (int j = 0; j < OPERATIONS_PER_THREAD; ++j) {
                if (dis(gen) <= 5 && !local_ptrs.empty()) {
                    // Deallocate
                    size_t index = gen() % local_ptrs.size();
                    pool.deallocate(local_ptrs[index]);
                    local_ptrs.erase(local_ptrs.begin() + index);
                    successful_operations.fetch_add(1);
                } else {
                    // Allocate
                    void* ptr = pool.allocate();
                    if (ptr) {
                        local_ptrs.push_back(ptr);
                        successful_operations.fetch_add(1);
                    } else {
                        failed_operations.fetch_add(1);
                    }
                }
            }
            
            // Clean up
            for (void* ptr : local_ptrs) {
                pool.deallocate(ptr);
            }
        });
    }
    
    for (auto& thread : threads) {
        thread.join();
    }
    
    std::cout << "Successful operations: " << successful_operations.load() << "\n";
    std::cout << "Failed operations: " << failed_operations.load() << "\n";
    std::cout << "Pool allocated count: " << pool.allocated_count() << "\n";
    
    if (pool.allocated_count() != 0) {
        std::cout << "FAIL: Pool should have 0 allocated blocks after cleanup\n";
        return false;
    }
    
    std::cout << "PASS: Concurrent access\n";
    return true;
}

bool test_memory_layout() {
    std::cout << "Testing memory layout and alignment...\n";
    
    MemoryPool<64, 16> pool;
    
    std::vector<void*> ptrs;
    
    // Allocate all blocks
    for (int i = 0; i < 16; ++i) {
        void* ptr = pool.allocate();
        if (!ptr) {
            std::cout << "FAIL: Could not allocate block " << i << "\n";
            return false;
        }
        
        // Check alignment (should be at least 64-byte aligned for cache efficiency)
        uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
        if (addr % 64 != 0) {
            std::cout << "FAIL: Block " << i << " is not properly aligned\n";
            return false;
        }
        
        // Check ownership
        if (!pool.owns(ptr)) {
            std::cout << "FAIL: Pool does not own allocated pointer\n";
            return false;
        }
        
        ptrs.push_back(ptr);
    }
    
    // Check that pointers are within expected range
    uintptr_t min_addr = UINTPTR_MAX;
    uintptr_t max_addr = 0;
    
    for (void* ptr : ptrs) {
        uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
        min_addr = std::min(min_addr, addr);
        max_addr = std::max(max_addr, addr);
    }
    
    uintptr_t total_size = max_addr - min_addr + 64; // Add one block size
    std::cout << "Memory layout spans " << total_size << " bytes\n";
    
    // Clean up
    for (void* ptr : ptrs) {
        pool.deallocate(ptr);
    }
    
    std::cout << "PASS: Memory layout\n";
    return true;
}

int main() {
    std::cout << "Memory Benchmark Tests\n";
    std::cout << "=====================\n\n";
    
    bool all_passed = true;
    
    all_passed &= test_allocation_performance();
    all_passed &= test_concurrent_access();
    all_passed &= test_memory_layout();
    
    std::cout << "\n=====================\n";
    if (all_passed) {
        std::cout << "ALL BENCHMARKS PASSED!\n";
        return 0;
    } else {
        std::cout << "SOME BENCHMARKS FAILED!\n";
        return 1;
    }
}