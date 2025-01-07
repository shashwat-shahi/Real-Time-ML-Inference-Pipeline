#include "timing.hpp"
#include <pthread.h>
#include <sched.h>
#include <unistd.h>
#include <sys/mman.h>
#include <algorithm>
#include <vector>

namespace realtime_ml {

bool RealtimeThreadConfig::set_realtime_priority(int priority) noexcept {
    struct sched_param param;
    param.sched_priority = priority;
    
    if (pthread_setschedparam(pthread_self(), SCHED_FIFO, &param) != 0) {
        return false;
    }
    
    return true;
}

bool RealtimeThreadConfig::set_cpu_affinity(int cpu_id) noexcept {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(cpu_id, &cpuset);
    
    if (pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset) != 0) {
        return false;
    }
    
    return true;
}

bool RealtimeThreadConfig::configure_realtime_thread(int cpu_id, int priority) noexcept {
    // Set CPU affinity first
    if (!set_cpu_affinity(cpu_id)) {
        return false;
    }
    
    // Then set real-time priority
    if (!set_realtime_priority(priority)) {
        return false;
    }
    
    // Disable thread migration
    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
        // Continue even if mlockall fails (may not have permissions)
    }
    
    return true;
}

void RealtimeThreadConfig::warmup_caches() noexcept {
    // Warm up CPU caches by doing some computation
    volatile int sum = 0;
    for (int i = 0; i < 10000; ++i) {
        sum += i * i;
    }
    
    // Touch some memory pages to ensure they're mapped
    constexpr size_t warmup_size = 1024 * 1024; // 1MB
    static thread_local std::vector<uint8_t> warmup_buffer(warmup_size);
    
    for (size_t i = 0; i < warmup_size; i += 4096) {
        warmup_buffer[i] = static_cast<uint8_t>(i & 0xFF);
    }
    
    // Ensure the buffer is actually used
    volatile uint8_t dummy = warmup_buffer[warmup_size - 1];
    (void)dummy;
}

} // namespace realtime_ml