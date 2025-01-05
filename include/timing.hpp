#pragma once

#include <chrono>
#include <cstdint>
#include <thread>
#include <array>
#include <algorithm>

namespace realtime_ml {

/**
 * High-resolution timing utilities for real-time systems
 * Uses CPU TSC (Time Stamp Counter) for microsecond precision
 */
class HighResTimer {
private:
    using clock_type = std::chrono::high_resolution_clock;
    using time_point = clock_type::time_point;
    
    time_point start_time_;
    
public:
    HighResTimer() : start_time_(clock_type::now()) {}
    
    /**
     * Reset the timer
     */
    void reset() noexcept {
        start_time_ = clock_type::now();
    }
    
    /**
     * Get elapsed time in nanoseconds
     */
    [[nodiscard]] uint64_t elapsed_ns() const noexcept {
        const auto now = clock_type::now();
        const auto duration = now - start_time_;
        return std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count();
    }
    
    /**
     * Get elapsed time in microseconds
     */
    [[nodiscard]] uint64_t elapsed_us() const noexcept {
        return elapsed_ns() / 1000;
    }
    
    /**
     * Get elapsed time in milliseconds
     */
    [[nodiscard]] uint64_t elapsed_ms() const noexcept {
        return elapsed_ns() / 1000000;
    }
};

/**
 * Latency measurement and statistics collection
 */
class LatencyStats {
private:
    static constexpr size_t MAX_SAMPLES = 10000;
    
    std::array<uint64_t, MAX_SAMPLES> samples_;
    size_t sample_count_{0};
    size_t current_index_{0};
    uint64_t min_latency_{UINT64_MAX};
    uint64_t max_latency_{0};
    uint64_t total_latency_{0};
    
public:
    /**
     * Record a latency sample in microseconds
     */
    void record_latency(uint64_t latency_us) noexcept {
        samples_[current_index_] = latency_us;
        current_index_ = (current_index_ + 1) % MAX_SAMPLES;
        
        if (sample_count_ < MAX_SAMPLES) {
            ++sample_count_;
        } else {
            // Remove oldest sample from total
            total_latency_ -= samples_[current_index_];
        }
        
        total_latency_ += latency_us;
        min_latency_ = std::min(min_latency_, latency_us);
        max_latency_ = std::max(max_latency_, latency_us);
    }
    
    /**
     * Get average latency in microseconds
     */
    [[nodiscard]] double average_latency() const noexcept {
        return sample_count_ > 0 ? static_cast<double>(total_latency_) / sample_count_ : 0.0;
    }
    
    /**
     * Get minimum latency in microseconds
     */
    [[nodiscard]] uint64_t min_latency() const noexcept {
        return min_latency_ == UINT64_MAX ? 0 : min_latency_;
    }
    
    /**
     * Get maximum latency in microseconds
     */
    [[nodiscard]] uint64_t max_latency() const noexcept {
        return max_latency_;
    }
    
    /**
     * Get percentile latency (e.g., 0.99 for P99)
     */
    [[nodiscard]] uint64_t percentile_latency(double p) const noexcept {
        if (sample_count_ == 0) return 0;
        
        // Copy and sort samples
        std::array<uint64_t, MAX_SAMPLES> sorted_samples;
        const size_t count = std::min(sample_count_, MAX_SAMPLES);
        
        for (size_t i = 0; i < count; ++i) {
            sorted_samples[i] = samples_[i];
        }
        
        std::sort(sorted_samples.begin(), sorted_samples.begin() + count);
        
        const size_t index = static_cast<size_t>(p * (count - 1));
        return sorted_samples[index];
    }
    
    /**
     * Get P99 latency
     */
    [[nodiscard]] uint64_t p99_latency() const noexcept {
        return percentile_latency(0.99);
    }
    
    /**
     * Get P95 latency
     */
    [[nodiscard]] uint64_t p95_latency() const noexcept {
        return percentile_latency(0.95);
    }
    
    /**
     * Get sample count
     */
    [[nodiscard]] size_t sample_count() const noexcept {
        return sample_count_;
    }
    
    /**
     * Reset all statistics
     */
    void reset() noexcept {
        sample_count_ = 0;
        current_index_ = 0;
        min_latency_ = UINT64_MAX;
        max_latency_ = 0;
        total_latency_ = 0;
    }
};

/**
 * Scoped latency measurement
 */
class ScopedLatencyMeasure {
private:
    LatencyStats& stats_;
    HighResTimer timer_;
    
public:
    explicit ScopedLatencyMeasure(LatencyStats& stats) 
        : stats_(stats), timer_() {}
    
    ~ScopedLatencyMeasure() {
        stats_.record_latency(timer_.elapsed_us());
    }
    
    // Non-copyable, non-movable
    ScopedLatencyMeasure(const ScopedLatencyMeasure&) = delete;
    ScopedLatencyMeasure& operator=(const ScopedLatencyMeasure&) = delete;
    ScopedLatencyMeasure(ScopedLatencyMeasure&&) = delete;
    ScopedLatencyMeasure& operator=(ScopedLatencyMeasure&&) = delete;
};

/**
 * CPU affinity and real-time thread configuration
 */
class RealtimeThreadConfig {
public:
    /**
     * Set current thread to real-time priority
     * Returns true if successful
     */
    static bool set_realtime_priority(int priority = 99) noexcept;
    
    /**
     * Set CPU affinity for current thread
     * Returns true if successful
     */
    static bool set_cpu_affinity(int cpu_id) noexcept;
    
    /**
     * Disable thread migration and set up for real-time execution
     */
    static bool configure_realtime_thread(int cpu_id, int priority = 99) noexcept;
    
    /**
     * Warm up CPU caches and page tables
     */
    static void warmup_caches() noexcept;
};

} // namespace realtime_ml