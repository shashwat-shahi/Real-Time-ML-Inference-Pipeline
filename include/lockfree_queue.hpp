#pragma once

#include <atomic>
#include <memory>
#include <array>
#include <cstdint>

namespace realtime_ml {

/**
 * Lock-free SPSC (Single Producer Single Consumer) queue optimized for real-time systems
 * Uses cache-aligned atomic operations to minimize contention and ensure deterministic performance
 */
template<typename T, size_t Capacity>
class LockFreeQueue {
private:
    static_assert((Capacity & (Capacity - 1)) == 0, "Capacity must be power of 2");
    
    struct alignas(64) CacheAlignedAtomic {
        std::atomic<size_t> value{0};
    };
    
    alignas(64) std::array<T, Capacity> buffer_;
    alignas(64) CacheAlignedAtomic head_;
    alignas(64) CacheAlignedAtomic tail_;
    
    static constexpr size_t mask_ = Capacity - 1;

public:
    LockFreeQueue() = default;
    
    // Non-copyable, non-movable for real-time guarantees
    LockFreeQueue(const LockFreeQueue&) = delete;
    LockFreeQueue& operator=(const LockFreeQueue&) = delete;
    LockFreeQueue(LockFreeQueue&&) = delete;
    LockFreeQueue& operator=(LockFreeQueue&&) = delete;
    
    /**
     * Try to enqueue an item (producer side)
     * Returns true if successful, false if queue is full
     * Guaranteed wait-free operation
     */
    [[nodiscard]] bool try_enqueue(const T& item) noexcept {
        const size_t current_tail = tail_.value.load(std::memory_order_relaxed);
        const size_t next_tail = (current_tail + 1) & mask_;
        
        if (next_tail == head_.value.load(std::memory_order_acquire)) {
            return false; // Queue is full
        }
        
        buffer_[current_tail] = item;
        tail_.value.store(next_tail, std::memory_order_release);
        return true;
    }
    
    /**
     * Try to dequeue an item (consumer side)
     * Returns true if successful, false if queue is empty
     * Guaranteed wait-free operation
     */
    [[nodiscard]] bool try_dequeue(T& item) noexcept {
        const size_t current_head = head_.value.load(std::memory_order_relaxed);
        
        if (current_head == tail_.value.load(std::memory_order_acquire)) {
            return false; // Queue is empty
        }
        
        item = buffer_[current_head];
        head_.value.store((current_head + 1) & mask_, std::memory_order_release);
        return true;
    }
    
    /**
     * Get approximate size (may not be exact due to concurrent operations)
     */
    [[nodiscard]] size_t size() const noexcept {
        const size_t tail = tail_.value.load(std::memory_order_acquire);
        const size_t head = head_.value.load(std::memory_order_acquire);
        return (tail - head) & mask_;
    }
    
    /**
     * Check if queue is empty (approximate)
     */
    [[nodiscard]] bool empty() const noexcept {
        return head_.value.load(std::memory_order_acquire) == 
               tail_.value.load(std::memory_order_acquire);
    }
    
    /**
     * Get maximum capacity
     */
    [[nodiscard]] static constexpr size_t capacity() noexcept {
        return Capacity;
    }
};

} // namespace realtime_ml