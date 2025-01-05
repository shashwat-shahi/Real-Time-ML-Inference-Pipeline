#pragma once

#include <memory>
#include <atomic>
#include <cstdint>
#include <new>
#include <array>

namespace realtime_ml {

/**
 * Real-time memory pool allocator with deterministic allocation/deallocation
 * Pre-allocates fixed-size blocks to avoid dynamic allocation during inference
 */
template<size_t BlockSize, size_t NumBlocks>
class MemoryPool {
private:
    struct alignas(64) Block {
        std::array<uint8_t, BlockSize> data;
        std::atomic<Block*> next{nullptr};
    };
    
    alignas(64) std::array<Block, NumBlocks> blocks_;
    alignas(64) std::atomic<Block*> free_list_;
    alignas(64) std::atomic<size_t> allocated_count_{0};
    
public:
    MemoryPool() {
        // Initialize free list in constructor (not real-time critical)
        for (size_t i = 0; i < NumBlocks - 1; ++i) {
            blocks_[i].next.store(&blocks_[i + 1], std::memory_order_relaxed);
        }
        blocks_[NumBlocks - 1].next.store(nullptr, std::memory_order_relaxed);
        free_list_.store(&blocks_[0], std::memory_order_relaxed);
    }
    
    // Non-copyable, non-movable
    MemoryPool(const MemoryPool&) = delete;
    MemoryPool& operator=(const MemoryPool&) = delete;
    MemoryPool(MemoryPool&&) = delete;
    MemoryPool& operator=(MemoryPool&&) = delete;
    
    /**
     * Allocate a block from the pool
     * Returns nullptr if no blocks available
     * Wait-free operation
     */
    [[nodiscard]] void* allocate() noexcept {
        Block* old_head = free_list_.load(std::memory_order_acquire);
        
        while (old_head != nullptr) {
            Block* new_head = old_head->next.load(std::memory_order_relaxed);
            
            if (free_list_.compare_exchange_weak(old_head, new_head,
                                               std::memory_order_release,
                                               std::memory_order_acquire)) {
                allocated_count_.fetch_add(1, std::memory_order_relaxed);
                return old_head->data.data();
            }
        }
        
        return nullptr; // Pool exhausted
    }
    
    /**
     * Deallocate a block back to the pool
     * Must be a valid block from this pool
     * Wait-free operation
     */
    void deallocate(void* ptr) noexcept {
        if (ptr == nullptr) return;
        
        // Calculate block from pointer
        const uintptr_t pool_start = reinterpret_cast<uintptr_t>(blocks_.data());
        const uintptr_t ptr_addr = reinterpret_cast<uintptr_t>(ptr);
        const uintptr_t offset = ptr_addr - pool_start;
        const size_t block_index = offset / sizeof(Block);
        
        if (block_index >= NumBlocks) {
            // Invalid pointer, should not happen in correct usage
            return;
        }
        
        Block* block = &blocks_[block_index];
        Block* old_head = free_list_.load(std::memory_order_acquire);
        
        do {
            block->next.store(old_head, std::memory_order_relaxed);
        } while (!free_list_.compare_exchange_weak(old_head, block,
                                                 std::memory_order_release,
                                                 std::memory_order_acquire));
        
        allocated_count_.fetch_sub(1, std::memory_order_relaxed);
    }
    
    /**
     * Get number of allocated blocks
     */
    [[nodiscard]] size_t allocated_count() const noexcept {
        return allocated_count_.load(std::memory_order_acquire);
    }
    
    /**
     * Get total capacity
     */
    [[nodiscard]] static constexpr size_t capacity() noexcept {
        return NumBlocks;
    }
    
    /**
     * Get block size
     */
    [[nodiscard]] static constexpr size_t block_size() noexcept {
        return BlockSize;
    }
    
    /**
     * Check if pointer belongs to this pool
     */
    [[nodiscard]] bool owns(const void* ptr) const noexcept {
        const uintptr_t pool_start = reinterpret_cast<uintptr_t>(blocks_.data());
        const uintptr_t pool_end = pool_start + sizeof(blocks_);
        const uintptr_t ptr_addr = reinterpret_cast<uintptr_t>(ptr);
        
        return ptr_addr >= pool_start && ptr_addr < pool_end;
    }
};

/**
 * RAII wrapper for memory pool allocation
 */
template<typename Pool>
class PoolAllocator {
private:
    Pool& pool_;
    void* ptr_;
    
public:
    explicit PoolAllocator(Pool& pool) 
        : pool_(pool), ptr_(pool.allocate()) {}
    
    ~PoolAllocator() {
        if (ptr_) {
            pool_.deallocate(ptr_);
        }
    }
    
    // Non-copyable, movable
    PoolAllocator(const PoolAllocator&) = delete;
    PoolAllocator& operator=(const PoolAllocator&) = delete;
    
    PoolAllocator(PoolAllocator&& other) noexcept 
        : pool_(other.pool_), ptr_(other.ptr_) {
        other.ptr_ = nullptr;
    }
    
    PoolAllocator& operator=(PoolAllocator&& other) noexcept {
        if (this != &other) {
            if (ptr_) {
                pool_.deallocate(ptr_);
            }
            ptr_ = other.ptr_;
            other.ptr_ = nullptr;
        }
        return *this;
    }
    
    [[nodiscard]] void* get() const noexcept { return ptr_; }
    [[nodiscard]] bool valid() const noexcept { return ptr_ != nullptr; }
    
    template<typename T>
    [[nodiscard]] T* as() const noexcept {
        return static_cast<T*>(ptr_);
    }
};

} // namespace realtime_ml