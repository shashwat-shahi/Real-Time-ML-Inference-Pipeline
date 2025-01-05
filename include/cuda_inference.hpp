#pragma once

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

#include <memory>
#include <vector>
#include <cstdint>
#include <chrono>

namespace realtime_ml {

#ifdef USE_CUDA
/**
 * CUDA error checking macro
 */
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            return false; \
        } \
    } while(0)
#else
#define CUDA_CHECK(call) (true)
#endif

/**
 * Memory management for inference (CUDA or CPU)
 */
class CudaMemoryManager {
private:
    void* device_memory_;
    size_t memory_size_;
#ifdef USE_CUDA
    cudaStream_t stream_;
#endif
    
public:
    explicit CudaMemoryManager(size_t size);
    ~CudaMemoryManager();
    
    // Non-copyable, movable
    CudaMemoryManager(const CudaMemoryManager&) = delete;
    CudaMemoryManager& operator=(const CudaMemoryManager&) = delete;
    CudaMemoryManager(CudaMemoryManager&& other) noexcept;
    CudaMemoryManager& operator=(CudaMemoryManager&& other) noexcept;
    
    /**
     * Get device memory pointer
     */
    void* device_ptr() const noexcept { return device_memory_; }
    
#ifdef USE_CUDA
    /**
     * Get CUDA stream
     */
    cudaStream_t stream() const noexcept { return stream_; }
#endif
    
    /**
     * Copy data from host to device asynchronously
     */
    bool copy_to_device_async(const void* host_data, size_t size, size_t offset = 0) noexcept;
    
    /**
     * Copy data from device to host asynchronously
     */
    bool copy_to_host_async(void* host_data, size_t size, size_t offset = 0) noexcept;
    
    /**
     * Synchronize stream
     */
    bool synchronize() noexcept;
    
    /**
     * Get memory size
     */
    size_t size() const noexcept { return memory_size_; }
};

/**
 * Tensor descriptor for input/output data
 */
struct TensorDescriptor {
    std::vector<int> dimensions;
    size_t element_size;
    size_t total_elements;
    size_t total_bytes;
    
    TensorDescriptor(const std::vector<int>& dims, size_t elem_size);
    
    /**
     * Get offset for multi-dimensional indexing
     */
    size_t get_offset(const std::vector<int>& indices) const noexcept;
};

/**
 * High-performance inference engine for computer vision models
 * Supports both CUDA GPU acceleration and CPU fallback
 */
class CudaInferenceEngine {
private:
    std::unique_ptr<CudaMemoryManager> input_memory_;
    std::unique_ptr<CudaMemoryManager> output_memory_;
    std::unique_ptr<CudaMemoryManager> workspace_memory_;
    
    TensorDescriptor input_descriptor_;
    TensorDescriptor output_descriptor_;
    
    // Model-specific parameters
    int batch_size_;
    bool initialized_;
    
#ifdef USE_CUDA
    // Performance monitoring
    mutable cudaEvent_t start_event_;
    mutable cudaEvent_t end_event_;
#endif
    
public:
    CudaInferenceEngine(const TensorDescriptor& input_desc,
                       const TensorDescriptor& output_desc,
                       int batch_size = 1);
    ~CudaInferenceEngine();
    
    // Non-copyable, non-movable
    CudaInferenceEngine(const CudaInferenceEngine&) = delete;
    CudaInferenceEngine& operator=(const CudaInferenceEngine&) = delete;
    CudaInferenceEngine(CudaInferenceEngine&&) = delete;
    CudaInferenceEngine& operator=(CudaInferenceEngine&&) = delete;
    
    /**
     * Initialize the inference engine
     */
    bool initialize();
    
    /**
     * Run inference on input data
     * Returns execution time in microseconds
     */
    uint64_t run_inference(const void* input_data, void* output_data) noexcept;
    
    /**
     * Run inference asynchronously
     * Call synchronize() to wait for completion
     */
    bool run_inference_async(const void* input_data, void* output_data) noexcept;
    
    /**
     * Synchronize with GPU execution
     */
    bool synchronize() noexcept;
    
    /**
     * Get input tensor descriptor
     */
    const TensorDescriptor& input_descriptor() const noexcept {
        return input_descriptor_;
    }
    
    /**
     * Get output tensor descriptor
     */
    const TensorDescriptor& output_descriptor() const noexcept {
        return output_descriptor_;
    }
    
    /**
     * Get batch size
     */
    int batch_size() const noexcept { return batch_size_; }
    
    /**
     * Check if engine is initialized
     */
    bool is_initialized() const noexcept { return initialized_; }

private:
    /**
     * Execute computer vision model (CUDA or CPU implementation)
     */
    bool execute_cv_model(const void* input, void* output) noexcept;
    
    /**
     * Calculate required memory sizes
     */
    size_t calculate_workspace_size() const noexcept;
};

/**
 * Factory for creating optimized inference engines
 */
class InferenceEngineFactory {
public:
    /**
     * Create a computer vision inference engine
     * For common CV models like ResNet, EfficientNet, etc.
     */
    static std::unique_ptr<CudaInferenceEngine> 
    create_cv_engine(int input_height, int input_width, int channels,
                    int num_classes, int batch_size = 1);
    
    /**
     * Create a custom inference engine
     */
    static std::unique_ptr<CudaInferenceEngine>
    create_custom_engine(const TensorDescriptor& input_desc,
                        const TensorDescriptor& output_desc,
                        int batch_size = 1);
};

} // namespace realtime_ml