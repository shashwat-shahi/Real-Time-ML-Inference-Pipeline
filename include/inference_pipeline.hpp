#pragma once

#include "scheduler.hpp"
#include "cuda_inference.hpp"
#include "memory_pool.hpp"
#include "timing.hpp"
#include <future>
#include <functional>

namespace realtime_ml {

/**
 * Inference request structure
 */
struct InferenceRequest {
    uint64_t request_id;
    const void* input_data;
    size_t input_size;
    void* output_data;
    size_t output_size;
    uint64_t deadline_us;
    std::promise<bool> completion_promise;
    
    InferenceRequest(uint64_t id, const void* input, size_t input_sz,
                    void* output, size_t output_sz, uint64_t deadline)
        : request_id(id), input_data(input), input_size(input_sz),
          output_data(output), output_size(output_sz), deadline_us(deadline) {}
};

/**
 * Main real-time ML inference pipeline
 * Orchestrates scheduling, memory management, and CUDA execution
 */
class RealtimeInferencePipeline {
private:
    // Core components
    std::unique_ptr<RealtimeScheduler> scheduler_;
    std::unique_ptr<CudaInferenceEngine> inference_engine_;
    
    // Memory management
    static constexpr size_t INPUT_BUFFER_SIZE = 1024 * 1024;  // 1MB per buffer
    static constexpr size_t OUTPUT_BUFFER_SIZE = 1024 * 1024; // 1MB per buffer
    static constexpr size_t NUM_BUFFERS = 16;
    
    MemoryPool<INPUT_BUFFER_SIZE, NUM_BUFFERS> input_pool_;
    MemoryPool<OUTPUT_BUFFER_SIZE, NUM_BUFFERS> output_pool_;
    
    // Request management
    std::atomic<uint64_t> next_request_id_{1};
    LockFreeQueue<std::unique_ptr<InferenceRequest>, 256> request_queue_;
    
    // Performance monitoring
    LatencyStats pipeline_stats_;
    LatencyStats inference_stats_;
    std::atomic<uint64_t> total_requests_{0};
    std::atomic<uint64_t> successful_requests_{0};
    std::atomic<uint64_t> deadline_violations_{0};
    
    // Configuration
    bool initialized_{false};
    std::vector<int> cpu_affinity_;
    
public:
    explicit RealtimeInferencePipeline(const std::vector<int>& cpu_affinity = {0, 1, 2, 3});
    ~RealtimeInferencePipeline();
    
    // Non-copyable, non-movable
    RealtimeInferencePipeline(const RealtimeInferencePipeline&) = delete;
    RealtimeInferencePipeline& operator=(const RealtimeInferencePipeline&) = delete;
    RealtimeInferencePipeline(RealtimeInferencePipeline&&) = delete;
    RealtimeInferencePipeline& operator=(RealtimeInferencePipeline&&) = delete;
    
    /**
     * Initialize the pipeline with a computer vision model
     */
    bool initialize_cv_pipeline(int input_height, int input_width, int channels,
                               int num_classes, int batch_size = 1);
    
    /**
     * Initialize the pipeline with custom tensor descriptors
     */
    bool initialize_custom_pipeline(const TensorDescriptor& input_desc,
                                   const TensorDescriptor& output_desc,
                                   int batch_size = 1);
    
    /**
     * Start the pipeline
     */
    bool start();
    
    /**
     * Stop the pipeline
     */
    void stop();
    
    /**
     * Submit an inference request
     * Returns a future that will be set when inference completes
     * deadline_us: maximum acceptable latency in microseconds (default: 100μs)
     */
    std::future<bool> submit_inference(const void* input_data, size_t input_size,
                                      void* output_data, size_t output_size,
                                      uint64_t deadline_us = 100);
    
    /**
     * Submit inference with automatic memory management
     */
    std::future<std::pair<bool, std::unique_ptr<uint8_t[]>>> 
    submit_inference_managed(const void* input_data, size_t input_size,
                           size_t expected_output_size, uint64_t deadline_us = 100);
    
    /**
     * Get pipeline performance statistics
     */
    struct PipelineStats {
        double avg_latency_us;
        uint64_t p95_latency_us;
        uint64_t p99_latency_us;
        uint64_t min_latency_us;
        uint64_t max_latency_us;
        uint64_t total_requests;
        uint64_t successful_requests;
        uint64_t deadline_violations;
        double success_rate;
        double deadline_miss_rate;
    };
    
    PipelineStats get_stats() const noexcept;
    
    /**
     * Reset performance statistics
     */
    void reset_stats() noexcept;
    
    /**
     * Check if pipeline is initialized
     */
    bool is_initialized() const noexcept { return initialized_; }
    
    /**
     * Get inference engine details
     */
    const CudaInferenceEngine* get_inference_engine() const noexcept {
        return inference_engine_.get();
    }

private:
    /**
     * Process a single inference request
     */
    void process_inference_request(std::unique_ptr<InferenceRequest> request);
    
    /**
     * Process a single inference request (shared_ptr version)
     */
    void process_inference_request_shared(std::shared_ptr<InferenceRequest> request);
    
    /**
     * Main request processing loop
     */
    void request_processing_loop();
    
    /**
     * Validate input/output sizes
     */
    bool validate_request_sizes(size_t input_size, size_t output_size) const noexcept;
    
    /**
     * Check if deadline can be met
     */
    bool can_meet_deadline(uint64_t deadline_us) const noexcept;
    
    /**
     * Get current time in microseconds
     */
    uint64_t current_time_us() const noexcept;
};

/**
 * Configuration helper for optimal pipeline setup
 */
class PipelineConfigHelper {
public:
    /**
     * Get recommended CPU affinity for different system configurations
     */
    static std::vector<int> get_recommended_cpu_affinity();
    
    /**
     * Configure system for real-time operation
     * Sets CPU governor, disables power management, etc.
     */
    static bool configure_system_for_realtime();
    
    /**
     * Get optimal memory pool sizes based on model requirements
     */
    static std::pair<size_t, size_t> get_optimal_pool_sizes(
        const TensorDescriptor& input_desc,
        const TensorDescriptor& output_desc,
        int batch_size);
    
    /**
     * Warm up the pipeline for consistent performance
     */
    static void warmup_pipeline(RealtimeInferencePipeline& pipeline,
                               int num_warmup_runs = 100);
};

} // namespace realtime_ml