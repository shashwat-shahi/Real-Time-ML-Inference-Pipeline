#include "inference_pipeline.hpp"
#include <chrono>
#include <algorithm>

namespace realtime_ml {

RealtimeInferencePipeline::RealtimeInferencePipeline(const std::vector<int>& cpu_affinity)
    : cpu_affinity_(cpu_affinity) {
    
    if (cpu_affinity_.empty()) {
        cpu_affinity_ = {0, 1, 2, 3}; // Default CPU affinity
    }
}

RealtimeInferencePipeline::~RealtimeInferencePipeline() {
    stop();
}

bool RealtimeInferencePipeline::initialize_cv_pipeline(int input_height, int input_width, 
                                                      int channels, int num_classes, 
                                                      int batch_size) {
    if (initialized_) {
        return false; // Already initialized
    }
    
    // Create inference engine
    inference_engine_ = InferenceEngineFactory::create_cv_engine(
        input_height, input_width, channels, num_classes, batch_size);
    
    if (!inference_engine_) {
        return false;
    }
    
    // Create scheduler
    scheduler_ = std::make_unique<RealtimeScheduler>(cpu_affinity_, 99);
    
    initialized_ = true;
    return true;
}

bool RealtimeInferencePipeline::initialize_custom_pipeline(const TensorDescriptor& input_desc,
                                                          const TensorDescriptor& output_desc,
                                                          int batch_size) {
    if (initialized_) {
        return false; // Already initialized
    }
    
    // Create inference engine
    inference_engine_ = InferenceEngineFactory::create_custom_engine(
        input_desc, output_desc, batch_size);
    
    if (!inference_engine_) {
        return false;
    }
    
    // Create scheduler
    scheduler_ = std::make_unique<RealtimeScheduler>(cpu_affinity_, 99);
    
    initialized_ = true;
    return true;
}

bool RealtimeInferencePipeline::start() {
    if (!initialized_ || !scheduler_ || !inference_engine_) {
        return false;
    }
    
    // Start the scheduler
    scheduler_->start();
    
    return true;
}

void RealtimeInferencePipeline::stop() {
    if (scheduler_) {
        scheduler_->stop();
    }
}

std::future<bool> RealtimeInferencePipeline::submit_inference(const void* input_data, 
                                                             size_t input_size,
                                                             void* output_data, 
                                                             size_t output_size,
                                                             uint64_t deadline_us) {
    
    if (!initialized_ || !validate_request_sizes(input_size, output_size)) {
        std::promise<bool> promise;
        auto future = promise.get_future();
        promise.set_value(false);
        return future;
    }
    
    // Create inference request
    const uint64_t request_id = next_request_id_.fetch_add(1);
    auto request = std::make_shared<InferenceRequest>(
        request_id, input_data, input_size, output_data, output_size, deadline_us);
    
    auto future = request->completion_promise.get_future();
    
    // Submit to scheduler
    const bool submitted = scheduler_->submit_inference_task(
        [this, request]() {
            process_inference_request_shared(request);
        }, deadline_us);
    
    if (!submitted) {
        std::promise<bool> promise;
        auto failed_future = promise.get_future();
        promise.set_value(false);
        return failed_future;
    }
    
    total_requests_.fetch_add(1);
    return future;
}

std::future<std::pair<bool, std::unique_ptr<uint8_t[]>>> 
RealtimeInferencePipeline::submit_inference_managed(const void* input_data, 
                                                   size_t input_size,
                                                   size_t expected_output_size, 
                                                   uint64_t deadline_us) {
    
    std::promise<std::pair<bool, std::unique_ptr<uint8_t[]>>> promise;
    auto future = promise.get_future();
    
    if (!initialized_) {
        promise.set_value({false, nullptr});
        return future;
    }
    
    // Allocate output buffer
    auto output_buffer = std::make_unique<uint8_t[]>(expected_output_size);
    
    // Submit inference
    auto inference_future = submit_inference(input_data, input_size, 
                                            output_buffer.get(), expected_output_size, 
                                            deadline_us);
    
    // Create a task to handle the result
    auto* future_ptr = new std::future<bool>(std::move(inference_future));
    auto* output_ptr = output_buffer.release();
    auto* promise_ptr = new std::promise<std::pair<bool, std::unique_ptr<uint8_t[]>>>(std::move(promise));
    
    scheduler_->submit_task([future_ptr, output_ptr, promise_ptr]() {
        const bool success = future_ptr->get();
        promise_ptr->set_value({success, std::unique_ptr<uint8_t[]>(output_ptr)});
        delete future_ptr;
        delete promise_ptr;
    }, 6); // High priority for result handling
    
    return future;
}

RealtimeInferencePipeline::PipelineStats RealtimeInferencePipeline::get_stats() const noexcept {
    PipelineStats stats;
    
    stats.avg_latency_us = pipeline_stats_.average_latency();
    stats.p95_latency_us = pipeline_stats_.p95_latency();
    stats.p99_latency_us = pipeline_stats_.p99_latency();
    stats.min_latency_us = pipeline_stats_.min_latency();
    stats.max_latency_us = pipeline_stats_.max_latency();
    stats.total_requests = total_requests_.load();
    stats.successful_requests = successful_requests_.load();
    stats.deadline_violations = deadline_violations_.load();
    
    stats.success_rate = stats.total_requests > 0 ? 
        static_cast<double>(stats.successful_requests) / stats.total_requests : 0.0;
    stats.deadline_miss_rate = stats.total_requests > 0 ? 
        static_cast<double>(stats.deadline_violations) / stats.total_requests : 0.0;
    
    return stats;
}

void RealtimeInferencePipeline::reset_stats() noexcept {
    pipeline_stats_.reset();
    inference_stats_.reset();
    total_requests_.store(0);
    successful_requests_.store(0);
    deadline_violations_.store(0);
}

void RealtimeInferencePipeline::process_inference_request(std::unique_ptr<InferenceRequest> request) {
    const uint64_t start_time = current_time_us();
    ScopedLatencyMeasure pipeline_measure(pipeline_stats_);
    
    bool success = false;
    
    // Check deadline before processing
    if (request->deadline_us > 0 && start_time > request->deadline_us) {
        deadline_violations_.fetch_add(1);
    } else {
        // Run inference
        ScopedLatencyMeasure inference_measure(inference_stats_);
        const uint64_t inference_time = inference_engine_->run_inference(
            request->input_data, request->output_data);
        
        success = (inference_time > 0);
        if (success) {
            successful_requests_.fetch_add(1);
        }
    }
    
    // Complete the request
    request->completion_promise.set_value(success);
}

void RealtimeInferencePipeline::process_inference_request_shared(std::shared_ptr<InferenceRequest> request) {
    const uint64_t start_time = current_time_us();
    ScopedLatencyMeasure pipeline_measure(pipeline_stats_);
    
    bool success = false;
    
    // Check deadline before processing
    if (request->deadline_us > 0 && start_time > request->deadline_us) {
        deadline_violations_.fetch_add(1);
    } else {
        // Run inference
        ScopedLatencyMeasure inference_measure(inference_stats_);
        const uint64_t inference_time = inference_engine_->run_inference(
            request->input_data, request->output_data);
        
        success = (inference_time > 0);
        if (success) {
            successful_requests_.fetch_add(1);
        }
    }
    
    // Complete the request
    request->completion_promise.set_value(success);
}

bool RealtimeInferencePipeline::validate_request_sizes(size_t input_size, size_t output_size) const noexcept {
    if (!inference_engine_) {
        return false;
    }
    
    const size_t expected_input = inference_engine_->input_descriptor().total_bytes * 
                                 inference_engine_->batch_size();
    const size_t expected_output = inference_engine_->output_descriptor().total_bytes * 
                                  inference_engine_->batch_size();
    
    return (input_size == expected_input && output_size == expected_output);
}

bool RealtimeInferencePipeline::can_meet_deadline(uint64_t deadline_us) const noexcept {
    // Simple heuristic: check if average latency + 2 std deviations < deadline
    const double avg_latency = pipeline_stats_.average_latency();
    const uint64_t p95_latency = pipeline_stats_.p95_latency();
    
    return (p95_latency < deadline_us);
}

uint64_t RealtimeInferencePipeline::current_time_us() const noexcept {
    const auto now = std::chrono::high_resolution_clock::now();
    const auto duration = now.time_since_epoch();
    return std::chrono::duration_cast<std::chrono::microseconds>(duration).count();
}

// PipelineConfigHelper implementation

std::vector<int> PipelineConfigHelper::get_recommended_cpu_affinity() {
    // For most systems, use the first 4 cores for optimal performance
    // In production, this should be configured based on system topology
    return {0, 1, 2, 3};
}

bool PipelineConfigHelper::configure_system_for_realtime() {
    // This would configure system-level settings for real-time operation
    // For now, just return true as system configuration requires root access
    return true;
}

std::pair<size_t, size_t> PipelineConfigHelper::get_optimal_pool_sizes(
    const TensorDescriptor& input_desc,
    const TensorDescriptor& output_desc,
    int batch_size) {
    
    // Calculate optimal pool sizes based on tensor sizes
    const size_t input_size = input_desc.total_bytes * batch_size;
    const size_t output_size = output_desc.total_bytes * batch_size;
    
    // Add some overhead for alignment and multiple requests
    const size_t optimal_input = ((input_size + 4095) / 4096) * 4096; // 4KB aligned
    const size_t optimal_output = ((output_size + 4095) / 4096) * 4096; // 4KB aligned
    
    return {optimal_input, optimal_output};
}

void PipelineConfigHelper::warmup_pipeline(RealtimeInferencePipeline& pipeline,
                                          int num_warmup_runs) {
    if (!pipeline.is_initialized()) {
        return;
    }
    
    const auto* engine = pipeline.get_inference_engine();
    if (!engine) {
        return;
    }
    
    // Create dummy input/output buffers
    const size_t input_size = engine->input_descriptor().total_bytes * engine->batch_size();
    const size_t output_size = engine->output_descriptor().total_bytes * engine->batch_size();
    
    auto input_buffer = std::make_unique<uint8_t[]>(input_size);
    auto output_buffer = std::make_unique<uint8_t[]>(output_size);
    
    // Fill input with dummy data
    std::fill_n(input_buffer.get(), input_size, 0x42);
    
    // Run warmup iterations
    for (int i = 0; i < num_warmup_runs; ++i) {
        auto future = pipeline.submit_inference(input_buffer.get(), input_size,
                                               output_buffer.get(), output_size, 1000);
        future.wait(); // Wait for completion
    }
}

} // namespace realtime_ml