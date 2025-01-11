#include "inference_pipeline.hpp"
#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <iomanip>

using namespace realtime_ml;

struct LatencyMeasurement {
    uint64_t latency_us;
    bool success;
    bool deadline_met;
};

std::vector<LatencyMeasurement> run_latency_test(RealtimeInferencePipeline& pipeline,
                                               const void* input_data, size_t input_size,
                                               void* output_data, size_t output_size,
                                               uint64_t deadline_us, int num_requests) {
    
    std::vector<LatencyMeasurement> measurements;
    measurements.reserve(num_requests);
    
    for (int i = 0; i < num_requests; ++i) {
        const auto start = std::chrono::high_resolution_clock::now();
        
        auto future = pipeline.submit_inference(input_data, input_size,
                                               output_data, output_size, deadline_us);
        
        const bool success = future.get();
        
        const auto end = std::chrono::high_resolution_clock::now();
        const auto latency_us = std::chrono::duration_cast<std::chrono::microseconds>(
            end - start).count();
        
        measurements.push_back({
            static_cast<uint64_t>(latency_us),
            success,
            static_cast<uint64_t>(latency_us) <= deadline_us
        });
        
        // Small delay to avoid overwhelming the system
        if (i % 100 == 0) {
            std::this_thread::sleep_for(std::chrono::microseconds(10));
        }
    }
    
    return measurements;
}

void analyze_latency_results(const std::vector<LatencyMeasurement>& measurements,
                           uint64_t target_latency_us) {
    
    if (measurements.empty()) {
        std::cout << "No measurements to analyze\n";
        return;
    }
    
    // Calculate statistics
    std::vector<uint64_t> latencies;
    size_t successful = 0;
    size_t deadline_met = 0;
    
    for (const auto& measurement : measurements) {
        latencies.push_back(measurement.latency_us);
        if (measurement.success) successful++;
        if (measurement.deadline_met) deadline_met++;
    }
    
    std::sort(latencies.begin(), latencies.end());
    
    const uint64_t min_latency = latencies.front();
    const uint64_t max_latency = latencies.back();
    const uint64_t median_latency = latencies[latencies.size() / 2];
    const uint64_t p95_latency = latencies[static_cast<size_t>(latencies.size() * 0.95)];
    const uint64_t p99_latency = latencies[static_cast<size_t>(latencies.size() * 0.99)];
    const uint64_t p999_latency = latencies[static_cast<size_t>(latencies.size() * 0.999)];
    
    const double avg_latency = std::accumulate(latencies.begin(), latencies.end(), 0.0) / latencies.size();
    const double success_rate = static_cast<double>(successful) / measurements.size() * 100.0;
    const double deadline_rate = static_cast<double>(deadline_met) / measurements.size() * 100.0;
    
    std::cout << "\n=== Latency Analysis Results ===\n";
    std::cout << "Total Requests: " << measurements.size() << "\n";
    std::cout << "Target Latency: " << target_latency_us << " μs\n";
    std::cout << "Success Rate: " << std::fixed << std::setprecision(2) << success_rate << "%\n";
    std::cout << "Deadline Met Rate: " << std::fixed << std::setprecision(2) << deadline_rate << "%\n\n";
    
    std::cout << "Latency Statistics (μs):\n";
    std::cout << "  Minimum:  " << std::setw(8) << min_latency << "\n";
    std::cout << "  Average:  " << std::setw(8) << std::fixed << std::setprecision(1) << avg_latency << "\n";
    std::cout << "  Median:   " << std::setw(8) << median_latency << "\n";
    std::cout << "  P95:      " << std::setw(8) << p95_latency << "\n";
    std::cout << "  P99:      " << std::setw(8) << p99_latency << "\n";
    std::cout << "  P99.9:    " << std::setw(8) << p999_latency << "\n";
    std::cout << "  Maximum:  " << std::setw(8) << max_latency << "\n\n";
    
    // Performance assessment
    const bool meets_p99_target = p99_latency <= target_latency_us;
    const bool meets_success_target = success_rate >= 99.0;
    const bool meets_deadline_target = deadline_rate >= 95.0;
    
    std::cout << "Performance Goals:\n";
    std::cout << "  P99 < " << target_latency_us << " μs:     " 
              << (meets_p99_target ? "✓ PASS" : "✗ FAIL") 
              << " (" << p99_latency << " μs)\n";
    std::cout << "  Success Rate ≥ 99%:   " 
              << (meets_success_target ? "✓ PASS" : "✗ FAIL") 
              << " (" << std::setprecision(2) << success_rate << "%)\n";
    std::cout << "  Deadline Met ≥ 95%:   " 
              << (meets_deadline_target ? "✓ PASS" : "✗ FAIL") 
              << " (" << std::setprecision(2) << deadline_rate << "%)\n";
    
    std::cout << "==============================\n\n";
}

int main() {
    std::cout << "Real-Time ML Inference Latency Benchmark\n";
    std::cout << "========================================\n\n";
    
    // Test configurations
    struct TestConfig {
        const char* name;
        int height, width, channels, classes;
        uint64_t target_latency_us;
        int num_requests;
    };
    
    std::vector<TestConfig> configs = {
        {"Small Image Classification", 224, 224, 3, 1000, 100, 1000},
        {"High Resolution", 512, 512, 3, 1000, 200, 500},
        {"Batch Processing", 224, 224, 3, 1000, 300, 100},
    };
    
    for (const auto& config : configs) {
        std::cout << "Testing: " << config.name << "\n";
        std::cout << "Image: " << config.width << "x" << config.height << "x" << config.channels << "\n";
        std::cout << "Classes: " << config.classes << "\n";
        std::cout << "Target: " << config.target_latency_us << " μs\n";
        std::cout << "Requests: " << config.num_requests << "\n\n";
        
        // Initialize pipeline
        RealtimeInferencePipeline pipeline({0, 1, 2, 3});
        
        if (!pipeline.initialize_cv_pipeline(config.height, config.width, 
                                            config.channels, config.classes, 1)) {
            std::cerr << "Failed to initialize pipeline for " << config.name << "\n";
            continue;
        }
        
        if (!pipeline.start()) {
            std::cerr << "Failed to start pipeline for " << config.name << "\n";
            continue;
        }
        
        // Warmup
        std::cout << "Warming up...\n";
        PipelineConfigHelper::warmup_pipeline(pipeline, 100);
        pipeline.reset_stats();
        
        // Prepare test data
        const size_t input_size = config.height * config.width * config.channels * sizeof(float);
        const size_t output_size = config.classes * sizeof(float);
        
        auto input_data = std::make_unique<float[]>(config.height * config.width * config.channels);
        auto output_data = std::make_unique<float[]>(config.classes);
        
        // Generate random input
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(0.0f, 255.0f);
        
        for (int i = 0; i < config.height * config.width * config.channels; ++i) {
            input_data[i] = dis(gen);
        }
        
        // Run benchmark
        std::cout << "Running latency benchmark...\n";
        auto measurements = run_latency_test(pipeline, input_data.get(), input_size,
                                           output_data.get(), output_size,
                                           config.target_latency_us, config.num_requests);
        
        // Analyze results
        analyze_latency_results(measurements, config.target_latency_us);
        
        pipeline.stop();
        
        std::cout << "----------------------------------------\n\n";
    }
    
    std::cout << "Latency benchmark completed!\n";
    return 0;
}