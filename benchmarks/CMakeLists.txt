# Benchmarks CMakeLists.txt
add_executable(latency_benchmark latency_benchmark.cpp)
target_link_libraries(latency_benchmark realtime_ml_core ${CMAKE_THREAD_LIBS_INIT})

if(CUDA_FOUND)
    target_link_libraries(latency_benchmark ${CUDA_LIBRARIES})
endif()

# Create placeholder benchmarks for now
add_executable(throughput_benchmark latency_benchmark.cpp)
target_link_libraries(throughput_benchmark realtime_ml_core ${CMAKE_THREAD_LIBS_INIT})

if(CUDA_FOUND)
    target_link_libraries(throughput_benchmark ${CUDA_LIBRARIES})
endif()

add_executable(memory_benchmark memory_benchmark.cpp)
target_link_libraries(memory_benchmark realtime_ml_core ${CMAKE_THREAD_LIBS_INIT})