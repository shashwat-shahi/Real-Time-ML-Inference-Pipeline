# Tests CMakeLists.txt
add_executable(test_lockfree_queue test_lockfree_queue.cpp)
target_link_libraries(test_lockfree_queue realtime_ml_core ${CMAKE_THREAD_LIBS_INIT})

add_executable(test_memory_pool test_memory_pool.cpp)
target_link_libraries(test_memory_pool realtime_ml_core ${CMAKE_THREAD_LIBS_INIT})

# Only build CUDA-dependent tests if CUDA is available
if(CUDA_FOUND)
    add_executable(test_scheduler test_scheduler.cpp)
    target_link_libraries(test_scheduler realtime_ml_core ${CMAKE_THREAD_LIBS_INIT} ${CUDA_LIBRARIES})
    
    add_executable(test_pipeline test_pipeline.cpp)
    target_link_libraries(test_pipeline realtime_ml_core ${CMAKE_THREAD_LIBS_INIT} ${CUDA_LIBRARIES})
else()
    add_executable(test_scheduler test_scheduler.cpp)
    target_link_libraries(test_scheduler realtime_ml_core ${CMAKE_THREAD_LIBS_INIT})
    
    add_executable(test_pipeline test_pipeline.cpp)
    target_link_libraries(test_pipeline realtime_ml_core ${CMAKE_THREAD_LIBS_INIT})
endif()

# Enable testing
enable_testing()

add_test(NAME LockFreeQueueTest COMMAND test_lockfree_queue)
add_test(NAME MemoryPoolTest COMMAND test_memory_pool)
add_test(NAME SchedulerTest COMMAND test_scheduler)
add_test(NAME PipelineTest COMMAND test_pipeline)