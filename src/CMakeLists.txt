# Source directory CMakeLists.txt

# Create source files list
set(CORE_SOURCES
    timing.cpp
    scheduler.cpp
    inference_pipeline.cpp
)

# Add CUDA source if available
if(CUDA_FOUND)
    list(APPEND CORE_SOURCES cuda_inference.cu)
else()
    list(APPEND CORE_SOURCES cuda_inference_cpu.cpp)
endif()

add_library(realtime_ml_core STATIC ${CORE_SOURCES})

target_include_directories(realtime_ml_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
)

target_link_libraries(realtime_ml_core ${CMAKE_THREAD_LIBS_INIT})

if(CUDA_FOUND)
    target_link_libraries(realtime_ml_core ${CUDA_LIBRARIES})
    # Set CUDA properties
    set_property(TARGET realtime_ml_core PROPERTY CUDA_STANDARD 17)
    set_property(TARGET realtime_ml_core PROPERTY CUDA_STANDARD_REQUIRED ON)
    set_property(TARGET realtime_ml_core PROPERTY CUDA_SEPARABLE_COMPILATION ON)
endif()