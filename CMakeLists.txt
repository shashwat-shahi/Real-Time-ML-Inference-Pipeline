cmake_minimum_required(VERSION 3.18)
project(RealTimeMLInference LANGUAGES CXX)

# Set C++17 standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find required packages
find_package(Threads REQUIRED)

# Try to find CUDA (optional)
find_package(CUDA QUIET)
if(CUDA_FOUND)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    add_definitions(-DUSE_CUDA)
    message(STATUS "CUDA found - enabling GPU acceleration")
else()
    message(STATUS "CUDA not found - building CPU-only version")
endif()

# Compiler flags for performance and real-time guarantees
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -mtune=native")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNDEBUG")

# CUDA flags (if available)
if(CUDA_FOUND)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_75,code=sm_75")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_80,code=sm_80")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_86,code=sm_86")
endif()

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
if(CUDA_FOUND)
    include_directories(${CUDA_INCLUDE_DIRS})
endif()

# Add subdirectories
add_subdirectory(src)
add_subdirectory(tests)
add_subdirectory(benchmarks)

# Add examples
add_subdirectory(examples)

# Main executable
add_executable(realtime_inference
    src/main.cpp
)

target_link_libraries(realtime_inference
    realtime_ml_core
    ${CMAKE_THREAD_LIBS_INIT}
)

if(CUDA_FOUND)
    target_link_libraries(realtime_inference ${CUDA_LIBRARIES})
endif()

# Set properties for real-time execution
set_target_properties(realtime_inference PROPERTIES
    LINK_FLAGS "-Wl,--hash-style=gnu -Wl,--as-needed"
)