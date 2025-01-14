# Examples CMakeLists.txt
add_executable(component_demo component_demo.cpp)
target_link_libraries(component_demo realtime_ml_core ${CMAKE_THREAD_LIBS_INIT})