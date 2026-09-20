cmake_minimum_required(VERSION 3.16)

project(AIInferenceRuntime
    VERSION 0.1.0
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(runtime_core
    src/tensor.cpp
)

target_include_directories(runtime_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_executable(runtime
    src/main.cpp
)

target_link_libraries(runtime
    PRIVATE
        runtime_core
)

add_executable(tensor_example
    examples/tensor_example.cpp
)

target_link_libraries(tensor_example
    PRIVATE
        runtime_core
)
