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
    src/operators/add.cpp
    src/operators/relu.cpp
    src/operators/matmul.cpp
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

add_executable(tensor_test
    tests/tensor_test.cpp
)

target_link_libraries(tensor_test
    PRIVATE
        runtime_core
)

add_executable(add_test
    tests/add_test.cpp
)

target_link_libraries(add_test
    PRIVATE
        runtime_core
)

add_executable(relu_test
    tests/relu_test.cpp
)

target_link_libraries(relu_test
    PRIVATE
        runtime_core
)


add_executable(matmul_test
    tests/matmul_test.cpp
)

target_link_libraries(matmul_test
    PRIVATE
        runtime_core
)

