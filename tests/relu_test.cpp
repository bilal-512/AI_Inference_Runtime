#include <cassert>
#include <iostream>

#include "operators/relu.hpp"

int main() {

    // -------------------------
    // Create input tensor
    // -------------------------

    Tensor input({2, 3});

    input({0, 0}) = -2.0f;
    input({0, 1}) =  3.0f;
    input({0, 2}) = -4.0f;

    input({1, 0}) =  5.0f;
    input({1, 1}) = -1.0f;
    input({1, 2}) =  0.0f;

    // -------------------------
    // Apply ReLU
    // -------------------------

    Tensor result = relu(input);

    // -------------------------
    // Verify shape
    // -------------------------

    assert(result.shape() == std::vector<size_t>({2, 3}));

    // -------------------------
    // Verify values
    // -------------------------

    assert(result({0, 0}) == 0.0f);
    assert(result({0, 1}) == 3.0f);
    assert(result({0, 2}) == 0.0f);

    assert(result({1, 0}) == 5.0f);
    assert(result({1, 1}) == 0.0f);
    assert(result({1, 2}) == 0.0f);

    // -------------------------
    // Verify input was not modified
    // -------------------------

    assert(input({0, 0}) == -2.0f);
    assert(input({0, 1}) == 3.0f);
    assert(input({1, 0}) == 5.0f);

    std::cout << "All ReLU tests passed!\n";

    return 0;
}
