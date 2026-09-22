#include "operators/gelu.hpp"

#include <cassert>
#include <cmath>
#include <iostream>

int main()
{
    Tensor input({5});

    input({0}) = -2.0f;
    input({1}) = -1.0f;
    input({2}) = 0.0f;
    input({3}) = 1.0f;
    input({4}) = 2.0f;

    Tensor output = gelu(input);

    assert(output.shape() ==
           std::vector<size_t>({5}));

    // Expected values using the tanh approximation.

    assert(std::abs(output({0}) - (-0.0454023f)) < 1e-4f);
    assert(std::abs(output({1}) - (-0.158808f)) < 1e-4f);
    assert(std::abs(output({2}) - 0.0f) < 1e-6f);
    assert(std::abs(output({3}) - 0.841192f) < 1e-4f);
    assert(std::abs(output({4}) - 1.95460f) < 1e-4f);

    std::cout << "GELU test passed!" << std::endl;

    return 0;
}
