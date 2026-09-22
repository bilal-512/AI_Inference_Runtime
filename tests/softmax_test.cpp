#include "operators/softmax.hpp"

#include <cassert>
#include <cmath>
#include <iostream>

int main()
{
    Tensor input({3});

    input({0}) = 1.0f;
    input({1}) = 2.0f;
    input({2}) = 3.0f;

    Tensor output = softmax(input);

    assert(output.shape() == std::vector<size_t>({3}));

    // Expected approximately:
    //
    // [0.0900, 0.2447, 0.6652]

    assert(std::abs(output({0}) - 0.0900306f) < 1e-5f);
    assert(std::abs(output({1}) - 0.2447285f) < 1e-5f);
    assert(std::abs(output({2}) - 0.6652409f) < 1e-5f);

    // Every value must be between 0 and 1.
    for (size_t i = 0; i < output.size(); ++i)
    {
        assert(output.data()[i] >= 0.0f);
        assert(output.data()[i] <= 1.0f);
    }

    // Sum should be approximately 1.
    float sum = 0.0f;

    for (size_t i = 0; i < output.size(); ++i)
    {
        sum += output.data()[i];
    }

    assert(std::abs(sum - 1.0f) < 1e-5f);

    std::cout << "Softmax test passed!" << std::endl;

    return 0;
}
