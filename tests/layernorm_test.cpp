#include "operators/layernorm.hpp"

#include <cassert>
#include <cmath>
#include <iostream>

int main()
{
    Tensor input({3});
    Tensor gamma({3});
    Tensor beta({3});

    input({0}) = 1.0f;
    input({1}) = 2.0f;
    input({2}) = 3.0f;

    // Scale = 1
    gamma({0}) = 1.0f;
    gamma({1}) = 1.0f;
    gamma({2}) = 1.0f;

    // Bias = 0
    beta({0}) = 0.0f;
    beta({1}) = 0.0f;
    beta({2}) = 0.0f;

    Tensor output = layernorm(
        input,
        gamma,
        beta
    );

    assert(output.shape() ==
           std::vector<size_t>({3}));

    // Mean should be approximately zero.
    float mean = 0.0f;

    for (size_t i = 0; i < output.size(); ++i)
    {
        mean += output.data()[i];
    }

    mean /= static_cast<float>(output.size());

    assert(std::abs(mean) < 1e-5f);

    // Variance should be approximately one.
    float variance = 0.0f;

    for (size_t i = 0; i < output.size(); ++i)
    {
        variance +=
            output.data()[i] * output.data()[i];
    }

    variance /= static_cast<float>(output.size());

    assert(std::abs(variance - 1.0f) < 1e-4f);

    std::cout << "LayerNorm test passed!" << std::endl;

    return 0;
}
