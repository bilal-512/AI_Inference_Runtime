#include "operators/softmax.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

Tensor softmax(const Tensor& input)
{
    if (input.ndim() != 1)
    {
        throw std::invalid_argument(
            "softmax currently supports 1D tensors only");
    }

    const size_t n = input.size();

    Tensor output({n});

    // Find the maximum value for numerical stability.
    float max_value = input.data()[0];

    for (size_t i = 1; i < n; ++i)
    {
        max_value = std::max(max_value, input.data()[i]);
    }

    // Compute exp(x - max).
    float sum = 0.0f;

    for (size_t i = 0; i < n; ++i)
    {
        output.data()[i] =
            std::exp(input.data()[i] - max_value);

        sum += output.data()[i];
    }

    // Normalize so all values sum to 1.
    for (size_t i = 0; i < n; ++i)
    {
        output.data()[i] /= sum;
    }

    return output;
}
