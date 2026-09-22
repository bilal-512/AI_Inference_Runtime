#include "operators/layernorm.hpp"

#include <cmath>
#include <stdexcept>

Tensor layernorm(
    const Tensor& input,
    const Tensor& gamma,
    const Tensor& beta,
    float epsilon)
{
    if (input.ndim() != 1 ||
        gamma.ndim() != 1 ||
        beta.ndim() != 1)
    {
        throw std::invalid_argument(
            "layernorm currently supports 1D tensors only");
    }

    if (input.size() != gamma.size() ||
        input.size() != beta.size())
    {
        throw std::invalid_argument(
            "layernorm input, gamma and beta must have the same size");
    }

    if (epsilon <= 0.0f)
    {
        throw std::invalid_argument(
            "layernorm epsilon must be positive");
    }

    const size_t n = input.size();

    Tensor output({n});

    // --------------------------------------------------
    // Step 1: Calculate mean
    // --------------------------------------------------

    float mean = 0.0f;

    for (size_t i = 0; i < n; ++i)
    {
        mean += input.data()[i];
    }

    mean /= static_cast<float>(n);

    // --------------------------------------------------
    // Step 2: Calculate variance
    // --------------------------------------------------

    float variance = 0.0f;

    for (size_t i = 0; i < n; ++i)
    {
        float difference = input.data()[i] - mean;

        variance += difference * difference;
    }

    variance /= static_cast<float>(n);

    // --------------------------------------------------
    // Step 3: Normalize, scale and shift
    // --------------------------------------------------

    const float denominator =
        std::sqrt(variance + epsilon);

    for (size_t i = 0; i < n; ++i)
    {
        float normalized =
            (input.data()[i] - mean) / denominator;

        output.data()[i] =
            gamma.data()[i] * normalized
            + beta.data()[i];
    }

    return output;
}
