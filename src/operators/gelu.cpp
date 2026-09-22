#include "operators/gelu.hpp"

#include <cmath>

Tensor gelu(const Tensor& input)
{
    Tensor output(input.shape());

    constexpr float sqrt_2_over_pi =
        0.7978845608028654f;

    constexpr float coefficient =
        0.044715f;

    for (size_t i = 0; i < input.size(); ++i)
    {
        const float x = input.data()[i];

        const float inner =
            sqrt_2_over_pi *
            (x + coefficient * x * x * x);

        output.data()[i] =
            0.5f * x * (1.0f + std::tanh(inner));
    }

    return output;
}
