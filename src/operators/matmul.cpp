#include "operators/matmul.hpp"

#include <stdexcept>

Tensor matmul(const Tensor& a, const Tensor& b)
{
    // MatMul currently supports 2D tensors only.
    if (a.ndim() != 2 || b.ndim() != 2)
    {
        throw std::invalid_argument(
            "matmul currently supports 2D tensors only");
    }

    const size_t M = a.shape()[0];
    const size_t K = a.shape()[1];

    const size_t B_rows = b.shape()[0];
    const size_t N = b.shape()[1];

    // Inner dimensions must match.
    if (K != B_rows)
    {
        throw std::invalid_argument(
            "matmul shape mismatch");
    }

    Tensor output({M, N});

    for (size_t i = 0; i < M; ++i)
    {
        for (size_t j = 0; j < N; ++j)
        {
            float sum = 0.0f;

            for (size_t k = 0; k < K; ++k)
            {
                sum += a({i, k}) * b({k, j});
            }

            output({i, j}) = sum;
        }
    }

    return output;
}
