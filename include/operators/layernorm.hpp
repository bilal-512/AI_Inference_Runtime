#pragma once

#include "tensor.hpp"

Tensor layernorm(
    const Tensor& input,
    const Tensor& gamma,
    const Tensor& beta,
    float epsilon = 1e-5f
);
