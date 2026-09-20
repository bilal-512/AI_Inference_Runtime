#include "operators/add.hpp"

#include <stdexcept>

Tensor add(const Tensor& a, const Tensor& b) {

    // Shapes must match.
    if (a.shape() != b.shape()) {
        throw std::invalid_argument(
            "Tensor shapes must match for addition"
        );
    }

    // Create output tensor with the same shape.
    Tensor result(a.shape());

    // Element-wise addition.
    for (size_t i = 0; i < a.size(); ++i) {
        result.data()[i] = a.data()[i] + b.data()[i];
    }

    return result;
}
