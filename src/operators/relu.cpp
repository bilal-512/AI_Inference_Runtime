#include "operators/relu.hpp"

Tensor relu(const Tensor& input) {

    Tensor result(input.shape());

    for (size_t i = 0; i < input.size(); ++i) {

        if (input.data()[i] > 0.0f) {
            result.data()[i] = input.data()[i];
        }
        else {
            result.data()[i] = 0.0f;
        }
    }

    return result;
}
