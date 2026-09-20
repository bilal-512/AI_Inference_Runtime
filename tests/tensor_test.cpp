#include <cassert>
#include <iostream>
#include <stdexcept>

#include "tensor.hpp"

int main() {

    // -------------------------
    // Basic construction
    // -------------------------

    Tensor tensor({2, 3, 4});

    assert(tensor.ndim() == 3);
    assert(tensor.size() == 24);

    // -------------------------
    // Element access
    // -------------------------

    tensor({0, 0, 0}) = 10.0f;
    tensor({1, 2, 3}) = 20.0f;

    assert(tensor({0, 0, 0}) == 10.0f);
    assert(tensor({1, 2, 3}) == 20.0f);

    // -------------------------
    // Wrong number of indices
    // -------------------------

    bool caught_invalid_indices = false;

    try {
        tensor({0, 0});
    }
    catch (const std::invalid_argument&) {
        caught_invalid_indices = true;
    }

    assert(caught_invalid_indices);

    // -------------------------
    // Out-of-bounds access
    // -------------------------

    bool caught_out_of_bounds = false;

    try {
        tensor({2, 0, 0});
    }
    catch (const std::out_of_range&) {
        caught_out_of_bounds = true;
    }

    assert(caught_out_of_bounds);

    // -------------------------
    // Zero dimension
    // -------------------------

    bool caught_zero_dimension = false;

    try {
        Tensor invalid({2, 0, 3});
    }
    catch (const std::invalid_argument&) {
        caught_zero_dimension = true;
    }

    assert(caught_zero_dimension);

    std::cout << "All tensor tests passed!\n";

    return 0;
}