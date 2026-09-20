#include <cassert>
#include <iostream>
#include <stdexcept>

#include "operators/add.hpp"

int main() {

    // -------------------------
    // Basic addition
    // -------------------------

    Tensor a({2, 3});
    Tensor b({2, 3});

    a({0, 0}) = 1.0f;
    a({0, 1}) = 2.0f;
    a({0, 2}) = 3.0f;

    a({1, 0}) = 4.0f;
    a({1, 1}) = 5.0f;
    a({1, 2}) = 6.0f;

    b({0, 0}) = 10.0f;
    b({0, 1}) = 20.0f;
    b({0, 2}) = 30.0f;

    b({1, 0}) = 40.0f;
    b({1, 1}) = 50.0f;
    b({1, 2}) = 60.0f;

    Tensor result = add(a, b);

    // -------------------------
    // Verify result
    // -------------------------

    assert(result.shape() == std::vector<size_t>({2, 3}));

    assert(result({0, 0}) == 11.0f);
    assert(result({0, 1}) == 22.0f);
    assert(result({0, 2}) == 33.0f);

    assert(result({1, 0}) == 44.0f);
    assert(result({1, 1}) == 55.0f);
    assert(result({1, 2}) == 66.0f);

    // -------------------------
    // Shape mismatch
    // -------------------------

    Tensor c({2, 3});
    Tensor d({3, 2});

    bool caught_shape_error = false;

    try {
        add(c, d);
    }
    catch (const std::invalid_argument&) {
        caught_shape_error = true;
    }

    assert(caught_shape_error);

    std::cout << "All Add tests passed!\n";

    return 0;
}
