#include <cassert>
#include <iostream>

#include "tensor.hpp"

int main() {
    Tensor tensor({2, 3});

    assert(tensor.size() == 6);

    tensor(0, 0) = 10.0f;
    tensor(1, 2) = 20.0f;

    assert(tensor(0, 0) == 10.0f);
    assert(tensor(1, 2) == 20.0f);

    std::cout << "Tensor tests passed!\n";

    return 0;
}
