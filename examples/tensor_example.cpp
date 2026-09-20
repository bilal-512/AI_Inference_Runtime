#include <iostream>

#include "tensor.hpp"

int main() {

    Tensor tensor({2, 3, 4});

    tensor({0, 0, 0}) = 1.0f;
    tensor({0, 0, 1}) = 2.0f;
    tensor({1, 2, 3}) = 42.0f;

    std::cout << "Number of dimensions: "
              << tensor.ndim()
              << "\n";

    std::cout << "Number of elements: "
              << tensor.size()
              << "\n";

    std::cout << "tensor(0,0,0): "
              << tensor({0, 0, 0})
              << "\n";

    std::cout << "tensor(1,2,3): "
              << tensor({1, 2, 3})
              << "\n";

    return 0;
}