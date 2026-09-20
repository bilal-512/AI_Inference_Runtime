#include <iostream>

#include "tensor.hpp"

int main() {
    Tensor tensor({2, 3});

    std::cout << "Tensor created successfully!\n";

    std::cout << "Shape: ";

    for (size_t dimension : tensor.shape()) {
        std::cout << dimension << " ";
    }

    std::cout << "\n";

    std::cout << "Number of elements: "
              << tensor.size()
              << "\n";

    return 0;
}
