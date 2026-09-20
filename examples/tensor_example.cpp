#include <iostream>

#include "tensor.hpp"

int main() {
    Tensor tensor({2, 3});

    tensor(0, 0) = 1.0f;
    tensor(0, 1) = 2.0f;
    tensor(0, 2) = 3.0f;

    tensor(1, 0) = 4.0f;
    tensor(1, 1) = 5.0f;
    tensor(1, 2) = 6.0f;

    std::cout << "Tensor:\n";

    for (size_t row = 0; row < 2; ++row) {
        for (size_t column = 0; column < 3; ++column) {
            std::cout << tensor(row, column) << " ";
        }

        std::cout << "\n";
    }

    std::cout << "\nShape: ";

    for (size_t dimension : tensor.shape()) {
        std::cout << dimension << " ";
    }

    std::cout << "\n";

    std::cout << "Number of elements: "
              << tensor.size()
              << "\n";

    return 0;
}
