#include "operators/matmul.hpp"

#include <cassert>
#include <iostream>

int main()
{
    Tensor a({2, 3});
    Tensor b({3, 2});

    // A =
    // [1 2 3]
    // [4 5 6]

    a({0, 0}) = 1.0f;
    a({0, 1}) = 2.0f;
    a({0, 2}) = 3.0f;

    a({1, 0}) = 4.0f;
    a({1, 1}) = 5.0f;
    a({1, 2}) = 6.0f;

    // B =
    // [ 7  8]
    // [ 9 10]
    // [11 12]

    b({0, 0}) = 7.0f;
    b({0, 1}) = 8.0f;

    b({1, 0}) = 9.0f;
    b({1, 1}) = 10.0f;

    b({2, 0}) = 11.0f;
    b({2, 1}) = 12.0f;

    Tensor result = matmul(a, b);

    // Expected:
    //
    // [ 58  64]
    // [139 154]

    assert(result.shape() == std::vector<size_t>({2, 2}));

    assert(result({0, 0}) == 58.0f);
    assert(result({0, 1}) == 64.0f);
    assert(result({1, 0}) == 139.0f);
    assert(result({1, 1}) == 154.0f);

    std::cout << "MatMul test passed!" << std::endl;

    return 0;
}
