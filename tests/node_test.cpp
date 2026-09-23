#include "node.hpp"

#include <cassert>
#include <iostream>

int main()
{
    Node matmul_node(0, "MatMul");

    assert(matmul_node.id() == 0);
    assert(matmul_node.operation() == "MatMul");

    assert(matmul_node.inputs().empty());
    assert(matmul_node.outputs().empty());

    matmul_node.add_output(1);

    assert(matmul_node.outputs().size() == 1);
    assert(matmul_node.outputs()[0] == 1);

    Node add_node(1, "Add");

    add_node.add_input(0);
    add_node.add_output(2);

    assert(add_node.inputs().size() == 1);
    assert(add_node.inputs()[0] == 0);

    assert(add_node.outputs().size() == 1);
    assert(add_node.outputs()[0] == 2);

    std::cout << "Node test passed!" << std::endl;

    return 0;
}
