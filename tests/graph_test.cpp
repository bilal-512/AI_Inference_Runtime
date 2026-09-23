#include "graph.hpp"

#include <cassert>
#include <iostream>
#include <stdexcept>

int main()
{
    Graph graph;

    size_t input = graph.addNode("Input");
    size_t matmul = graph.addNode("MatMul");
    size_t add = graph.addNode("Add");
    size_t relu = graph.addNode("ReLU");

    assert(input == 0);
    assert(matmul == 1);
    assert(add == 2);
    assert(relu == 3);

    assert(graph.size() == 4);

    // Test node information
    assert(graph.node(input).operation() == "Input");
    assert(graph.node(matmul).operation() == "MatMul");
    assert(graph.node(add).operation() == "Add");
    assert(graph.node(relu).operation() == "ReLU");

    // Build connections:
    //
    // Input -> MatMul -> Add -> ReLU

    graph.connect(input, matmul);
    graph.connect(matmul, add);
    graph.connect(add, relu);

    // Input -> MatMul
    assert(graph.node(input).outputs().size() == 1);
    assert(graph.node(input).outputs()[0] == matmul);

    assert(graph.node(matmul).inputs().size() == 1);
    assert(graph.node(matmul).inputs()[0] == input);

    // MatMul -> Add
    assert(graph.node(matmul).outputs().size() == 1);
    assert(graph.node(matmul).outputs()[0] == add);

    assert(graph.node(add).inputs().size() == 1);
    assert(graph.node(add).inputs()[0] == matmul);

    // Add -> ReLU
    assert(graph.node(add).outputs().size() == 1);
    assert(graph.node(add).outputs()[0] == relu);

    assert(graph.node(relu).inputs().size() == 1);
    assert(graph.node(relu).inputs()[0] == add);

    // Invalid connection
    bool exception_thrown = false;

    try
    {
        graph.connect(0, 10);
    }
    catch (const std::out_of_range&)
    {
        exception_thrown = true;
    }

    assert(exception_thrown);

    std::cout << "Graph connection test passed!" << std::endl;

    return 0;
}
