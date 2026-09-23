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

    // Valid graph
    assert(graph.validate());

    // Self-connection must fail
    bool self_connection_thrown = false;

    try
    {
        graph.connect(relu, relu);
    }
    catch (const std::invalid_argument&)
    {
        self_connection_thrown = true;
    }

    assert(self_connection_thrown);

    // Duplicate connection must fail
    bool duplicate_connection_thrown = false;

    try
    {
        graph.connect(input, matmul);
    }
    catch (const std::invalid_argument&)
    {
        duplicate_connection_thrown = true;
    }

    assert(duplicate_connection_thrown);

    // Invalid node connection must fail
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

    // Graph should still be valid
    assert(graph.validate());


    // Test execution order
    std::vector<size_t> order =
        graph.execution_order();

    assert(order.size() == 4);

    assert(order[0] == input);
    assert(order[1] == matmul);
    assert(order[2] == add);
    assert(order[3] == relu);


       // Test graph input nodes
    std::vector<size_t> input_nodes =
        graph.input_nodes();

    assert(input_nodes.size() == 1);
    assert(input_nodes[0] == input);


    // Test graph output nodes
    std::vector<size_t> output_nodes =
        graph.output_nodes();

    assert(output_nodes.size() == 1);
    assert(output_nodes[0] == relu);




       // Test multiple graph inputs

    Graph multi_input_graph;

    size_t x =
        multi_input_graph.addNode("Input");

    size_t weights =
        multi_input_graph.addNode("Input");

    size_t multi_matmul =
        multi_input_graph.addNode("MatMul");

    multi_input_graph.connect(x, multi_matmul);
    multi_input_graph.connect(weights, multi_matmul);

    assert(multi_input_graph.validate());

    std::vector<size_t> multi_inputs =
        multi_input_graph.input_nodes();

    assert(multi_inputs.size() == 2);
    assert(multi_inputs[0] == x);
    assert(multi_inputs[1] == weights);

    std::vector<size_t> multi_outputs =
        multi_input_graph.output_nodes();

    assert(multi_outputs.size() == 1);
    assert(multi_outputs[0] == multi_matmul);

    // Test branching graph
    Graph branching_graph;

    size_t branch_input =
        branching_graph.addNode("Input");

    size_t branch_matmul =
        branching_graph.addNode("MatMul");

    size_t branch_add =
        branching_graph.addNode("Add");

    size_t branch_relu =
        branching_graph.addNode("ReLU");

    branching_graph.connect(
        branch_input,
        branch_matmul
    );

    branching_graph.connect(
        branch_input,
        branch_add
    );

    branching_graph.connect(
        branch_matmul,
        branch_relu
    );

    branching_graph.connect(
        branch_add,
        branch_relu
    );

    assert(branching_graph.validate());

    std::vector<size_t> branch_order =
        branching_graph.execution_order();

    assert(branch_order.size() == 4);

    // Input must execute first.
    assert(branch_order[0] == branch_input);

    // ReLU must execute last.
    assert(branch_order[3] == branch_relu);




    // Test cycle detection
    Graph cyclic_graph;

    size_t a =
        cyclic_graph.addNode("A");

    size_t b =
        cyclic_graph.addNode("B");

    size_t c =
        cyclic_graph.addNode("C");

    cyclic_graph.connect(a, b);
    cyclic_graph.connect(b, c);

    // We cannot create c -> a using the current
    // public API if validation rejects cycles.




    cyclic_graph.connect(c, a);

    bool cycle_exception_thrown = false;

    try
    {
        cyclic_graph.execution_order();
    }
    catch (const std::runtime_error&)
    {
        cycle_exception_thrown = true;
    }

    assert(cycle_exception_thrown);




    std::cout << "Graph execution order  test passed!" << std::endl;

    return 0;

}
