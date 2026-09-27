#include "executor.hpp"

#include <cassert>
#include <iostream>
#include <vector>

int main()
{
    Graph graph;

    const size_t input_a =
        graph.addNode("Input");

    const size_t input_b =
        graph.addNode("Input");

    const size_t matmul_node =
        graph.addNode("MatMul");

    graph.connect(input_a, matmul_node);
    graph.connect(input_b, matmul_node);

    assert(graph.validate());

    Executor executor;

    Tensor tensor_a({2, 3});
    Tensor tensor_b({3, 2});

    tensor_a({0, 0}) = 1.0f;
    tensor_a({0, 1}) = 2.0f;
    tensor_a({0, 2}) = 3.0f;

    tensor_a({1, 0}) = 4.0f;
    tensor_a({1, 1}) = 5.0f;
    tensor_a({1, 2}) = 6.0f;

    tensor_b({0, 0}) = 10.0f;
    tensor_b({0, 1}) = 20.0f;

    tensor_b({1, 0}) = 30.0f;
    tensor_b({1, 1}) = 40.0f;

    tensor_b({2, 0}) = 50.0f;
    tensor_b({2, 1}) = 60.0f;

    std::vector<Tensor> outputs =
        executor.run(
            graph,
            {tensor_a, tensor_b});

    assert(outputs.size() == 1);

    assert(
        outputs[0].shape() ==
        std::vector<size_t>({2, 2}));

    assert(outputs[0]({0, 0}) == 220.0f);
    assert(outputs[0]({0, 1}) == 280.0f);

    assert(outputs[0]({1, 0}) == 490.0f);
    assert(outputs[0]({1, 1}) == 640.0f);

    std::cout
        << "Executor MatMul test passed!"
        << std::endl;

    return 0;
}
