#include "executor.hpp"

#include <cassert>
#include <iostream>
#include <vector>

int main()
{
    Graph graph;

    size_t input =
        graph.addNode("Input");

    Executor executor;

    Tensor input_tensor({2});

    input_tensor.data()[0] = 10.0f;
    input_tensor.data()[1] = 20.0f;

    std::vector<Tensor> inputs;

    inputs.push_back(input_tensor);

    std::vector<Tensor> outputs =
        executor.run(graph, inputs);

    assert(outputs.size() == 1);

    assert(outputs[0].shape().size() == 1);
    assert(outputs[0].shape()[0] == 2);

    assert(outputs[0].data()[0] == 10.0f);
    assert(outputs[0].data()[1] == 20.0f);

    std::cout
        << "Executor tensor storage test passed!"
        << std::endl;

    return 0;
}
