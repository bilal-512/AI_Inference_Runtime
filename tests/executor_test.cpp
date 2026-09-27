#include "executor.hpp"

#include <cassert>
#include <iostream>
#include <vector>

int main()
{
    Graph graph;

    const size_t input =
        graph.addNode("Input");

    const size_t relu_node =
        graph.addNode("ReLU");

    graph.connect(input, relu_node);

    assert(graph.validate());

    Executor executor;

    Tensor input_tensor({5});

    input_tensor.data()[0] = -2.0f;
    input_tensor.data()[1] = -1.0f;
    input_tensor.data()[2] = 0.0f;
    input_tensor.data()[3] = 3.0f;
    input_tensor.data()[4] = 5.0f;

    std::vector<Tensor> outputs =
        executor.run(
            graph,
            {input_tensor});

    assert(outputs.size() == 1);

    assert(outputs[0].shape() ==
           input_tensor.shape());

    assert(outputs[0].data()[0] == 0.0f);
    assert(outputs[0].data()[1] == 0.0f);
    assert(outputs[0].data()[2] == 0.0f);
    assert(outputs[0].data()[3] == 3.0f);
    assert(outputs[0].data()[4] == 5.0f);

    std::cout
        << "Executor ReLU test passed!"
        << std::endl;

    return 0;
}
