#include "executor.hpp"
#include "operators/gelu.hpp"

#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

int main()
{
    Graph graph;

    const size_t input_node =
        graph.addNode("Input");

    const size_t gelu_node =
        graph.addNode("GELU");

    graph.connect(
        input_node,
        gelu_node);

    assert(graph.validate());

    Executor executor;

    Tensor input({3});

    input({0}) = -1.0f;
    input({1}) = 0.0f;
    input({2}) = 1.0f;

    std::vector<Tensor> outputs =
        executor.run(
            graph,
            {input});

    assert(outputs.size() == 1);

    assert(
        outputs[0].shape() ==
        std::vector<size_t>({3}));

    Tensor expected = gelu(input);

    for (size_t i = 0;
         i < outputs[0].size();
         ++i)
    {
        assert(
            std::abs(
                outputs[0].data()[i] -
                expected.data()[i]
            ) < 1e-6f
        );
    }

    std::cout
        << "Executor GELU test passed!"
        << std::endl;

    return 0;
}
