#include "executor.hpp"

#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

int main()
{
    Graph graph;

    const size_t input_node =
        graph.addNode("Input");

    const size_t softmax_node =
        graph.addNode("Softmax");

    graph.connect(
        input_node,
        softmax_node);

    assert(graph.validate());

    Executor executor;

    Tensor input({3});

    input({0}) = 1.0f;
    input({1}) = 2.0f;
    input({2}) = 3.0f;

    std::vector<Tensor> outputs =
        executor.run(
            graph,
            {input});

    // One graph output.
    assert(outputs.size() == 1);

    // Softmax preserves the input shape.
    assert(
        outputs[0].shape() ==
        std::vector<size_t>({3}));

    // Expected:
    //
    // [0.0900306, 0.2447285, 0.6652409]

    assert(
        std::abs(
            outputs[0]({0}) - 0.0900306f
        ) < 1e-5f
    );

    assert(
        std::abs(
            outputs[0]({1}) - 0.2447285f
        ) < 1e-5f
    );

    assert(
        std::abs(
            outputs[0]({2}) - 0.6652409f
        ) < 1e-5f
    );

    // Every probability must be between 0 and 1.
    for (size_t i = 0;
         i < outputs[0].size();
         ++i)
    {
        assert(
            outputs[0].data()[i] >= 0.0f);

        assert(
            outputs[0].data()[i] <= 1.0f);
    }

    // Probabilities must sum to approximately 1.
    float sum = 0.0f;

    for (size_t i = 0;
         i < outputs[0].size();
         ++i)
    {
        sum += outputs[0].data()[i];
    }

    assert(
        std::abs(sum - 1.0f) < 1e-5f);

    std::cout
        << "Executor Softmax test passed!"
        << std::endl;

    return 0;
}
