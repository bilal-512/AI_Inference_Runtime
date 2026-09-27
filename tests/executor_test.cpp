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

    const size_t gamma_node =
        graph.addNode("Input");

    const size_t beta_node =
        graph.addNode("Input");

    const size_t layernorm_node =
        graph.addNode("LayerNorm");

    graph.connect(
        input_node,
        layernorm_node);

    graph.connect(
        gamma_node,
        layernorm_node);

    graph.connect(
        beta_node,
        layernorm_node);

    assert(graph.validate());

    Executor executor;

    Tensor input({3});
    Tensor gamma({3});
    Tensor beta({3});

    input({0}) = 1.0f;
    input({1}) = 2.0f;
    input({2}) = 3.0f;

    // Scale = 1
    gamma({0}) = 1.0f;
    gamma({1}) = 1.0f;
    gamma({2}) = 1.0f;

    // Bias = 0
    beta({0}) = 0.0f;
    beta({1}) = 0.0f;
    beta({2}) = 0.0f;

    std::vector<Tensor> outputs =
        executor.run(
            graph,
            {
                input,
                gamma,
                beta
            });

    // One graph output.
    assert(outputs.size() == 1);

    // LayerNorm preserves the input shape.
    assert(
        outputs[0].shape() ==
        std::vector<size_t>({3}));

    // The output should have approximately
    // zero mean and unit variance.

    float mean = 0.0f;

    for (size_t i = 0;
         i < outputs[0].size();
         ++i)
    {
        mean += outputs[0].data()[i];
    }

    mean /= static_cast<float>(
        outputs[0].size());

    assert(
        std::abs(mean) < 1e-5f);

    float variance = 0.0f;

    for (size_t i = 0;
         i < outputs[0].size();
         ++i)
    {
        variance +=
            outputs[0].data()[i] *
            outputs[0].data()[i];
    }

    variance /= static_cast<float>(
        outputs[0].size());

    assert(
        std::abs(variance - 1.0f) < 1e-4f);

    std::cout
        << "Executor LayerNorm test passed!"
        << std::endl;

    return 0;
}

