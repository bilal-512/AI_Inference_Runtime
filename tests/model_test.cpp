
#include "model.hpp"
#include "executor.hpp"

#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

int main() {
    Model model;

    // Build: Input -> MatMul -> Add
    size_t input_node =
        model.graph().addNode("Input");

    size_t matmul_node =
        model.graph().addNode("MatMul");

    size_t add_node =
        model.graph().addNode("Add");

    model.graph().connect(
        input_node,
        matmul_node
    );

    model.graph().connect(
        matmul_node,
        add_node
    );

    // Store W with shape (2, 3).
    Tensor weight({2, 3});

    float weight_values[] = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f
    };

    for (size_t i = 0; i < weight.size(); ++i) {
        weight.data()[i] = weight_values[i];
    }

    model.add_parameter("weight", weight);

    // Store b with shape (1, 3).
    Tensor bias({1, 3});

    bias.data()[0] = 0.5f;
    bias.data()[1] = 1.0f;
    bias.data()[2] = -1.0f;

    model.add_parameter("bias", bias);

    // Connect operation nodes to their parameters.
    model.graph().node(matmul_node).set_attribute(
        "weight",
        "weight"
    );

    model.graph().node(add_node).set_attribute(
        "bias",
        "bias"
    );

    // Input X = [1, 2].
    Tensor input({1, 2});

    input.data()[0] = 1.0f;
    input.data()[1] = 2.0f;

    Executor executor;

    std::vector<Tensor> outputs =
        executor.run(model, {input});

    assert(outputs.size() == 1);

    const Tensor& output = outputs[0];

    assert(
        output.shape() ==
        std::vector<size_t>({1, 3})
    );

    const float expected[] = {
        9.5f, 13.0f, 14.0f
    };

    for (size_t i = 0; i < output.size(); ++i) {
        assert(
            std::abs(output.data()[i] - expected[i])
            < 1e-5f
        );
    }

    std::cout
        << "Model linear layer test passed!"
        << std::endl;

    return 0;
}
