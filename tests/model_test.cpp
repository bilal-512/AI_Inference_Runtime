
#include "model.hpp"
#include "executor.hpp"

#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

int main() {
    Model model;

    // 1. Create the computation graph.
    size_t input_node =
        model.graph().addNode("Input");

    size_t matmul1_node =
        model.graph().addNode("MatMul");

    size_t add1_node =
        model.graph().addNode("Add");

    size_t relu_node =
        model.graph().addNode("ReLU");

    size_t matmul2_node =
        model.graph().addNode("MatMul");

    size_t add2_node =
        model.graph().addNode("Add");

    // 2. Connect the layers.
    model.graph().connect(input_node, matmul1_node);
    model.graph().connect(matmul1_node, add1_node);
    model.graph().connect(add1_node, relu_node);
    model.graph().connect(relu_node, matmul2_node);
    model.graph().connect(matmul2_node, add2_node);

    // 3. First layer weights: shape (2, 3).
    Tensor weight1({2, 3});

    float w1[] = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f
    };

    for (size_t i = 0; i < weight1.size(); ++i) {
        weight1.data()[i] = w1[i];
    }

    // First layer bias: shape (1, 3).
    Tensor bias1({1, 3});

    bias1.data()[0] = 0.5f;
    bias1.data()[1] = 1.0f;
    bias1.data()[2] = -1.0f;

    // 4. Second layer weights: shape (3, 2).
    Tensor weight2({3, 2});

    float w2[] = {
        1.0f,  2.0f,
        0.0f,  1.0f,
        1.0f, -1.0f
    };

    for (size_t i = 0; i < weight2.size(); ++i) {
        weight2.data()[i] = w2[i];
    }

    // Second layer bias: shape (1, 2).
    Tensor bias2({1, 2});

    bias2.data()[0] = 0.5f;
    bias2.data()[1] = -2.0f;

    // 5. Store parameters in the model.
    model.add_parameter("weight1", weight1);
    model.add_parameter("bias1", bias1);
    model.add_parameter("weight2", weight2);
    model.add_parameter("bias2", bias2);

    // 6. Tell each operation which parameter to use.
    model.graph().node(matmul1_node).set_attribute(
        "weight", "weight1");

    model.graph().node(add1_node).set_attribute(
        "bias", "bias1");

    model.graph().node(matmul2_node).set_attribute(
        "weight", "weight2");

    model.graph().node(add2_node).set_attribute(
        "bias", "bias2");

    // 7. Input tensor X = [1, 2].
    Tensor input({1, 2});

    input.data()[0] = 1.0f;
    input.data()[1] = 2.0f;

    // 8. Execute the complete neural network.
    Executor executor;

    std::vector<Tensor> outputs =
        executor.run(model, {input});

    assert(outputs.size() == 1);

    const Tensor& output = outputs[0];

    assert(
        output.shape() ==
        std::vector<size_t>({1, 2})
    );

    // Expected output: [24, 16].
    const float expected[] = {
        24.0f, 16.0f
    };

    for (size_t i = 0; i < output.size(); ++i) {
        assert(
            std::abs(output.data()[i] - expected[i])
            < 1e-5f
        );
    }

    std::cout << "Two-layer neural network test passed!"
              << std::endl;

    std::cout << "Output: ["
              << output.data()[0] << ", "
              << output.data()[1] << "]"
              << std::endl;

    return 0;
}
