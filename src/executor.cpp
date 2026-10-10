
#include "executor.hpp"

#include "operators/relu.hpp"
#include "operators/add.hpp"
#include "operators/matmul.hpp"
#include "operators/softmax.hpp"
#include "operators/layernorm.hpp"
#include "operators/gelu.hpp"

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// Graph-only execution.
std::vector<Tensor> Executor::run(
    const Graph& graph,
    const std::vector<Tensor>& inputs
) const {
    return run_impl(graph, nullptr, inputs);
}

// Model-aware execution.
std::vector<Tensor> Executor::run(
    const Model& model,
    const std::vector<Tensor>& inputs
) const {
    return run_impl(model.graph(), &model, inputs);
}

// Shared execution implementation.
std::vector<Tensor> Executor::run_impl(
    const Graph& graph,
    const Model* model,
    const std::vector<Tensor>& inputs
) const {
    if (!graph.validate()) {
        throw std::runtime_error(
            "Cannot execute invalid graph");
    }

    const std::vector<size_t> order =
        graph.execution_order();

    const std::vector<size_t> input_nodes =
        graph.input_nodes();

    if (inputs.size() != input_nodes.size()) {
        throw std::runtime_error(
            "Number of input tensors does not match "
            "number of graph input nodes");
    }

    // Store the computed tensor for each graph node.
    std::vector<std::unique_ptr<Tensor>> values(
        graph.size());

    size_t input_index = 0;

    for (size_t node_id : order) {
        const Node& current = graph.node(node_id);

        const std::string& operation =
            current.operation();

        // Input
        if (operation == "Input") {
            values[node_id] =
                std::make_unique<Tensor>(
                    inputs[input_index]);

            ++input_index;
        }

        // ReLU
        else if (operation == "ReLU") {
            if (current.inputs().size() != 1) {
                throw std::runtime_error(
                    "ReLU expects exactly one input");
            }

            const size_t input_node =
                current.inputs()[0];

            if (!values[input_node]) {
                throw std::runtime_error(
                    "ReLU input tensor is not available");
            }

            values[node_id] =
                std::make_unique<Tensor>(
                    relu(*values[input_node]));
        }



        // Add
        else if (operation == "Add") {
            // Mode 1: add a named bias stored in Model.
            if (model != nullptr &&
                current.inputs().size() == 1 &&
                current.has_attribute("bias")) {

                const size_t input_node =
                    current.inputs()[0];

                if (!values[input_node]) {
                    throw std::runtime_error(
                        "Add input tensor is not available");
                }

                const std::string& bias_name =
                    current.attribute("bias");

                const Tensor& bias =
                    model->parameter(bias_name);

                values[node_id] =
                    std::make_unique<Tensor>(
                        add(*values[input_node], bias));
            }

            // Mode 2: add two graph input tensors.
            else if (current.inputs().size() == 2) {
                const size_t input_a =
                    current.inputs()[0];

                const size_t input_b =
                    current.inputs()[1];

                if (!values[input_a] ||
                    !values[input_b]) {
                    throw std::runtime_error(
                        "Add input tensor is not available");
                }

                values[node_id] =
                    std::make_unique<Tensor>(
                        add(
                            *values[input_a],
                            *values[input_b]));
            }

            else {
                throw std::runtime_error(
                    "Add expects two graph inputs, or one "
                    "graph input and a named model bias");
            }
        }


        // MatMul
        else if (operation == "MatMul") {
            // Mode 1: one graph input and a named Model weight.
            if (model != nullptr &&
                current.inputs().size() == 1 &&
                current.has_attribute("weight")) {

                const size_t input_node =
                    current.inputs()[0];

                if (!values[input_node]) {
                    throw std::runtime_error(
                        "MatMul input tensor is not available");
                }

                const std::string& weight_name =
                    current.attribute("weight");

                const Tensor& weight =
                    model->parameter(weight_name);

                values[node_id] =
                    std::make_unique<Tensor>(
                        matmul(
                            *values[input_node],
                            weight));
            }

            // Mode 2: two graph input tensors.
            else if (current.inputs().size() == 2) {
                const size_t input_a =
                    current.inputs()[0];

                const size_t input_b =
                    current.inputs()[1];

                if (!values[input_a] ||
                    !values[input_b]) {
                    throw std::runtime_error(
                        "MatMul input tensor is not available");
                }

                values[node_id] =
                    std::make_unique<Tensor>(
                        matmul(
                            *values[input_a],
                            *values[input_b]));
            }

            else {
                throw std::runtime_error(
                    "MatMul expects two graph inputs, or one "
                    "graph input and a named model weight");
            }
        }

        // Softmax
        else if (operation == "Softmax") {
            if (current.inputs().size() != 1) {
                throw std::runtime_error(
                    "Softmax expects exactly one input");
            }

            const size_t input_node =
                current.inputs()[0];

            if (!values[input_node]) {
                throw std::runtime_error(
                    "Softmax input tensor is not available");
            }

            values[node_id] =
                std::make_unique<Tensor>(
                    softmax(*values[input_node]));
        }

        // LayerNorm
        else if (operation == "LayerNorm") {
            if (current.inputs().size() != 3) {
                throw std::runtime_error(
                    "LayerNorm expects exactly three inputs");
            }

            const size_t input_node =
                current.inputs()[0];

            const size_t gamma_node =
                current.inputs()[1];

            const size_t beta_node =
                current.inputs()[2];

            if (!values[input_node] ||
                !values[gamma_node] ||
                !values[beta_node]) {
                throw std::runtime_error(
                    "LayerNorm input tensor is not available");
            }

            values[node_id] =
                std::make_unique<Tensor>(
                    layernorm(
                        *values[input_node],
                        *values[gamma_node],
                        *values[beta_node]));
        }

        // GELU
        else if (operation == "GELU") {
            if (current.inputs().size() != 1) {
                throw std::runtime_error(
                    "GELU expects exactly one input");
            }

            const size_t input_node =
                current.inputs()[0];

            if (!values[input_node]) {
                throw std::runtime_error(
                    "GELU input tensor is not available");
            }

            values[node_id] =
                std::make_unique<Tensor>(
                    gelu(*values[input_node]));
        }

        // Unsupported operation
        else {
            throw std::runtime_error(
                "Unsupported operation: " + operation);
        }
    }

    // Collect graph outputs.
    std::vector<Tensor> outputs;

    const std::vector<size_t> output_nodes =
        graph.output_nodes();

    for (size_t node_id : output_nodes) {
        if (!values[node_id]) {
            throw std::runtime_error(
                "Graph output was not computed");
        }

        outputs.push_back(*values[node_id]);
    }

    return outputs;
}
