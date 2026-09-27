#include "executor.hpp"
#include "operators/relu.hpp"
#include "operators/add.hpp"
#include "operators/matmul.hpp"
#include "operators/softmax.hpp"
#include "operators/layernorm.hpp"
#include "operators/gelu.hpp"

#include <memory>
#include <stdexcept>
#include <vector>

std::vector<Tensor> Executor::run(
    const Graph& graph,
    const std::vector<Tensor>& inputs
) const
{
    if (!graph.validate())
    {
        throw std::runtime_error(
            "Cannot execute invalid graph");
    }

    const std::vector<size_t> order =
        graph.execution_order();

    const std::vector<size_t> input_nodes =
        graph.input_nodes();

    if (inputs.size() != input_nodes.size())
    {
        throw std::runtime_error(
            "Number of input tensors does not match "
            "number of graph input nodes");
    }

    // One tensor slot for every graph node.
    //
    // values[node_id] contains the tensor
    // produced by that node.
    std::vector<std::unique_ptr<Tensor>> values(
        graph.size());

    size_t input_index = 0;

    for (size_t node_id : order)
    {
        const Node& current =
            graph.node(node_id);

        const std::string& operation =
            current.operation();

        // --------------------------------
        // Input
        // --------------------------------
        if (operation == "Input")
        {
            values[node_id] =
                std::make_unique<Tensor>(
                    inputs[input_index]);

            ++input_index;
        }

        // --------------------------------
        // ReLU
        // --------------------------------
        else if (operation == "ReLU")
        {
            if (current.inputs().size() != 1)
            {
                throw std::runtime_error(
                    "ReLU expects exactly one input");
            }

            const size_t input_node =
                current.inputs()[0];

            if (!values[input_node])
            {
                throw std::runtime_error(
                    "ReLU input tensor is not available");
            }

            Tensor result =
                relu(*values[input_node]);

            values[node_id] =
                std::make_unique<Tensor>(
                    std::move(result));
        }
	        // --------------------------------
        // Add
        // --------------------------------
        else if (operation == "Add")
        {
            if (current.inputs().size() != 2)
            {
                throw std::runtime_error(
                    "Add expects exactly two inputs");
            }

            const size_t input_a =
                current.inputs()[0];

            const size_t input_b =
                current.inputs()[1];

            if (!values[input_a] ||
                !values[input_b])
            {
                throw std::runtime_error(
                    "Add input tensor is not available");
            }

            Tensor result =
                add(
                    *values[input_a],
                    *values[input_b]);

            values[node_id] =
                std::make_unique<Tensor>(
                    std::move(result));
        }
	// --------------------------------
// MatMul
// --------------------------------
else if (operation == "MatMul")
{
    if (current.inputs().size() != 2)
    {
        throw std::runtime_error(
            "MatMul expects exactly two inputs");
    }

    const size_t input_a =
        current.inputs()[0];

    const size_t input_b =
        current.inputs()[1];

    if (!values[input_a] ||
        !values[input_b])
    {
        throw std::runtime_error(
            "MatMul input tensor is not available");
    }

    Tensor result =
        matmul(
            *values[input_a],
            *values[input_b]);

    values[node_id] =
        std::make_unique<Tensor>(
            std::move(result));
}
	// --------------------------------
// Softmax
// --------------------------------
else if (operation == "Softmax")
{
    if (current.inputs().size() != 1)
    {
        throw std::runtime_error(
            "Softmax expects exactly one input");
    }

    const size_t input_node =
        current.inputs()[0];

    if (!values[input_node])
    {
        throw std::runtime_error(
            "Softmax input tensor is not available");
    }

    Tensor result =
        softmax(*values[input_node]);

    values[node_id] =
        std::make_unique<Tensor>(
            std::move(result));
}

// --------------------------------
// LayerNorm
// --------------------------------
else if (operation == "LayerNorm")
{
    if (current.inputs().size() != 3)
    {
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
        !values[beta_node])
    {
        throw std::runtime_error(
            "LayerNorm input tensor is not available");
    }

    Tensor result =
        layernorm(
            *values[input_node],
            *values[gamma_node],
            *values[beta_node]);

    values[node_id] =
        std::make_unique<Tensor>(
            std::move(result));
}



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

    Tensor result =
        gelu(*values[input_node]);

    values[node_id] =
        std::make_unique<Tensor>(
            std::move(result));
}

        // --------------------------------
        // Unsupported operation
        // --------------------------------
        else
        {
            throw std::runtime_error(
                "Unsupported operation: " +
                operation);
        }
    }

    std::vector<Tensor> outputs;

    const std::vector<size_t> output_nodes =
        graph.output_nodes();

    for (size_t node_id : output_nodes)
    {
        if (!values[node_id])
        {
            throw std::runtime_error(
                "Graph output was not computed");
        }

        outputs.push_back(
            *values[node_id]);
    }

    return outputs;
}
