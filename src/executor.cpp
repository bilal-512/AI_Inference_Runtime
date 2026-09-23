#include "executor.hpp"

#include "operators/add.hpp"
#include "operators/gelu.hpp"
#include "operators/layernorm.hpp"
#include "operators/matmul.hpp"
#include "operators/relu.hpp"
#include "operators/softmax.hpp"

#include <stdexcept>
#include <string>
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

    std::vector<Tensor> values;

    values.reserve(graph.size());

    std::vector<bool> computed(graph.size(), false);

    size_t input_index = 0;

    std::vector<size_t> order =
        graph.execution_order();

    for (size_t node_id : order)
    {
        const Node& current = graph.node(node_id);

        const std::string& operation =
            current.operation();

        if (operation == "Input")
        {
            if (input_index >= inputs.size())
            {
                throw std::runtime_error(
                    "Not enough input tensors");
            }

            values.push_back(inputs[input_index]);

            ++input_index;
            computed[node_id] = true;

            continue;
        }

        if (current.inputs().empty())
        {
            throw std::runtime_error(
                "Non-input node has no inputs");
        }

        const size_t first_input =
            current.inputs()[0];

        if (!computed[first_input])
        {
            throw std::runtime_error(
                "Input dependency has not been computed");
        }

        const Tensor& input =
            values[first_input];

        Tensor result =
            [&]() -> Tensor
            {
                if (operation == "ReLU")
                {
                    return relu(input);
                }

                if (operation == "Softmax")
                {
                    return softmax(input);
                }

                if (operation == "GELU")
                {
                    return gelu(input);
                }

                throw std::runtime_error(
                    "Unsupported unary operation: " +
                    operation);
            }();

        values.push_back(result);

        computed[node_id] = true;
    }

    return values;
}
