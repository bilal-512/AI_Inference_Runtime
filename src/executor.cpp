#include "executor.hpp"

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

        if (current.operation() == "Input")
        {
            values[node_id] =
                std::make_unique<Tensor>(
                    inputs[input_index]);

            ++input_index;
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
