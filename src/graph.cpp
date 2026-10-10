#include <algorithm>

#include "graph.hpp"
#include <queue>
#include <stdexcept>

size_t Graph::addNode(const std::string& operation)
{
    size_t id = nodes_.size();

    nodes_.emplace_back(id, operation);

    return id;
}

const Node& Graph::node(size_t id) const
{
    if (id >= nodes_.size())
    {
        throw std::out_of_range(
            "Graph node ID out of range");
    }

    return nodes_[id];
}

Node& Graph::node(size_t id)
{
    if (id >= nodes_.size())
    {
        throw std::out_of_range(
            "Graph node ID out of range");
    }

    return nodes_[id];
}

size_t Graph::size() const
{
    return nodes_.size();
}

void Graph::connect(size_t from, size_t to)
{
    if (from >= nodes_.size() ||
        to >= nodes_.size())
    {
        throw std::out_of_range(
            "Graph connection node ID out of range");
    }

    if (from == to)
    {
        throw std::invalid_argument(
            "Graph node cannot connect to itself");
    }

    const auto& outputs = nodes_[from].outputs();

    if (std::find(
            outputs.begin(),
            outputs.end(),
            to
        ) != outputs.end())
    {
        throw std::invalid_argument(
            "Graph connection already exists");
    }

    nodes_[from].add_output(to);
    nodes_[to].add_input(from);
}



bool Graph::validate() const {
    for (const Node& current : nodes_) {
        const size_t current_id = current.id();

        // Check that this node has a valid ID.
        if (current_id >= nodes_.size()) {
            return false;
        }

        // Check input connections.
        for (size_t input_id : current.inputs()) {
            if (input_id >= nodes_.size()) {
                return false;
            }

            if (input_id == current_id) {
                return false;
            }

            // The input node must list this node as an output.
            const auto& input_outputs =
                nodes_[input_id].outputs();

            if (std::find(
                    input_outputs.begin(),
                    input_outputs.end(),
                    current_id
                ) == input_outputs.end()) {
                return false;
            }
        }

        // Check output connections.
        for (size_t output_id : current.outputs()) {
            if (output_id >= nodes_.size()) {
                return false;
            }

            if (output_id == current_id) {
                return false;
            }

            // The output node must list this node as an input.
            const auto& output_inputs =
                nodes_[output_id].inputs();

            if (std::find(
                    output_inputs.begin(),
                    output_inputs.end(),
                    current_id
                ) == output_inputs.end()) {
                return false;
            }
        }
    }
    // A valid graph must also be acyclic.
    try {
        (void)execution_order();
    }
    catch (const std::runtime_error&) {
        return false;
    }

    return true;

}




std::vector<size_t> Graph::execution_order() const
{
    std::vector<size_t> in_degree(nodes_.size(), 0);

    for (const Node& current : nodes_)
    {
        in_degree[current.id()] =
            current.inputs().size();
    }

    std::queue<size_t> ready;

    for (size_t i = 0; i < nodes_.size(); ++i)
    {
        if (in_degree[i] == 0)
        {
            ready.push(i);
        }
    }

    std::vector<size_t> order;
    order.reserve(nodes_.size());

    while (!ready.empty())
    {
        size_t current_id = ready.front();
        ready.pop();

        order.push_back(current_id);

        const Node& current = nodes_[current_id];

        for (size_t output_id : current.outputs())
        {
            --in_degree[output_id];

            if (in_degree[output_id] == 0)
            {
                ready.push(output_id);
            }
        }
    }

    if (order.size() != nodes_.size())
    {
        throw std::runtime_error(
            "Graph contains a cycle");
    }

    return order;
}


std::vector<size_t> Graph::input_nodes() const
{
    std::vector<size_t> inputs;

    for (const Node& current : nodes_)
    {
        if (current.inputs().empty())
        {
            inputs.push_back(current.id());
        }
    }

    return inputs;
}




std::vector<size_t> Graph::output_nodes() const
{
    std::vector<size_t> outputs;

    for (const Node& current : nodes_)
    {
        if (current.outputs().empty())
        {
            outputs.push_back(current.id());
        }
    }

    return outputs;
}
