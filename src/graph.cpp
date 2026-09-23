#include "graph.hpp"

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

    nodes_[from].add_output(to);
    nodes_[to].add_input(from);
}
