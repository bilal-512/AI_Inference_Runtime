#include "node.hpp"

Node::Node(
    size_t id,
    const std::string& operation
)
    : id_(id),
      operation_(operation)
{
}

size_t Node::id() const
{
    return id_;
}

const std::string& Node::operation() const
{
    return operation_;
}

const std::vector<size_t>& Node::inputs() const
{
    return inputs_;
}

const std::vector<size_t>& Node::outputs() const
{
    return outputs_;
}

void Node::add_input(size_t node_id)
{
    inputs_.push_back(node_id);
}

void Node::add_output(size_t node_id)
{
    outputs_.push_back(node_id);
}
