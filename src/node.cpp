
#include "node.hpp"

#include <stdexcept>

Node::Node(
    size_t id,
    const std::string& operation
)
    : id_(id),
      operation_(operation) {
}

size_t Node::id() const {
    return id_;
}

const std::string& Node::operation() const {
    return operation_;
}

const std::vector<size_t>& Node::inputs() const {
    return inputs_;
}

const std::vector<size_t>& Node::outputs() const {
    return outputs_;
}

void Node::add_input(size_t node_id) {
    inputs_.push_back(node_id);
}

void Node::add_output(size_t node_id) {
    outputs_.push_back(node_id);
}

void Node::set_attribute(
    const std::string& name,
    const std::string& value
) {
    if (name.empty()) {
        throw std::invalid_argument(
            "Attribute name cannot be empty"
        );
    }

    attributes_[name] = value;
}

const std::string& Node::attribute(
    const std::string& name
) const {
    auto it = attributes_.find(name);

    if (it == attributes_.end()) {
        throw std::out_of_range(
            "Attribute not found: " + name
        );
    }

    return it->second;
}

bool Node::has_attribute(
    const std::string& name
) const {
    return attributes_.find(name) !=
           attributes_.end();
}
