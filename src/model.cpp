#include "model.hpp"

#include <stdexcept>

Graph& Model::graph() {
    return graph_;
}

const Graph& Model::graph() const {
    return graph_;
}

void Model::add_parameter(
    const std::string& name,
    const Tensor& tensor
) {
    if (name.empty()) {
        throw std::invalid_argument(
            "Parameter name cannot be empty"
        );
    }

    if (parameters_.find(name) != parameters_.end()) {
        throw std::invalid_argument(
            "Parameter already exists: " + name
        );
    }

    parameters_.emplace(name, tensor);
}

const Tensor& Model::parameter(
    const std::string& name
) const {
    auto it = parameters_.find(name);

    if (it == parameters_.end()) {
        throw std::out_of_range(
            "Parameter not found: " + name
        );
    }

    return it->second;
}
