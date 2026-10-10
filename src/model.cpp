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




bool Model::validate() const {
    // 1. The underlying graph must be valid.
    if (!graph_.validate()) {
        return false;
    }

    // 2. Check each node in the graph.
    for (size_t i = 0; i < graph_.size(); ++i) {
        const Node& current = graph_.node(i);

        // 3. Model-aware MatMul nodes reference a named weight.
        if (current.operation() == "MatMul" &&
            current.inputs().size() == 1) {

            if (!current.has_attribute("weight")) {
                return false;
            }

            const std::string& weight_name =
                current.attribute("weight");

            if (parameters_.find(weight_name) == parameters_.end()) {
                return false;
            }

            // A matrix multiplication weight must be rank 2.
            if (parameters_.at(weight_name).ndim() != 2) {
                return false;
            }
        }

        // 4. Model-aware Add nodes reference a named bias.
        if (current.operation() == "Add" &&
            current.inputs().size() == 1) {

            if (!current.has_attribute("bias")) {
                return false;
            }

            const std::string& bias_name =
                current.attribute("bias");

            if (parameters_.find(bias_name) == parameters_.end()) {
                return false;
            }
        }
    }

    return true;
}


