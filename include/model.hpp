#pragma once

#include "graph.hpp"
#include "tensor.hpp"

#include <string>
#include <unordered_map>

class Model {
public:
    Graph& graph();

    const Graph& graph() const;

    void add_parameter(
        const std::string& name,
        const Tensor& tensor
    );

    const Tensor& parameter(
        const std::string& name
    ) const;

private:
    Graph graph_;

    std::unordered_map<
        std::string,
        Tensor
    > parameters_;
};
