
#pragma once

#include "graph.hpp"
#include "model.hpp"
#include "tensor.hpp"

#include <vector>

class Executor {
public:
    std::vector<Tensor> run(
        const Graph& graph,
        const std::vector<Tensor>& inputs
    ) const;

    std::vector<Tensor> run(
        const Model& model,
        const std::vector<Tensor>& inputs
    ) const;

private:
    std::vector<Tensor> run_impl(
        const Graph& graph,
        const Model* model,
        const std::vector<Tensor>& inputs
    ) const;
};
