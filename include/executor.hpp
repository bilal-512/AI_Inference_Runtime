#pragma once

#include "graph.hpp"
#include "tensor.hpp"

#include <vector>

class Executor
{
public:
    std::vector<Tensor> run(
        const Graph& graph,
        const std::vector<Tensor>& inputs
    ) const;
};
