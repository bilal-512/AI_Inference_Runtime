#pragma once

#include "node.hpp"

#include <cstddef>
#include <string>
#include <vector>

class Graph
{
public:
    size_t addNode(const std::string& operation);

    void connect(size_t from, size_t to);

    const Node& node(size_t id) const;

    Node& node(size_t id);

    size_t size() const;

    bool validate() const;

    std::vector<size_t> execution_order() const;

    std::vector<size_t> input_nodes() const;

    std::vector<size_t> output_nodes() const;

private:
    std::vector<Node> nodes_;
};
