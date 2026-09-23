#pragma once

#include <cstddef>
#include <string>
#include <vector>

class Node
{
public:
    Node(
        size_t id,
        const std::string& operation
    );

    size_t id() const;

    const std::string& operation() const;

    const std::vector<size_t>& inputs() const;

    const std::vector<size_t>& outputs() const;

    void add_input(size_t node_id);

    void add_output(size_t node_id);

private:
    size_t id_;

    std::string operation_;

    std::vector<size_t> inputs_;

    std::vector<size_t> outputs_;
};
