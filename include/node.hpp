
#pragma once

#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>

class Node {
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

    // Store a named attribute.
    void set_attribute(
        const std::string& name,
        const std::string& value
    );

    // Retrieve an attribute by name.
    const std::string& attribute(
        const std::string& name
    ) const;

    // Check whether an attribute exists.
    bool has_attribute(
        const std::string& name
    ) const;

private:
    size_t id_;
    std::string operation_;

    std::vector<size_t> inputs_;
    std::vector<size_t> outputs_;

    std::unordered_map<std::string, std::string>
        attributes_;
};
