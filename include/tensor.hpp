#pragma once

#include <cstddef>
#include <vector>

class Tensor
{
public:
    explicit Tensor(const std::vector<size_t> &shape);

    const std::vector<size_t> &shape() const;

    size_t size() const;

    size_t ndim() const;

    float *data();
    const float *data() const;

    float &operator()(const std::vector<size_t> &indices);
    const float &operator()(const std::vector<size_t> &indices) const;

private:
    size_t compute_offset(
        const std::vector<size_t> &indices) const;

    std::vector<size_t> shape_;
    std::vector<size_t> strides_;
    std::vector<float> data_;
};