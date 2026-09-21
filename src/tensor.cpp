#include "tensor.hpp"

#include <stdexcept>

Tensor::Tensor(const std::vector<size_t> &shape)
    : shape_(shape)
{

    if (shape.empty())
    {
        throw std::invalid_argument(
            "Tensor shape cannot be empty");
    }

    size_t total_elements = 1;

    for (size_t dimension : shape)
    {
        if (dimension == 0)
        {
            throw std::invalid_argument(
                "Tensor dimensions cannot be zero");
        }

        total_elements *= dimension;
    }

    data_.resize(total_elements);

    strides_.resize(shape_.size());

    size_t stride = 1;

    for (size_t i = shape_.size(); i-- > 0;)
    {
        strides_[i] = stride;
        stride *= shape_[i];
    }
}

const std::vector<size_t> &Tensor::shape() const
{
    return shape_;
}

size_t Tensor::size() const
{
    return data_.size();
}

size_t Tensor::ndim() const
{
    return shape_.size();
}

float *Tensor::data()
{
    return data_.data();
}

const float *Tensor::data() const
{
    return data_.data();
}

size_t Tensor::compute_offset(
    const std::vector<size_t> &indices) const
{

    if (indices.size() != shape_.size())
    {
        throw std::invalid_argument(
            "Number of indices must match tensor dimensions");
    }

    size_t offset = 0;

    for (size_t i = 0; i < indices.size(); ++i)
    {

        if (indices[i] >= shape_[i])
        {
            throw std::out_of_range(
                "Tensor index out of bounds");
        }

        offset += indices[i] * strides_[i];
    }

    return offset;
}

float &Tensor::operator()(
    const std::vector<size_t> &indices)
{
    return data_[compute_offset(indices)];
}

const float &Tensor::operator()(
    const std::vector<size_t> &indices) const
{
    return data_[compute_offset(indices)];
}