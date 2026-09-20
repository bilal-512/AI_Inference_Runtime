#include<tensor.hpp>

#include<numeric>
#include<stdexcept>

Tensor::Tensor(const std::vector<size_t>& shape) 
	: shape_(shape) {
	if (shape.empty()) {
		throw std::invalid_argument(" Tensor can not be Empty");
	}
	size_t total_elements = 1;
	for (size_t dimention: shape) {
	if (dimention == 0) {
		throw std::invalid_argument("Tensor dimention can be zero");
	}
	total_elements*= dimention;
	}
	data_.resize(total_elements);
}

const std::vector<size_t>& Tensor::shape() const {
	return shape_;
}

size_t Tensor::size() const {
	return data_.size();
}

float* Tensor::data() {
	return data_.data();
}

const float* Tensor::data() const {
 return data_.data();
}

float& Tensor::operator()(size_t row, size_t column){
	return data_[row * shape_[1] + column];
}

const float& Tensor::operator()(size_t row, size_t column)const {
	return data_[row * shape_[1]+column];
}

