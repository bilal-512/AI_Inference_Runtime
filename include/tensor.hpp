#pragma once

#include<iostream>
#include<vector>

class Tensor {
public:
	Tensor(const std::vector<size_t>& shape);
	const std::vector<size_t>& shape() const;
	size_t size() const ;
	float* data() ;
	const float* data() const;
	float& operator()(size_t row, size_t column);
	const float& operator()(size_t row, size_t column) const;
private:
	std::vector<size_t> shape_ ;
	std::vector<float> data_ ;
};
