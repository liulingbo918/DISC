/*
 * norm_layer.cpp
 *
 *  Created on: 2015年5月18日
 *      Author: tchen
 */
#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template<typename Dtype>
void SVMNormLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		vector<Blob<Dtype>*>* top) {
	const Dtype* bottom_data = bottom[0]->cpu_data();
	Dtype* top_data = (*top)[0]->mutable_cpu_data();
	const int count = bottom[0]->count();
	Dtype negative_slope = this->layer_param_.relu_param().negative_slope();
	for (int i = 0; i < count; ++i) {
		if (bottom_data[i] < -1)
			top_data[i] = 0;
		else if (bottom_data[i] > 1)
			top_data[i] = 255;
		else
			top_data[i] = 255 * (bottom_data[i] + 1) / 2;
	}
}

template<typename Dtype>
void SVMNormLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {
	if (propagate_down[0]) {
		const Dtype* bottom_data = (*bottom)[0]->cpu_data();
		const Dtype* top_diff = top[0]->cpu_diff();
		Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();
		const int count = (*bottom)[0]->count();
		Dtype negative_slope = this->layer_param_.relu_param().negative_slope();
		for (int i = 0; i < count; ++i) {
			if (bottom_data[i] < -1)
				bottom_diff[i] = 0;
			else if (bottom_data[i] > 1)
				bottom_diff[i] = 0;
			else
				bottom_diff[i] = 255 * top_diff[i] / 2;
				(bottom_data[i] + 1) / 2;
		}
	}
}

#ifdef CPU_ONLY
STUB_GPU(SVMNormLayer);
#endif

INSTANTIATE_CLASS(SVMNormLayer);

}  // namespace caffe

