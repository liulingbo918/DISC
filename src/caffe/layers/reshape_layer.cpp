/*
 * reshape_layer.cpp
 *
 *  Created on: 2015年5月17日
 *      Author: tchen
 */

/*
 * intra_smoothing.cpp
 *
 *  Created on: 2015年5月3日
 *      Author: tchen
 */
#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>

namespace caffe {
template<typename Dtype>
void ReshapeLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		vector<Blob<Dtype>*>* top) {
}

template<typename Dtype>
void ReshapeLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		vector<Blob<Dtype>*>* top) {
	(*top)[0]->Reshape(bottom[0]->num(), 1, 64, 64);
}

template<typename Dtype>
void ReshapeLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		vector<Blob<Dtype>*>* top) {
	const Dtype *bottom_data = bottom[0]->cpu_data();
	Dtype* top_data = (*top)[0]->mutable_cpu_data();
	const int count = bottom[0]->count();
	for (int i = 0; i < count; i++)
		top_data[i] = bottom_data[i];
}

template<typename Dtype>
void ReshapeLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {
	const Dtype* top_diff = top[0]->cpu_diff();
	Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();
	const int count = (*bottom)[0]->count();
	for (int i = 0; i < count; i++)
		bottom_diff[i] = top_diff[i];
}

#ifdef CPU_ONLY
STUB_GPU(ReshapeLayer);
#endif

INSTANTIATE_CLASS(ReshapeLayer);

}  // namespace caffe

