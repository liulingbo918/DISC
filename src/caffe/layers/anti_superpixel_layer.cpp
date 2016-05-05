/*
 * anti_superpixel_layer.cpp
 *
 *  Created on: 2015年5月26日
 *      Author: tchen
 */
/*
 * intra_smoothing_layer.cpp
 *
 *  Created on: 2015年5月22日
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
void AntiSuperpixelLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		vector<Blob<Dtype>*>* top) {
}

template<typename Dtype>
void AntiSuperpixelLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		vector<Blob<Dtype>*>* top) {
	const int map_size = 64;
	(*top)[0]->Reshape(bottom[0]->num(), 1, map_size, map_size);
}

template<typename Dtype>
void AntiSuperpixelLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		vector<Blob<Dtype>*>* top) {
	const Dtype *bottom_data = bottom[0]->cpu_data();
	const Dtype* seg_data = bottom[1]->cpu_data();
	Dtype* top_data = (*top)[0]->mutable_cpu_data();
	const int count = bottom[0]->count();
	const int num = bottom[0]->num();
	const int dim = count / num;

	const int top_count = (*top)[0]->count();
	const int top_num = (*top)[0]->num();
	const int top_dim = top_count / top_num;
	const int bias_count = count;

	for (int n = 0; n < num; n++) {
		for (int i = 0; i < top_dim; i++) {
			int tmp = int(seg_data[bias_count + n * top_dim + i]);
			top_data[n * top_dim + i] = bottom_data[n * dim + tmp];
		}
	}
}

template<typename Dtype>
void AntiSuperpixelLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {
	if (propagate_down[0]) {
		//const Dtype* bottom_data = (*bottom)[0]->cpu_data();
		const Dtype* seg_data = (*bottom)[1]->cpu_data();
		const Dtype* top_diff = top[0]->cpu_diff();
		Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();
		const int count = (*bottom)[0]->count();
		const int num = (*bottom)[0]->num();
		const int dim = count / num;

		const int top_count = top[0]->count();
		const int top_num = top[0]->num();
		const int top_dim = top_count / top_num;
		const int bias_count = count;

		for(int i = 0; i < count; i++)
			bottom_diff[i] = 0;

		for (int n = 0; n < num; n++) {
			for (int i = 0; i < top_dim; i++) {
				int tmp = int(seg_data[bias_count + n * top_dim + i]);
				bottom_diff[n * dim + tmp] += top_diff[n * top_dim + i];;
			}
		}
	}
}

#ifdef CPU_ONLY
STUB_GPU(AntiSuperpixelLayer);
#endif

INSTANTIATE_CLASS(AntiSuperpixelLayer);

}  // namespace caffe

