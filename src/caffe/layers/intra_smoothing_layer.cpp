/*
 * intra_smoothing_layer.cpp
 *
 *  Created on: 2015年5月22日
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
void IntraSmoothingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		vector<Blob<Dtype>*>* top) {
}

template<typename Dtype>
void IntraSmoothingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		vector<Blob<Dtype>*>* top) {
	const int seg_num = 200;
	(*top)[0]->Reshape(bottom[0]->num(), seg_num,
			1, 1);
}

template<typename Dtype>
void IntraSmoothingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		vector<Blob<Dtype>*>* top) {
	const int seg_num = 200;
	const Dtype *bottom_data = bottom[0]->cpu_data();
	const Dtype* seg_data = bottom[1]->cpu_data();
	Dtype* top_data = (*top)[0]->mutable_cpu_data();
	const int count = bottom[0]->count();
	const int num = bottom[0]->num();
	const int channel = bottom[0]->channels();
	const int dim = count / num;
	Dtype average_[seg_num];
	int number_[seg_num];
	const int bias_count = seg_num * num;

	for (int n = 0; n < num; n++) {
		for (int i = 0; i < seg_num; i++) {
			average_[i] = 0;
			number_[i] = 0;
		}
		for (int i = 0; i < dim; i++) {
			average_[int(seg_data[bias_count + n * dim + i])] += bottom_data[n * dim
					+ i];
			number_[int(seg_data[bias_count + n * dim + i])]++;
			//LOG(INFO) << seg_data[bias_count + n * dim + i];
		}
		for (int i = 0; i < seg_num; i++) {
			if (number_[i] == 0) LOG(INFO)<<"error: number=0";
			average_[i] /= number_[i];
			top_data[i+n*seg_num] = average_[i];
		}

	}
}

template<typename Dtype>
void IntraSmoothingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {
	if (propagate_down[0]) {
		const Dtype* bottom_data = (*bottom)[0]->cpu_data();
		const Dtype* seg_data = (*bottom)[1]->cpu_data();
		const Dtype* top_diff = top[0]->cpu_diff();
		Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();
		const int count = (*bottom)[0]->count();
		const int num = (*bottom)[0]->num();
		const int channel = (*bottom)[0]->channels();
		const int dim = count / num;
		//const int channel_dim = dim / channel;
		const int seg_num = 200;
		int number_[seg_num];
		const int bias_count = seg_num * num;

		for (int n = 0; n < num; n++) {
			for (int i = 0; i < seg_num; i++) {
				number_[i] = 0;
			}
			for (int i = 0; i < dim; i++) {
				number_[int(seg_data[bias_count + n * dim + i])]++;
			}
			for (int i = 0; i < dim; i++) {
				int tmp = int(seg_data[bias_count + n * dim + i]);
				bottom_diff[n * dim + i] = top_diff[tmp] / number_[tmp];
				//LOG(INFO) << seg_data[bias_count + n * dim + i];
			}

		}
	}
}

#ifdef CPU_ONLY
STUB_GPU(IntraSmoothingLayer);
#endif

INSTANTIATE_CLASS(IntraSmoothingLayer);

}  // namespace caffe

