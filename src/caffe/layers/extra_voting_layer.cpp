/*
 * extra_voting_layer.cpp
 *
 *  Created on: 2015年5月29日
 *      Author: tchen
 */
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
void ExtraVotingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		vector<Blob<Dtype>*>* top) {
}

template<typename Dtype>
void ExtraVotingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		vector<Blob<Dtype>*>* top) {
	const int seg_num = 200;
	(*top)[0]->Reshape(bottom[0]->num(), seg_num, 1, 1);
}

template<typename Dtype>
void ExtraVotingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		vector<Blob<Dtype>*>* top) {
	const Dtype *bottom_data = bottom[0]->cpu_data();
	const Dtype* seg_data = bottom[1]->cpu_data();
	Dtype* top_data = (*top)[0]->mutable_cpu_data();
	const int count = bottom[0]->count();
	const int num = bottom[0]->num();
	const int dim = count / num;
	const Dtype alph = 0.3;
	const int max_nei = 25;

	for (int n = 0; n < num; n++) {
		for (int i = 0; i < dim; i++) {
			top_data[n * dim + i] = (1 - alph) * bottom_data[n * dim + i];
			for (int j = 1; j <= seg_data[n * dim * max_nei * 2 + i * max_nei * 2]; j++) {
				int tmp_index_nei = seg_data[n * dim * max_nei * 2 + i * max_nei * 2 + j];
				Dtype tmp_weight = seg_data[n * dim * max_nei * 2 + i * max_nei * 2 + max_nei + j];
				top_data[n * dim + i] += alph * bottom_data[n * dim + tmp_index_nei] * tmp_weight;
				//LOG(INFO) << tmp_index_nei << " " << tmp_weight;
			}
		}
	}
}

template<typename Dtype>
void ExtraVotingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {
	if (propagate_down[0]) {
		const Dtype* bottom_data = (*bottom)[0]->cpu_data();
		const Dtype* seg_data = (*bottom)[1]->cpu_data();
		const Dtype* top_diff = top[0]->cpu_diff();
		Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();
		const int count = (*bottom)[0]->count();
		const int num = (*bottom)[0]->num();
		const int dim = count / num;
		const Dtype alph = 0.3;
		const int max_nei = 25;

		for (int n = 0; n < num; n++) {
			for (int i = 0; i < dim; i++) {
				bottom_diff[n * dim + i] = (1 - alph) * top_diff[n * dim + i];
				for (int j = 1;
						j <= seg_data[n * dim * max_nei * 2 + i * max_nei * 2];
						j++) {
					int tmp_index_nei = seg_data[n * dim * max_nei * 2
							+ i * max_nei * 2 + j];
					Dtype tmp_weight = seg_data[n * dim * max_nei * 2
							+ i * max_nei * 2 + max_nei + j];
					bottom_diff[n * dim + i] += alph
							* top_diff[n * dim + tmp_index_nei] * tmp_weight;
				}
			}
		}
	}
}

#ifdef CPU_ONLY
STUB_GPU(ExtraVotingLayer);
#endif

INSTANTIATE_CLASS(ExtraVotingLayer);

}  // namespace caffe

