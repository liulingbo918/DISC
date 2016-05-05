/*
 * multi_hinge_loss_layer.cpp
 *
 *  Created on: 2015年5月3日
 *      Author: tchen
 */
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>

namespace caffe {
int rng_count = 0;
template<typename Dtype>
void MultiHingeLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		vector<Blob<Dtype>*>* top) {
	const Dtype* bottom_data = bottom[0]->cpu_data();
	Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
	const Dtype* label = bottom[1]->cpu_data();
	int num = bottom[0]->num();
	int count = bottom[0]->count();
	int dim = count / num;

	//for(int i = 0; i < count; i++){
	//	LOG(INFO) << bottom_data[i] << " " << label[i];
	//}
	caffe_copy(count, bottom_data, bottom_diff);
	for (int i = 0; i < num; ++i) {
		for (int j = 0; j < dim; j++) {
			if (static_cast<int>(label[i * dim + j]) == 1)
				bottom_diff[i * dim + j] *= -1;
			//LOG(INFO) << num << " " << i * dim + j << " " << label[i * dim + j];
		}
	}
	for (int i = 0; i < num; ++i) {
		for (int j = 0; j < dim; ++j) {
			bottom_diff[i * dim + j] = std::max(Dtype(0),
					1 + bottom_diff[i * dim + j]);
			//LOG(INFO) << label[i * dim + j];
		}
	}
	Dtype* loss = (*top)[0]->mutable_cpu_data();
	//LOG(INFO) << loss << " " << count;
	loss[0] = caffe_cpu_dot(count, bottom_diff, bottom_diff) / num;

	const int seg_num = 200;
	const int label_size = 64;
	Dtype pre_data[num * label_size * label_size];
	for (int n = 0; n < num; n++) {
		for (int i = 0; i < label_size * label_size; i++) {
			int index = label[num * seg_num + n * label_size * label_size + i];
			pre_data[n * label_size * label_size + i] = bottom_data[n * seg_num + index];
		}
	}

#if 1
	if (rng_count % 100 == 0) {
		const int img_size = 64;
		cv::Mat img__(img_size, img_size, CV_8UC1);
		//cv::resize(cv_img_2, img__, cv::Size(64, 64));
		int i__ = 0;
		for (int h = 0; h < img_size; h++) {
			for (int w = 0; w < img_size; w++) {
				if (pre_data[label_size * label_size * 0  + i__] <= -1) {
					img__.at<unsigned char>(h, w) = (static_cast<uint8_t>(0));
				} else if (pre_data[label_size * label_size * 0 + i__] >= 1) {
					img__.at<unsigned char>(h, w) = (static_cast<uint8_t>(255));
				} else {
					img__.at<unsigned char>(h, w) = (static_cast<uint8_t>((1
											+ pre_data[label_size * label_size* 0 + i__]) * 255 / 2));
				}

				i__++;
			}
		}
		cv::imshow("pre_1", img__);
		i__ = 0;
		for (int h = 0; h < img_size; h++) {
			for (int w = 0; w < img_size; w++) {

				img__.at<unsigned char>(h, w) = label[num * (seg_num + label_size * label_size) + label_size * label_size * 0+ i__] * 255;
				i__++;
			}
		}
		cv::imshow("label_11", img__);
		cv::waitKey(100);
	}

	rng_count = (rng_count + 1) % 10000;
#endif
}

template<typename Dtype>
void MultiHingeLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {
	if (propagate_down[1]) {
		LOG(FATAL) << this->type_name()
				<< " Layer cannot backpropagate to label inputs.";
	}
	if (propagate_down[0]) {
		Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();
		const Dtype* label = (*bottom)[1]->cpu_data();
		int num = (*bottom)[0]->num();
		int count = (*bottom)[0]->count();
		int dim = count / num;

		for (int i = 0; i < num; ++i) {
			for (int j = 0; j < dim; ++j) {
				if (static_cast<int>(label[i * dim + j]) == 1)
					bottom_diff[i * dim + j] *= -1;
			}
		}

		const Dtype loss_weight = top[0]->cpu_diff()[0];
		caffe_scal(count, loss_weight * 2 / num, bottom_diff);
	}
}

INSTANTIATE_CLASS(MultiHingeLossLayer);

}  // namespace caffe

