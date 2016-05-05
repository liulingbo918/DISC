/*
 * multi_hinge_loss_norm.cpp
 *
 *  Created on: 2015年5月23日
 *      Author: tchen
 */
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
int rng_count_9 = 0;
template<typename Dtype>
void MultiHingeLossNormLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		vector<Blob<Dtype>*>* top) {
	const Dtype* bottom_data = bottom[0]->cpu_data();
	Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
	const Dtype* label = bottom[1]->cpu_data();

	//LOG(INFO) << bottom[0]->count() << " " <<  bottom[1]->count();
	CHECK_EQ(bottom[0]->count(), bottom[1]->count());
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

#if 1
	if (rng_count_9 % 100 == 0) {
		const int img_size = 64;//bottom[0]->width();
		cv::Mat img__(img_size, img_size, CV_8UC1);
		//cv::resize(cv_img_2, img__, cv::Size(64, 64));
		int i__ = 0;
		for (int h = 0; h < img_size; h++) {
			for (int w = 0; w < img_size; w++) {
				if (bottom_data[i__] <= -1) {
					img__.at<unsigned char>(h, w) = (static_cast<uint8_t>(0));
				} else if (bottom_data[i__] >= 1) {
					img__.at<unsigned char>(h, w) = (static_cast<uint8_t>(255));
				} else {
					img__.at<unsigned char>(h, w) = (static_cast<uint8_t>((1
											+ bottom_data[i__]) * 255 / 2));
				}

				i__++;
			}
		}
		cv::imshow("pre_1", img__);
		i__ = 0;
		for (int h = 0; h < img_size; h++) {
			for (int w = 0; w < img_size; w++) {

				img__.at<unsigned char>(h, w) = label[i__] * 255;
				i__++;
			}
		}
		cv::imshow("label_1", img__);
		cv::waitKey(100);
	}

	rng_count_9 = (rng_count_9 + 1) % 10000;
#endif
	/*switch (this->layer_param_.hinge_loss_param().norm()) {
	 case HingeLossParameter_Norm_L1:
	 loss[0] = caffe_cpu_asum(count, bottom_diff) / num;
	 break;
	 case HingeLossParameter_Norm_L2:
	 loss[0] = caffe_cpu_dot(count, bottom_diff, bottom_diff) / num;
	 break;
	 default:
	 LOG(FATAL) << "Unknown Norm";
	 }*/
}

template<typename Dtype>
void MultiHingeLossNormLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
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

				//if((*bottom)[0]->width() != 1)
				//	bottom_diff[i * dim + j] = 0;  // add by llb
			}
		}

		const Dtype loss_weight = top[0]->cpu_diff()[0];
		caffe_scal(count, loss_weight * 2 / num, bottom_diff);
		/*switch (this->layer_param_.hinge_loss_param().norm()) {
		 case HingeLossParameter_Norm_L1:
		 caffe_cpu_sign(count, bottom_diff, bottom_diff);
		 caffe_scal(count, loss_weight / num, bottom_diff);
		 break;
		 case HingeLossParameter_Norm_L2:
		 caffe_scal(count, loss_weight * 2 / num, bottom_diff);
		 break;
		 default:
		 LOG(FATAL) << "Unknown Norm";
		 }*/
	}
}

INSTANTIATE_CLASS(MultiHingeLossNormLayer);

}  // namespace caffe





