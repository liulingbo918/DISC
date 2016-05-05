/*
 * extra_vote_layer.cpp
 *
 *  Created on: 2015年5月14日
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
int count_5 = 0;
template<typename Dtype>
void ExtraVoteLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		vector<Blob<Dtype>*>* top) {
}

template<typename Dtype>
void ExtraVoteLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		vector<Blob<Dtype>*>* top) {
	(*top)[0]->Reshape(bottom[0]->num(), bottom[0]->channels(),
			bottom[0]->height(), bottom[0]->width());
}

template<typename Dtype>
void ExtraVoteLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		vector<Blob<Dtype>*>* top) {
	const int seg_num = 500;
	//LOG(INFO) << "eeeee";
	const Dtype *bottom_data = bottom[0]->cpu_data();
	const Dtype* seg_data = bottom[1]->cpu_data();
	const Dtype* cos_data = bottom[2]->cpu_data();
	Dtype* top_data = (*top)[0]->mutable_cpu_data();
	const int count = bottom[0]->count();
	const int num = bottom[0]->num();
	const int channel = bottom[0]->channels();
	const int dim = count / num;
	const int channel_dim = dim / channel;
	Dtype average_[seg_num];
	Dtype average_1[seg_num];
	int number_[seg_num];
	const Dtype alph = 0.1;
	const int max_num = 15;
	//for(int i = 0; i < bottom[2]->count() / 2; i++){
	//	LOG(INFO) << i << " / " << bottom[2]->count() << " " << cos_data[i] << " " << cos_data[seg_num * max_num + i];
	//}
	for (int n = 0; n < num; n++) {
		for (int c = 0; c < channel; c++) {
			for (int i = 0; i < seg_num; i++) {
				average_[i] = 0;
				number_[i] = 0;
			}
			for (int i = 0; i < channel_dim; i++) {
				//LOG(INFO) << i << "  " << int(seg_data[i]);
				//if(int(seg_data[i]) > 255)
				//LOG(INFO) << int(seg_data[i]);
				average_[int(seg_data[i])] += bottom_data[n * dim
						+ channel_dim * c + i];
				number_[int(seg_data[i])]++;
			}
			for (int i = 0; i < seg_num; i++) {
				if (number_[i] != 0)
					average_[i] /= number_[i];
				//LOG(INFO) << average_1[i];
				//LOG(INFO) << i << " " << average_[i] << " " << number_[i];
			}

			for (int i = 1; i < seg_num; i++) {
				average_1[i] = (1 - alph) * average_[i];
				//LOG(INFO) << average_[i];
				for (int j = 0; j < max_num; j++) {
					if (int(cos_data[i * max_num + j]) != 0) {
						average_1[i] += alph
								* cos_data[seg_num * max_num + (i - 1) * max_num + j]
								* average_[int(cos_data[(i - 1) * max_num + j])];
						//LOG(INFO) << j << " "<< cos_data[seg_num * max_num + i * max_num + j] << " " << average_[int(cos_data[i * max_num + j])];
						//LOG(INFO) << cos_data[seg_num * max_num + i * max_num + j] << " " << cos_data[i * max_num + j];
					} else
						break;
				}
				//if(average_1[i] == average_[i])
				//	LOG(INFO) << "Error";
				//cv::waitKey(0);
			}
			for (int i = 0; i < channel_dim; i++) {
				top_data[n * dim + channel_dim * c + i] = average_1[int(
						seg_data[i])];
				//LOG(INFO) << top_data[i] << " " << bottom[1]->count();
			}
		}
	}

#if 1
	if (count_5 % 100 == 0) {
		const int img_size = 128;
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
				if (top_data[i__] <= -1) {
					img__.at<unsigned char>(h, w) = (static_cast<uint8_t>(0));
				} else if (top_data[i__] >= 1) {
					img__.at<unsigned char>(h, w) = (static_cast<uint8_t>(255));
				} else {
					img__.at<unsigned char>(h, w) = (static_cast<uint8_t>((1
							+ top_data[i__]) * 255 / 2));
				}

				i__++;
			}
		}
		cv::imshow("label_1", img__);
		cv::waitKey(100);
	}

	count_5 = (count_5 + 1) % 10000;
#endif
}

template<typename Dtype>
void ExtraVoteLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {
	if (propagate_down[0]) {
		//const Dtype* bottom_data = (*bottom)[0]->cpu_data();
		const Dtype* seg_data = (*bottom)[1]->cpu_data();
		const Dtype* cos_data = (*bottom)[2]->cpu_data();
		const Dtype* top_diff = top[0]->cpu_diff();
		Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();
		const int count = (*bottom)[0]->count();
		const int num = (*bottom)[0]->num();
		const int channel = (*bottom)[0]->channels();
		const int dim = count / num;
		const int channel_dim = dim / channel;
		const int seg_num = 500;
		//int number_[seg_num];
		Dtype seg_diff[seg_num];
		const Dtype alph = 0.1;
		const int max_num = 15;

		for (int n = 0; n < num; n++) {
			for (int c = 0; c < channel; c++) {
				for (int i = 0; i < seg_num; i++) {
					seg_diff[i] = 1 - alph;
				}

				for (int i = 0; i < seg_num; i++) {
					for (int j = 0; j < max_num; j++) {
						if (int(cos_data[i * max_num + j]) != 0) {
							seg_diff[int(cos_data[i * max_num + j])] += alph
									* cos_data[seg_num * max_num + i * max_num
											+ j];
						} else
							break;
					}
				}

				for (int i = 0; i < channel_dim; i++) {
					bottom_diff[n * dim + channel_dim * c + i] = top_diff[n
							* dim + channel_dim * c + i]
							* seg_diff[int(seg_data[i])];
				}
			}
		}
	}
}

#ifdef CPU_ONLY
STUB_GPU(ExtraVoteLayer);
#endif

INSTANTIATE_CLASS(ExtraVoteLayer);

}  // namespace caffe

