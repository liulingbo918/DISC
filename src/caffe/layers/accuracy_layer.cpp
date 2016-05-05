#include <algorithm>
#include <functional>
#include <utility>
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
int rng_count_8 = 0;
template<typename Dtype>
void AccuracyLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		vector<Blob<Dtype>*>* top) {
	top_k_ = this->layer_param_.accuracy_param().top_k();
}

template<typename Dtype>
void AccuracyLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		vector<Blob<Dtype>*>* top) {
	CHECK_EQ(bottom[0]->num(), bottom[1]->num())
			<< "The data and label should have the same number.";
	CHECK_LE(top_k_, bottom[0]->count() / bottom[0]->num())
			<< "top_k must be less than or equal to the number of classes.";
	CHECK_EQ(bottom[1]->channels(), 1);
	CHECK_EQ(bottom[1]->height(), 1);
	CHECK_EQ(bottom[1]->width(), 1);
	(*top)[0]->Reshape(1, 1, 1, 1);
}

template<typename Dtype>
void AccuracyLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		vector<Blob<Dtype>*>* top) {
	Dtype accuracy = 0;
	const Dtype* bottom_data = bottom[0]->cpu_data();
	const Dtype* bottom_label = bottom[1]->cpu_data();
	int num = bottom[0]->num();
	int dim = bottom[0]->count() / bottom[0]->num();
	vector<Dtype> maxval(top_k_ + 1);
	vector<int> max_id(top_k_ + 1);
	for (int i = 0; i < num; ++i) {
		// Top-k accuracy
		std::vector < std::pair<Dtype, int> > bottom_data_vector;
		for (int j = 0; j < dim; ++j) {
			bottom_data_vector.push_back(
					std::make_pair(bottom_data[i * dim + j], j));
		}
		std::partial_sort(bottom_data_vector.begin(),
				bottom_data_vector.begin() + top_k_, bottom_data_vector.end(),
				std::greater<std::pair<Dtype, int> >());
		// check if true label is in top k predictions
		for (int k = 0; k < top_k_; k++) {
			if (bottom_data_vector[k].second
					== static_cast<int>(bottom_label[i])) {
				++accuracy;
				break;
			}
		}
	}

	// LOG(INFO) << "Accuracy: " << accuracy;
	(*top)[0]->mutable_cpu_data()[0] = accuracy / num;
	// Accuracy layer should not be used as a loss function.
}

template<typename Dtype>
void HingeAccuracyLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		vector<Blob<Dtype>*>* top) {
}

template<typename Dtype>
void HingeAccuracyLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		vector<Blob<Dtype>*>* top) {
	(*top)[0]->Reshape(1, 3, 1, 1);
}

template<typename Dtype>
void HingeAccuracyLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		vector<Blob<Dtype>*>* top) {
	const Dtype* bottom_data = bottom[0]->cpu_data();
	Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
	const Dtype* label = bottom[1]->cpu_data();
	const int num = bottom[0]->num();
	int count = bottom[0]->count();
	int dim = count / num;
	Dtype loss = 0;
	Dtype accuracy = 0;
	//for(int i = 0; i < count; i++)
	//	LOG(INFO) << label[i];
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
//			/LOG(INFO) << label[i * dim + j];
		}
	}

	for (int i = 0; i < num; i++) {
		for (int j = 0; j < dim; j++) {
			if ((static_cast<int>(label[i * dim + j]) == 1)
					&& bottom_data[i * dim + j] > 0)
				accuracy++;
			if ((static_cast<int>(label[i * dim + j]) == 0)
					&& bottom_data[i * dim + j] <= 0)
				accuracy++;
		}
	}
	loss = caffe_cpu_dot(count, bottom_diff, bottom_diff) / num;

	// LOG(INFO) << "Accuracy: " << accuracy;
	(*top)[0]->mutable_cpu_data()[0] = accuracy / count;
	(*top)[0]->mutable_cpu_data()[1] = loss;
	// Accuracy layer should not be used as a loss function.
	const int seg_num =200;
	const int label_size = 64;
	Dtype pre_data[num * label_size * label_size];
	for (int n = 0; n < num; n++) {
		for (int i = 0; i < label_size * label_size; i++) {
			int index = label[num * seg_num + n * label_size * label_size + i];
			pre_data[n * label_size * label_size + i] = bottom_data[n * seg_num + index];
		}
	}
	accuracy = 0;
	for (int i = 0; i < num * label_size * label_size; i++) {
		if ((static_cast<int>(label[num * (seg_num + label_size * label_size)
				+ i]) == 1) && pre_data[i] > 0)
			accuracy++;
		if ((static_cast<int>(label[num * (seg_num + label_size * label_size)
				+ i]) == 0) && pre_data[i] <= 0)
			accuracy++;
	}
	(*top)[0]->mutable_cpu_data()[2] = accuracy
			/ (num * label_size * label_size);
#if 1
	if (rng_count_8 % 10 == 0) {
		const int img_size = 64;
		cv::Mat img__(img_size, img_size, CV_8UC1);
		//cv::resize(cv_img_2, img__, cv::Size(64, 64));
		int i__ = 0;
		for (int h = 0; h < img_size; h++) {
			for (int w = 0; w < img_size; w++) {
				if (pre_data[i__] <= -1) {
					img__.at<unsigned char>(h, w) = (static_cast<uint8_t>(0));
				} else if (pre_data[i__] >= 1) {
					img__.at<unsigned char>(h, w) = (static_cast<uint8_t>(255));
				} else {
					img__.at<unsigned char>(h, w) = (static_cast<uint8_t>((1
											+ pre_data[i__]) * 255 / 2));
				}

				i__++;
			}
		}
		cv::imshow("pre", img__);
		i__ = 0;
		for (int h = 0; h < img_size; h++) {
			for (int w = 0; w < img_size; w++) {

				img__.at<unsigned char>(h, w) = label[num
				* (seg_num + label_size * label_size) + i__] * 255;
				i__++;
			}
		}
		cv::imshow("label", img__);
		cv::waitKey(100);
	}

	rng_count_8 = (rng_count_8 + 1) % 10000;
#endif

#if 0
	vector < std::string > filenames;
	std::string filename;
	std::ifstream file("/home/tchen/My_Project/DISC/datasets/ECSSD/ECSSD_total.list");
	while (file >> filename) {
		filenames.push_back(filename + +"_g_seg_1.png");
	}

	const int img_size = 64;
	cv::Mat img__(img_size, img_size, CV_8UC1);
	//cv::resize(cv_img_2, img__, cv::Size(64, 64));
	int i__ = 0;
	for (int h = 0; h < img_size; h++) {
		for (int w = 0; w < img_size; w++) {
			if (pre_data[i__] <= -1) {
				img__.at<unsigned char>(h, w) = (static_cast<uint8_t>(0));
			} else if (pre_data[i__] >= 1) {
				img__.at<unsigned char>(h, w) = (static_cast<uint8_t>(255));
			} else {
				img__.at<unsigned char>(h, w) = (static_cast<uint8_t>((1
						+ pre_data[i__]) * 255 / 2));
			}

			i__++;
		}
	}
	cv::imshow("pre", img__);
	cv::waitKey(100);
	LOG(INFO) << "/home/tchen/My_Project/DISC/datasets/ECSSD/ECSSD_result/" + filenames[rng_count_8];
	cv::imwrite("/home/tchen/My_Project/DISC/datasets/ECSSD/ECSSD_result/" + filenames[rng_count_8], img__);
	rng_count_8++;
#endif

}

template<typename Dtype>
void HingeNormAccuracyLayer<Dtype>::LayerSetUp(
		const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
}

template<typename Dtype>
void HingeNormAccuracyLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		vector<Blob<Dtype>*>* top) {
	(*top)[0]->Reshape(1, 2, 1, 1);
}

template<typename Dtype>
void HingeNormAccuracyLayer<Dtype>::Forward_cpu(
		const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {

	(*top)[0]->mutable_cpu_data()[0] = 0;
	(*top)[0]->mutable_cpu_data()[1] = 0;
	return;

	const Dtype* bottom_data = bottom[0]->cpu_data();
	Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
	const Dtype* label = bottom[1]->cpu_data();
	const int num = bottom[0]->num();
	int count = bottom[0]->count();
	int dim = count / num;
	Dtype loss = 0;
	Dtype accuracy = 0;
	//for(int i = 0; i < count; i++)
	//	LOG(INFO) << label[i];
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
//			/LOG(INFO) << label[i * dim + j];
		}
	}

	for (int i = 0; i < num; i++) {
		for (int j = 0; j < dim; j++) {
			if ((static_cast<int>(label[i * dim + j]) == 1)
					&& bottom_data[i * dim + j] > 0)
				accuracy++;
			if ((static_cast<int>(label[i * dim + j]) == 0)
					&& bottom_data[i * dim + j] <= 0)
				accuracy++;
		}
	}
	loss = caffe_cpu_dot(count, bottom_diff, bottom_diff) / num;

	// LOG(INFO) << "Accuracy: " << accuracy;
	(*top)[0]->mutable_cpu_data()[0] = accuracy / count;
	(*top)[0]->mutable_cpu_data()[1] = loss;
	// Accuracy layer should not be used as a loss function.

#if 0
	if (rng_count_8 % 100 == 0) {
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
		cv::imshow("pre", img__);
		i__ = 0;
		for (int h = 0; h < img_size; h++) {
			for (int w = 0; w < img_size; w++) {

				img__.at<unsigned char>(h, w) = label[i__] * 255;
				i__++;
			}
		}
		cv::imshow("label", img__);
		cv::waitKey(100);
	}

	rng_count_8 = (rng_count_8 + 1) % 10000;
#endif

}

/*template<typename Dtype>
 void SigmoidAccuracyLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
 vector<Blob<Dtype>*>* top) {
 }

 template<typename Dtype>
 void SigmoidAccuracyLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
 vector<Blob<Dtype>*>* top) {
 (*top)[0]->Reshape(1, 2, 1, 1);
 }

 template<typename Dtype>
 void SigmoidAccuracyLayer<Dtype>::Forward_cpu(
 const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
 Dtype loss = 0;
 Dtype accuracy = 0;

 const int count = bottom[0]->count();
 const int num = bottom[0]->num();
 // Stable version of loss computation from input data
 const Dtype* input_data = bottom[0]->cpu_data();
 const Dtype* target = bottom[1]->cpu_data();
 for (int i = 0; i < count; ++i) {
 loss -= input_data[i] * (target[i] - (input_data[i] >= 0)) -
 log(1 + exp(input_data[i] - 2 * input_data[i] * (input_data[i] >= 0)));
 }
 for (int i = 0; i < count; i++) {
 if ((static_cast<int>(target[i]) == 1)
 && input_data[i] > 0)
 accuracy++;
 if ((static_cast<int>(target[i]) == 0)
 && input_data[i] <= 0)
 accuracy++;
 }
 // LOG(INFO) << "Accuracy: " << accuracy;
 (*top)[0]->mutable_cpu_data()[0] = accuracy / count;
 (*top)[0]->mutable_cpu_data()[1] = loss / num;
 // Accuracy layer should not be used as a loss function.
 }*/

INSTANTIATE_CLASS(AccuracyLayer);
INSTANTIATE_CLASS(HingeAccuracyLayer);
INSTANTIATE_CLASS(HingeNormAccuracyLayer);

}	// namespace caffe
