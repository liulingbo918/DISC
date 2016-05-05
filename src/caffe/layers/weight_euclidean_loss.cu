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

template<typename Dtype>
void WeightEuclideanLossLayer<Dtype>::Forward_gpu(
		const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
	int count = bottom[0]->count();
	caffe_gpu_sub(count, bottom[0]->gpu_data(), bottom[1]->gpu_data(),
			diff_.mutable_gpu_data());

	// scale the diff
	Dtype weights[3] = { 0, 0, 1 };
	int height_mul_width = bottom[0]->height() * bottom[0]->width();
	for (int i = 0; i < bottom[1]->num(); i++) {
		for (int j = 0; j < 3; j++) {
			//LOG(INFO)<<i<<" "<<j;
			caffe_gpu_scal(height_mul_width, weights[j],
					diff_.mutable_gpu_data() + diff_.offset(i, j));
		}
	}

	Dtype dot;
	caffe_gpu_dot(count, diff_.gpu_data(), diff_.gpu_data(), &dot);

	// pixel level loss
	//Dtype loss = dot / bottom[0]->num() / Dtype(2);
	Dtype loss = dot / bottom[0]->count();

	(*top)[0]->mutable_cpu_data()[0] = loss;

//	int count = bottom[0]->count();
//
//	const Dtype *label_data = bottom[1]->cpu_data();
//	Dtype *pseudo_label_data = pseudo_label_.mutable_cpu_data();
//
////	for (int i = 0; i < bottom[1]->count(); i++){
////		LOG(INFO)<<"DEBUG:"<<i<<" "<<bottom[1]->cpu_data()[i];
////	}
//
//	int height_mul_width = bottom[0]->height() * bottom[0]->width();
//	//LOG(INFO)<<height_mul_width;
//	for (int i = 0; i < bottom[0]->num(); i++) {
//		//LOG(INFO)<<*(label_data + bottom[0]->offset(i, 0));
//		caffe_copy(height_mul_width, label_data + bottom[1]->offset(i) + 2 * height_mul_width, // move to the last channel
//				pseudo_label_data + pseudo_label_.offset(i));
//	}
//	//
//
//	caffe_gpu_sub(count, bottom[0]->gpu_data(), pseudo_label_.gpu_data(),
//			diff_.mutable_gpu_data());
//	Dtype dot;
//	caffe_gpu_dot(count, diff_.gpu_data(), diff_.gpu_data(), &dot);
//
//	// pixel level loss
//	//Dtype loss = dot / bottom[0]->num() / Dtype(2);
//	Dtype loss = dot / bottom[0]->count();
//
//	(*top)[0]->mutable_cpu_data()[0] = loss;

//	bool visualize = (rand() % 200 == 0);
//	bool resizeTo500 = true;
//	bool isColor = false;
//
//	visualize = false;
//	if (visualize) {
//		if (isColor) {
//			//int label_size = 38;
//			int height = bottom[0]->height();
//			int width = bottom[0]->width();
//
//			// show the gt
//			const Dtype* gt = bottom[1]->cpu_data();
//			cv::Mat gt_img(height, width, CV_8UC3);
//			int offset2 = 0;
//			for (int c = 0; c < 3; ++c) {
//				for (int h = 0; h < gt_img.rows; ++h) {
//					for (int w = 0; w < gt_img.cols; ++w) {
//						gt_img.at<cv::Vec3b>(h, w)[c] =
//								(MIN(255, MAX(0, gt[offset2])));
//						offset2++;
//					}
//				}
//			}
//
//			if (resizeTo500) {
//				cv::Mat gt_img_resize(500, 500, CV_8UC3);
//				cv::resize(gt_img, gt_img_resize, cv::Size(500, 500));
//				cv::imshow("gt_train", gt_img_resize);
//				cv::waitKey(1000);
//			} else {
//				cv::imshow("gt_train", gt_img);
//				cv::waitKey(1000);
//			}
//
//			// show the predict
//			const Dtype* pred = bottom[0]->cpu_data();
//			cv::Mat pred_img(height, width, CV_8UC3);
//			int offset = 0;
//			for (int c = 0; c < 3; ++c) {
//				for (int h = 0; h < pred_img.rows; ++h) {
//					for (int w = 0; w < pred_img.cols; ++w) {
//						pred_img.at<cv::Vec3b>(h, w)[c] =
//								(MIN(255, MAX(0, pred[offset])));
//						offset++;
//					}
//				}
//			}
//			if (resizeTo500) {
//				cv::Mat pred_img_resize(500, 500, CV_8UC3);
//				cv::resize(pred_img, pred_img_resize, cv::Size(500, 500));
//				cv::imshow("predict_train", pred_img_resize);
//				cv::waitKey(30);
//			} else {
//				cv::imshow("predict_train", pred_img);
//				cv::waitKey(30);
//			}
//		} else {
//			//int label_size = 38;
//			int height = bottom[0]->height();
//			int width = bottom[0]->width();
//
//			// show the gt
//			const Dtype* gt = pseudo_label_.cpu_data();
//			cv::Mat gt_img(height, width, CV_8UC1);
//			int offset2 = 0;
//			for (int h = 0; h < gt_img.rows; ++h) {
//				for (int w = 0; w < gt_img.cols; ++w) {
//					gt_img.at<uchar>(h, w) = (MIN(255, MAX(0, gt[offset2])));
//					offset2++;
//				}
//			}
//
//			if (resizeTo500) {
//				cv::Mat gt_img_resize(500, 500, CV_8UC1);
//				cv::resize(gt_img, gt_img_resize, cv::Size(500, 500));
//				cv::imshow("gt_train", gt_img_resize);
//				cv::waitKey(1000);
//			} else {
//				cv::imshow("gt_train", gt_img);
//				cv::waitKey(1000);
//			}
//
//			// show the predict
//			const Dtype* pred = bottom[0]->cpu_data();
//			cv::Mat pred_img(height, width, CV_8UC1);
//			int offset = 0;
//			for (int h = 0; h < pred_img.rows; ++h) {
//				for (int w = 0; w < pred_img.cols; ++w) {
//					pred_img.at<uchar>(h, w) = (MIN(255, MAX(0, pred[offset])));
//					offset++;
//				}
//			}
//			if (resizeTo500) {
//				cv::Mat pred_img_resize(500, 500, CV_8UC1);
//				cv::resize(pred_img, pred_img_resize, cv::Size(500, 500));
//				cv::imshow("predict_train", pred_img_resize);
//				cv::waitKey(30);
//			} else {
//				cv::imshow("predict_train", pred_img);
//				cv::waitKey(30);
//			}
//		}
//	}
}

template<typename Dtype>
void WeightEuclideanLossLayer<Dtype>::Backward_gpu(
		const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
		vector<Blob<Dtype>*>* bottom) {
	for (int i = 0; i < 2; ++i) {
		if (propagate_down[i]) {
			const Dtype sign = (i == 0) ? 1 : -1;
			const Dtype alpha = sign * top[0]->cpu_diff()[0]
					/ (*bottom)[i]->num();
			caffe_gpu_axpby((*bottom)[i]->count(),              // count
					alpha,                              // alpha
					diff_.gpu_data(),                   // a
					Dtype(0),                           // beta
					(*bottom)[i]->mutable_gpu_diff());  // b
		}
	}
}

INSTANTIATE_CLASS(WeightEuclideanLossLayer);

}  // namespace caffe
