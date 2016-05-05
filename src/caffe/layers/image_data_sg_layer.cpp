/*
 * image_data_sg_layer.cpp
 *
 *  Created on: 2015年5月3日
 *      Author: tchen
 */
#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "caffe/data_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>

namespace caffe {
int count_3 = 0;
template<typename Dtype>
ImageDataSGLayer<Dtype>::~ImageDataSGLayer<Dtype>() {
	this->JoinPrefetchThread();
}

template<typename Dtype>
void ImageDataSGLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
		vector<Blob<Dtype>*>* top) {
	const int new_height = this->layer_param_.image_data_param().new_height();
	const int new_width = this->layer_param_.image_data_param().new_width();
	CHECK(
			(new_height == 0 && new_width == 0)
					|| (new_height > 0 && new_width > 0))
			<< "Current implementation requires "
					"new_height and new_width to be set at the same time.";
	// Read the file with filenames and labels
	const string& source = this->layer_param_.image_data_param().source();
	LOG(INFO) << "Opening file " << source;
	std::ifstream infile(source.c_str());
	string filename;
	while (infile >> filename) {
		lines_.push_back(std::make_pair(filename, filename + ".png"));
	}

	if (this->layer_param_.image_data_param().shuffle()) {
		// randomly shuffle data
		LOG(INFO) << "Shuffling data";
		const unsigned int prefetch_rng_seed = caffe_rng_rand();
		prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
		ShuffleImages();
	}
	LOG(INFO) << "A total of " << lines_.size() << " images.";

	lines_id_ = 0;
	// Check if we would need to randomly skip a few data points
	if (this->layer_param_.image_data_param().rand_skip()) {
		unsigned int skip = caffe_rng_rand()
				% this->layer_param_.image_data_param().rand_skip();
		LOG(INFO) << "Skipping first " << skip << " data points.";
		CHECK_GT(lines_.size(), skip) << "Not enough points to skip";
		lines_id_ = skip;
	}
	// Read a data point, and use it to initialize the top blob.
	Datum datum;
	CHECK(
			ReadImageToDatumSG(lines_[lines_id_].first,
					lines_[lines_id_].second, new_height, new_width, &datum));
	// image
	const int crop_size = this->layer_param_.transform_param().crop_size();
	const int batch_size = this->layer_param_.image_data_param().batch_size();
	if (crop_size > 0) {
		(*top)[0]->Reshape(batch_size, datum.channels(), crop_size, crop_size);
		this->prefetch_data_.Reshape(batch_size, datum.channels(), crop_size,
				crop_size);
	} else {
		(*top)[0]->Reshape(batch_size, datum.channels(), datum.height(),
				datum.width());
		this->prefetch_data_.Reshape(batch_size, datum.channels(),
				datum.height(), datum.width());
	}
	LOG(INFO) << "output data size: " << (*top)[0]->num() << ","
			<< (*top)[0]->channels() << "," << (*top)[0]->height() << ","
			<< (*top)[0]->width();
	// label
	const int label_size = 128;
	const int pad_size = 0;
	(*top)[1]->Reshape(batch_size, 2 * (label_size - pad_size) * (label_size - pad_size), 1, 1);
	this->prefetch_label_.Reshape(batch_size, 2 * (label_size - pad_size) * (label_size - pad_size), 1,
			1);
	// datum size
	this->datum_channels_ = datum.channels();
	this->datum_height_ = datum.height();
	this->datum_width_ = datum.width();
	this->datum_size_ = datum.channels() * datum.height() * datum.width();
}

template<typename Dtype>
void ImageDataSGLayer<Dtype>::ShuffleImages() {
	caffe::rng_t* prefetch_rng =
			static_cast<caffe::rng_t*>(prefetch_rng_->generator());
	shuffle(lines_.begin(), lines_.end(), prefetch_rng);
}

// This function is used to create a thread that prefetches the data.
template<typename Dtype>
void ImageDataSGLayer<Dtype>::InternalThreadEntry() {
	Datum datum;
	CHECK(this->prefetch_data_.count());
	Dtype* top_data = this->prefetch_data_.mutable_cpu_data();
	Dtype* top_label = this->prefetch_label_.mutable_cpu_data();
	ImageDataParameter image_data_param = this->layer_param_.image_data_param();
	const int batch_size = image_data_param.batch_size();
	const int new_height = image_data_param.new_height();
	const int new_width = image_data_param.new_width();

	// datum scales
	const int lines_size = lines_.size();
	for (int item_id = 0; item_id < batch_size; ++item_id) {
		// get a blob
		CHECK_GT(lines_size, lines_id_);
		if (!ReadImageToDatumSG(lines_[lines_id_].first,
				lines_[lines_id_].second, new_height, new_width, &datum)) {
			continue;
		}

		// Apply transformations (mirror, crop...) to the data
		this->data_transformer_.Transform(item_id, datum, this->mean_,
				top_data);
		const int label_size = 128;
		const int pad_size = 0;
		/*for (int i = 0; i < label_size * label_size; i++) {
		 //LOG(INFO) << datum.label(i);
		 top_label[item_id * label_size * label_size + i] = datum.label(i);
		 }*/

		int i = 0;
		for (int h = pad_size; h < label_size - pad_size; h++) {
			for (int w = pad_size; w < label_size - pad_size; w++) {
				top_label[i] = datum.label(h * label_size + w);
				i++;
			}
		}
		LOG(INFO) << "item_id" << item_id;
		i = 0;
		for (int h = pad_size; h < label_size - pad_size; h++) {
			for (int w = pad_size; w < label_size - pad_size; w++) {
				top_label[(batch_size + item_id) * (label_size -2 * pad_size )* (label_size -2 * pad_size ) + i] = datum.label(label_size * label_size + h * label_size + w);
				i++;
			}
		}

		//LOG(INFO) << "label_size " << i;

		/*for (int i = 0; i < label_size * label_size; i++) {
			//LOG(INFO) << datum.label(i);
			top_label[(batch_size + item_id) * label_size * label_size + i] =
					datum.label(label_size * label_size + i);
		}*/

		//for (int i = 128*128*4; i < 128 * 128 * 5; i++) {
		//	LOG(INFO) << i << " " << top_data[i];
		//}
#if 0
		if (count_3 % 100 == 0) {
			const int img_size = 64;
			cv::Mat img__(img_size, img_size, CV_8UC1);
			//cv::resize(cv_img_2, img__, cv::Size(64, 64));
			int i__ = 0;
			for (int h = 0; h < img_size; h++) {
				for (int w = 0; w < img_size; w++) {

					img__.at<unsigned char>(h, w) = 255 * top_label[i__];
					i__++;
				}
			}
			cv::imshow("seg_l", img__);
			cv::waitKey(0);
		}

		count_3 = (count_3 + 1) % 10000;
#endif
		// go to the next iter
		lines_id_++;
		if (lines_id_ >= lines_size) {
			// We have reached the end. Restart from the first.
			DLOG(INFO) << "Restarting data prefetching from start.";
			lines_id_ = 0;
			if (this->layer_param_.image_data_param().shuffle()) {
				ShuffleImages();
			}
		}
	}
}

INSTANTIATE_CLASS(ImageDataSGLayer);

}  // namespace caffe

