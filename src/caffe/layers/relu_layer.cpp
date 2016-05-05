#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>

namespace caffe {
int image_num = 0;

template<typename Dtype>
void ReLULayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		vector<Blob<Dtype>*>* top) {
	const Dtype* bottom_data = bottom[0]->cpu_data();
	Dtype* top_data = (*top)[0]->mutable_cpu_data();
	const int count = bottom[0]->count();
	Dtype negative_slope = this->layer_param_.relu_param().negative_slope();
	for (int i = 0; i < count; ++i) {
		top_data[i] = std::max(bottom_data[i], Dtype(0))
				+ negative_slope * std::min(bottom_data[i], Dtype(0));
	}

/*#if 0
	if (bottom[0]->width() == 62 && bottom[0]->channels() == 96) {
		const int num = bottom[0]->num();
		const int channel = bottom[0]->channels();
		const int width = bottom[0]->width();
		const int height = bottom[0]->height();
		const int dim = width * height * channel;
		cv::Mat cv_img(width, height, CV_8U), cv_img_tmp;
		Dtype min, max;
		Dtype *temp_data;
		temp_data = new Dtype[width * height];

		for (int i = 0; i < num; i++) {

			for (int k = 0; k < width * height; k++)
				temp_data[k] = 0;
			max = 0;
			min = 9999999999.9;
			for (int j = 0; j < channel; j++) {
				for (int r = 0; r < height; r++) {
					for (int c = 0; c < width; c++) {
						temp_data[r * width + c] += bottom_data[i * dim
								+ j * width * height + r * width + c];
					}
				}
			}
			for (int r = 0; r < height; r++) {
				for (int c = 0; c < width; c++) {
					if (temp_data[r * width + c] > max)
						max = temp_data[r * width + c];
					if (temp_data[r * width + c] < min)
						min = temp_data[r * width + c];
				}
			}
			for (int r = 0; r < height; r++) {
				for (int c = 0; c < width; c++) {
					cv_img.at<unsigned char>(r, c) = static_cast<char>(256
							* (temp_data[r * width + c] - min) / (max - min));
//					/LOG(INFO) << temp_data[r * width + c] << " " << max << " " << min;
				}
			}

			cv::resize(cv_img, cv_img_tmp, cv::Size(227, 227));
			vector < std::string > filenames;
			std::string filename;
			std::ifstream file(
					"/home/d302/tianshui/DISC/datasets/MSRA/test_1k.list");
			while (file >> filename) {
				filenames.push_back(filename + "_sr");
			}
			//char buffer[100];
			//sprintf(buffer,"%d",image_num);
			//string str = buffer;
			cv::imwrite("/home/d302/tianshui/DISC/datasets/MSRA/conv1_map/" + filenames[image_num] + ".png", cv_img);
			cv::imwrite("/home/d302/tianshui/DISC/datasets/MSRA/conv1_map/" + filenames[image_num] + "_resize.png", cv_img_tmp);
			//LOG(INFO) << image_filename_const_[image_num];
			//cv::imshow("Feature_org", cv_img);
			//cv::imshow("Feature_res", cv_img_tmp);
			//cv::waitKey(0);
			image_num++;
			LOG(INFO) << "/home/d302/tianshui/DISC/datasets/MSRA/conv1_map/" + filenames[image_num] + ".png";

		}
	}
#endif*/


}

template<typename Dtype>
void ReLULayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {
	if (propagate_down[0]) {
		const Dtype* bottom_data = (*bottom)[0]->cpu_data();
		const Dtype* top_diff = top[0]->cpu_diff();
		Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();
		const int count = (*bottom)[0]->count();
		Dtype negative_slope = this->layer_param_.relu_param().negative_slope();
		for (int i = 0; i < count; ++i) {
			bottom_diff[i] = top_diff[i]
					* ((bottom_data[i] > 0)
							+ negative_slope * (bottom_data[i] <= 0));
		}
	}
}

#ifdef CPU_ONLY
STUB_GPU(ReLULayer);
#endif

INSTANTIATE_CLASS(ReLULayer);

}  // namespace caffe
