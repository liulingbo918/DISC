#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {
template <typename Dtype>
void IntraSmoothingLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
	Forward_cpu(bottom, top);
}


template <typename Dtype>
void IntraSmoothingLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    vector<Blob<Dtype>*>* bottom) {
    Backward_cpu(top, propagate_down, bottom);
}


INSTANTIATE_CLASS(IntraSmoothingLayer);


}  // namespace caffe
