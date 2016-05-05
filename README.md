# [<i>DISC: Deep Image Saliency Computing via Progressive Representation Learning][1]
===================

### Introduction
-----------
Salient object detection increasingly receives attention as an important component or step in several pattern recognition and image processing tasks. Although a variety of powerful saliency models have been intensively proposed, they usually involve heavy feature (or model) engineering based on priors (or assumptions) about the properties of objects and backgrounds. Inspired by the effectiveness of recently developed feature learning, we provide a novel Deep Image Saliency Computing (DISC) framework for fine-grained image saliency computing. In particular, we model the image saliency from both the coarse- and fine-level observations, and utilize the deep convolutional neural network (CNN) to learn the saliency representation in a progressive manner. Specifically, our saliencymodel is built upon two stacked CNNs. The first CNN generates a coarse-level saliency map by taking the overall image asthe input, roughly identifying saliency regions in the global context. Furthermore, we integrate superpixel-based local context information in the first CNN to refine the coarse-level saliency map. Guided by the coarse saliency map, the second CNN focuses on the local context to produce fine-grained and accurate saliency map while preserving object details. For a testing image, the two CN s collaboratively conduct the saliency computing in one shot. Our DISC framework is capable of uniformly highlighting the objects-of-interest from complex background while preserving well object details. Extensive experiments on several standard benchmarks suggest that DISC outperforms other state-of-the- art methods and it also generalizes well across datasets without additional training. 


### Usage
This experiment is based on Convolution Neural Network. To run our code, you need to install the `Caffe` in your computer  (see: [<i>Caffe installation instructions][2].
We'll call the directory that you cloned our code into `DISC_ROOT`.

Download DISC Model
>1). Download our model from [<i>here][3] and put it into `$DISC_ROOT/DISC_Net/DISC_Model/`.

Compiling
  ```Shell
  cd $DISC_ROOT
  make
  ```

Testing
>1). Put the images(jpg) into `$DISC_ROOT/DISC_Data/DISC_Input/` and the result will be sorted in `$DISC_ROOT/DISC_Data/DISC_Output/`.
>2). To generate DISC saliency map, run:
  ```Shell
  cd $DISC_ROOT
  sh DISC/DISC_Script/DISC_test.sh
  ```

### Citing DISC
If you find DISC useful in your research, please consider citing:

	@article{chen2015disc,
	  title={DISC: Deep Image Saliency Computing via Progressive Representation Learning},
	  author={Chen, Tianshui and Lin, Liang and Liu, Lingbo and Luo, Xiaonan and Li, Xuelong},
	  year={2015},
	  publisher={IEEE}
	}


If you have any question for this code，contact us with tianshuichen@gmail.com or liulingbo918@gmail.com .



  [1]: http://vision.sysu.edu.cn/vision_sysu/wp-content/uploads/2015/12/TNNLS_DeepImSaliency.pdf
  [2]: http://caffe.berkeleyvision.org/installation.html
  [3]: http://pan.baidu.com/s/1mhUPf5Y

