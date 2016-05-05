#include <fcntl.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>
#include <leveldb/db.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdint.h>

#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"

namespace caffe {

using google::protobuf::io::FileInputStream;
using google::protobuf::io::FileOutputStream;
using google::protobuf::io::ZeroCopyInputStream;
using google::protobuf::io::CodedInputStream;
using google::protobuf::io::ZeroCopyOutputStream;
using google::protobuf::io::CodedOutputStream;
using google::protobuf::Message;

bool ReadProtoFromTextFile(const char* filename, Message* proto) {
	int fd = open(filename, O_RDONLY);
	CHECK_NE(fd, -1) << "File not found: " << filename;
	FileInputStream* input = new FileInputStream(fd);
	bool success = google::protobuf::TextFormat::Parse(input, proto);
	delete input;
	close(fd);
	return success;
}

void WriteProtoToTextFile(const Message& proto, const char* filename) {
	int fd = open(filename, O_WRONLY | O_CREAT | O_TRUNC, 0644);
	FileOutputStream* output = new FileOutputStream(fd);
	CHECK(google::protobuf::TextFormat::Print(proto, output));
	delete output;
	close(fd);
}

bool ReadProtoFromBinaryFile(const char* filename, Message* proto) {
	int fd = open(filename, O_RDONLY);
	CHECK_NE(fd, -1) << "File not found: " << filename;
	ZeroCopyInputStream* raw_input = new FileInputStream(fd);
	CodedInputStream* coded_input = new CodedInputStream(raw_input);
	coded_input->SetTotalBytesLimit(1073741824, 536870912);

	bool success = proto->ParseFromCodedStream(coded_input);

	delete coded_input;
	delete raw_input;
	close(fd);
	return success;
}

void WriteProtoToBinaryFile(const Message& proto, const char* filename) {
	fstream output(filename, ios::out | ios::trunc | ios::binary);
	CHECK(proto.SerializeToOstream(&output));
}

bool ReadImageToDatum(const string& filename, const int label, const int height,
		const int width, const bool is_color, Datum* datum) {
	cv::Mat cv_img;
	int cv_read_flag =
			(is_color ? CV_LOAD_IMAGE_COLOR : CV_LOAD_IMAGE_GRAYSCALE);

	cv::Mat cv_img_origin = cv::imread(filename, cv_read_flag);
	if (!cv_img_origin.data) {
		LOG(ERROR) << "Could not open or find file " << filename;
		return false;
	}
	if (height > 0 && width > 0) {
		cv::resize(cv_img_origin, cv_img, cv::Size(width, height));
	} else {
		cv_img = cv_img_origin;
	}

	int num_channels = (is_color ? 3 : 1);
	datum->set_channels(num_channels);
	datum->set_height(cv_img.rows);
	datum->set_width(cv_img.cols);
	datum->set_label(label, 0);
	datum->clear_data();
	datum->clear_float_data();
	string* datum_string = datum->mutable_data();
	if (is_color) {
		for (int c = 0; c < num_channels; ++c) {
			for (int h = 0; h < cv_img.rows; ++h) {
				for (int w = 0; w < cv_img.cols; ++w) {
					datum_string->push_back(
							static_cast<char>(cv_img.at < cv::Vec3b > (h, w)[c]));
				}
			}
		}
	} else {  // Faster than repeatedly testing is_color for each pixel w/i loop
		for (int h = 0; h < cv_img.rows; ++h) {
			for (int w = 0; w < cv_img.cols; ++w) {
				datum_string->push_back(
						static_cast<char>(cv_img.at < uchar > (h, w)));
			}
		}
	}
	return true;
}

bool ReadLrAndGtToDatum(const string& filename, const int height,
		const int width, const bool is_color, Datum* datum) {
	cv::Mat cv_img;
	int cv_read_flag =
			(is_color ? CV_LOAD_IMAGE_COLOR : CV_LOAD_IMAGE_GRAYSCALE);

	cv::Mat cv_img_origin = cv::imread(filename + ".bmp", cv_read_flag);
	if (!cv_img_origin.data) {
		LOG(ERROR) << "Could not open or find file " << filename + ".bmp";
		return false;
	}
	if (height > 0 && width > 0) {
		cv::resize(cv_img_origin, cv_img, cv::Size(width, height));
	} else {
		cv_img = cv_img_origin;
	}

	int num_channels = (is_color ? 3 : 1);
	datum->set_channels(num_channels);
	datum->set_height(cv_img.rows);
	datum->set_width(cv_img.cols);
	datum->clear_data();
	datum->clear_float_data();
	string* datum_string = datum->mutable_data();
	if (is_color) {
		for (int c = 0; c < 3; ++c) {
			for (int h = 0; h < cv_img.rows; ++h) {
				for (int w = 0; w < cv_img.cols; ++w) {
					datum_string->push_back(
							static_cast<char>(cv_img.at < cv::Vec3b > (h, w)[c]));
				}
			}
		}
	} else {  // Faster than repeatedly testing is_color for each pixel w/i loop
		for (int h = 0; h < cv_img.rows; ++h) {
			for (int w = 0; w < cv_img.cols; ++w) {
				datum_string->push_back(
						static_cast<char>(cv_img.at < uchar > (h, w)));
			}
		}
	}
	// set as labels
	cv::Mat label_img = cv::imread(filename + ".png", CV_LOAD_IMAGE_COLOR);
	if (!label_img.data) {
		LOG(ERROR) << "Could not open or find file " << filename + ".png";
		return false;
	}

	datum->clear_label();
	for (int c = 0; c < 3; ++c) {
		for (int h = 0; h < label_img.rows; ++h) {
			for (int w = 0; w < label_img.cols; ++w) {
				int value = static_cast<int>(label_img.at < cv::Vec3b
						> (h, w)[c]);
				datum->add_label(value);
			}
		}
	}
	return true;
}

bool ReadLrAndGtToDatum_Y_Only(const string& filename, const int height,
		const int width, const bool is_color, Datum* datum) {
	//CHECK_EQ(is_color, false)<< "Both the data and the label should convert to grayscale";

	cv::Mat cv_img;
	//int cv_read_flag =
	//(is_color ? CV_LOAD_IMAGE_COLOR : CV_LOAD_IMAGE_GRAYSCALE);

	cv::Mat cv_img_origin = cv::imread(filename + ".bmp",
			CV_LOAD_IMAGE_UNCHANGED);
	if (!cv_img_origin.data) {
		LOG(ERROR) << "Could not open or find file " << filename + ".bmp";
		return false;
	}
	if (height > 0 && width > 0) {
		cv::resize(cv_img_origin, cv_img, cv::Size(width, height));
	} else {
		cv_img = cv_img_origin;
	}

	int num_channels = 1;
	datum->set_channels(num_channels);
	datum->set_height(cv_img.rows);
	datum->set_width(cv_img.cols);
	datum->clear_data();
	datum->clear_float_data();
	string* datum_string = datum->mutable_data();
	// Faster than repeatedly testing is_color for each pixel w/i loop
	for (int h = 0; h < cv_img.rows; ++h) {
		for (int w = 0; w < cv_img.cols; ++w) {
			datum_string->push_back(
					static_cast<char>(cv_img.at < uchar > (h, w)));
		}
	}

	// set as labels
	cv::Mat label_img = cv::imread(filename + ".png", CV_LOAD_IMAGE_UNCHANGED);
	if (!label_img.data) {
		LOG(ERROR) << "Could not open or find file " << filename + ".png";
		return false;
	}

	datum->clear_label();
	for (int h = 0; h < label_img.rows; ++h) {
		for (int w = 0; w < label_img.cols; ++w) {
			int value = static_cast<int>(label_img.at < uchar > (h, w));
			datum->add_label(value);
		}
	}
	return true;
}

bool ReadImageToDatumSG(const string& filename, const string label,
		const int height, const int width, const bool is_color, Datum* datum) {
	cv::Mat cv_img;
	std::ifstream file_seg;
	int cv_read_flag =
			(is_color ? CV_LOAD_IMAGE_COLOR : CV_LOAD_IMAGE_GRAYSCALE);

	cv::Mat cv_img_origin = cv::imread(filename + ".jpg", cv_read_flag);
	if (!cv_img_origin.data) {
		LOG(ERROR) << "Could not open or find file " << filename + ".jpg";
		return false;
	}
	if (height > 0 && width > 0) {
		cv::resize(cv_img_origin, cv_img, cv::Size(width, height));
	} else {
		cv_img = cv_img_origin;
	}

	int num_channels = ((is_color ? 3 : 1) + 1);
	datum->set_channels(num_channels);
	datum->set_height(cv_img.rows);
	datum->set_width(cv_img.cols);
	datum->clear_label();

	const int label_height = 128;
	const int label_width = 128;
	cv::Mat label_img = cv::imread(label, CV_LOAD_IMAGE_GRAYSCALE);
	cv::resize(label_img, label_img, cv::Size(label_width, label_height));

	for (int h = 0; h < label_height; ++h) {
		for (int w = 0; w < label_width; ++w) {
			int tmp = int(label_img.at < uchar > (h, w));
			if (tmp >= 128) {
				datum->add_label(1);
			} else
				datum->add_label(0);
		}
	}

	std::string filename_seg = filename + "_seg.list";
	file_seg.open(filename_seg.c_str());
	if (!file_seg.is_open()) {
		LOG(INFO) << "Can not open file " << filename_seg;
	}

	int tmp;

	while (file_seg >> tmp) {
		datum->add_label(tmp);
	}

	datum->clear_data();
	datum->clear_float_data();
	string* datum_string = datum->mutable_data();
	if (is_color) {
		for (int c = 0; c < 3; ++c) {
			for (int h = 0; h < cv_img.rows; ++h) {
				for (int w = 0; w < cv_img.cols; ++w) {
					datum_string->push_back(
							static_cast<char>(cv_img.at < cv::Vec3b > (h, w)[c]));
				}
			}
		}
	} else {  // Faster than repeatedly testing is_color for each pixel w/i loop
		for (int h = 0; h < cv_img.rows; ++h) {
			for (int w = 0; w < cv_img.cols; ++w) {
				datum_string->push_back(
						static_cast<char>(cv_img.at < uchar > (h, w)));
			}
		}
	}

	for (int h = 0; h < cv_img.rows; ++h) {
		for (int w = 0; w < cv_img.cols; ++w) {
			float center_bias = (h - 127.5) * (h - 127.5)
					+ (w - 127.5) * (w - 127.5);
			float tmp = 255 * exp(-center_bias / 10000);
			datum_string->push_back(static_cast<char>(tmp));
		}
	}

	file_seg.close();

	return true;
}

bool ReadImageToDatumSEG(const string& filename, const string label,
		const int height, const int width, const bool is_color, Datum* datum) {
	cv::Mat cv_img;
	std::ifstream file_seg, file_label;
	int cv_read_flag =
			(is_color ? CV_LOAD_IMAGE_COLOR : CV_LOAD_IMAGE_GRAYSCALE);

	cv::Mat cv_img_origin = cv::imread(filename + ".jpg", cv_read_flag);
	if (!cv_img_origin.data) {
		LOG(ERROR) << "Could not open or find file " << filename + ".jpg";
		return false;
	}
	if (height > 0 && width > 0) {
		cv::resize(cv_img_origin, cv_img, cv::Size(width, height));
	} else {
		cv_img = cv_img_origin;
	}

	int num_channels = ((is_color ? 3 : 1) + 1);
	datum->set_channels(num_channels);
	datum->set_height(cv_img.rows);
	datum->set_width(cv_img.cols);
	datum->clear_label();

	std::string filename_seg = filename + "_seg.list";
	std::string filename_label = filename + "_l.list";
	file_seg.open(filename_seg.c_str());
	file_label.open(filename_label.c_str());

	if (!file_label.is_open()) {
		LOG(INFO) << "Can not open file " << filename_label;
	}

	if (!file_seg.is_open()) {
		LOG(INFO) << "Can not open file " << filename_seg;
	}
	int tmp;
	int num1 = 0;
	int num2 = 0;
	while (file_label >> tmp) {
		if (tmp >= 128) {
			datum->add_label(1);
		} else
			datum->add_label(0);
		num1++;
	}

	while (file_seg >> tmp) {
		datum->add_label(tmp);
		num2++;
	}

	if (num1 != 200 || num2 != 64 * 64)
		LOG(INFO) << num1 << " " << num2;

	const int label_height = 128;
	const int label_width = 128;
	cv::Mat label_img = cv::imread(label, CV_LOAD_IMAGE_GRAYSCALE);
	cv::resize(label_img, label_img, cv::Size(label_width, label_height));

	for (int h = 0; h < label_height; ++h) {
		for (int w = 0; w < label_width; ++w) {
			int tmp = int(label_img.at < uchar > (h, w));
			//			/LOG(INFO) << tmp;
			if (tmp >= 128) {
				datum->add_label(1);
				//				/LOG(INFO) << "1";
			} else
				datum->add_label(0);
		}
	}

	datum->clear_data();
	datum->clear_float_data();
	string* datum_string = datum->mutable_data();
	if (is_color) {
		for (int c = 0; c < 3; ++c) {
			for (int h = 0; h < cv_img.rows; ++h) {
				for (int w = 0; w < cv_img.cols; ++w) {
					datum_string->push_back(
							static_cast<char>(cv_img.at < cv::Vec3b > (h, w)[c]));
				}
			}
		}
	} else {  // Faster than repeatedly testing is_color for each pixel w/i loop
		for (int h = 0; h < cv_img.rows; ++h) {
			for (int w = 0; w < cv_img.cols; ++w) {
				datum_string->push_back(
						static_cast<char>(cv_img.at < uchar > (h, w)));
			}
		}
	}

	for (int h = 0; h < cv_img.rows; ++h) {
		for (int w = 0; w < cv_img.cols; ++w) {
			float center_bias = (h - 127.5) * (h - 127.5)
					+ (w - 127.5) * (w - 127.5);
			float tmp = 255 * exp(-center_bias / 10000);
			datum_string->push_back(static_cast<char>(tmp));
		}
	}
	file_seg.close();
	file_label.close();

	return true;
}

bool ReadImageToDatumJ(const string& filename, const string label,
		const int height, const int width, const bool is_color, Datum* datum, vector<float>& nei_weights) {
	cv::Mat cv_img;
	std::ifstream file_seg, file_label, file_weight;
	int cv_read_flag =
			(is_color ? CV_LOAD_IMAGE_COLOR : CV_LOAD_IMAGE_GRAYSCALE);

	cv::Mat cv_img_origin = cv::imread(filename + ".jpg", cv_read_flag);
	if (!cv_img_origin.data) {
		LOG(ERROR) << "Could not open or find file " << filename + ".jpg";
		return false;
	}
	if (height > 0 && width > 0) {
		cv::resize(cv_img_origin, cv_img, cv::Size(width, height));
	} else {
		cv_img = cv_img_origin;
	}

	int num_channels = ((is_color ? 3 : 1) + 1);
	//int num_channels = ((is_color ? 3 : 1));
	datum->set_channels(num_channels);
	datum->set_height(cv_img.rows);
	datum->set_width(cv_img.cols);
	datum->clear_label();

	std::string filename_seg = filename + "_seg.list";
	file_seg.open(filename_seg.c_str());
	if (!file_seg.is_open()) {
		LOG(INFO) << "Can not open file " << filename_seg;
	}

	int tmp;
	int num1 = 0;
	int num2 = 0;
	int num3 = 0;

	while(num1 < 200) {
		datum->add_label(0);
		num1++;
	}


	while (file_seg >> tmp) {
		datum->add_label(tmp);
		num2++;
	}

	const int label_height = 64;
	const int label_width = 64;

	for (int h = 0; h < label_height; ++h) {
		for (int w = 0; w < label_width; ++w) {
			datum->add_label(0);
		}
	}


	const int new_size = 128;
	for (int h = 0; h < new_size; ++h) {
		for (int w = 0; w < new_size; ++w) {
			datum->add_label(0);
		}
	}

	const int max_nei = 25;
	int tmp_num;
	nei_weights.clear();
	
	while(num3 < max_nei * 200 * 2) {
		nei_weights.push_back(0);
		num3++;
	}

	if (num1 != 200 || num2 != 64 * 64 || num3 != max_nei * 200 * 2) {
		LOG(INFO) << num1 << " " << num2 << " " << num3;
		LOG(INFO) << filename;
	}

	datum->clear_data();
	datum->clear_float_data();
	string* datum_string = datum->mutable_data();
	if (is_color) {
		for (int c = 0; c < 3; ++c) {
			for (int h = 0; h < cv_img.rows; ++h) {
				for (int w = 0; w < cv_img.cols; ++w) {
					datum_string->push_back(
							static_cast<char>(cv_img.at < cv::Vec3b > (h, w)[c]));
				}
			}
		}
	} else {  // Faster than repeatedly testing is_color for each pixel w/i loop
		for (int h = 0; h < cv_img.rows; ++h) {
			for (int w = 0; w < cv_img.cols; ++w) {
				datum_string->push_back(
						static_cast<char>(cv_img.at < uchar > (h, w)));
			}
		}
	}

	
	for (int h = 0; h < cv_img.rows; ++h) {
		for (int w = 0; w < cv_img.cols; ++w) {
			float center_bias = (h - 127.5) * (h - 127.5)
					+ (w - 127.5) * (w - 127.5);
			float tmp = 255 * exp(-center_bias / 10000);
			datum_string->push_back(static_cast<char>(tmp));
		}
	}

	file_seg.close();
	file_label.close();

	return true;
}

bool ReadImageToDatumSM(const string& filename, const string label,
		const int height, const int width, const bool is_color, Datum* datum) {
	cv::Mat cv_img;
	int cv_read_flag =
			(is_color ? CV_LOAD_IMAGE_COLOR : CV_LOAD_IMAGE_GRAYSCALE);

	cv::Mat cv_img_origin = cv::imread(filename + ".jpg", cv_read_flag);
	if (!cv_img_origin.data) {
		LOG(ERROR) << "Could not open or find file " << filename + ".jpg";
		return false;
	}
	if (height > 0 && width > 0) {
		cv::resize(cv_img_origin, cv_img, cv::Size(width, height));
	} else {
		cv_img = cv_img_origin;
	}

	int num_channels = ((is_color ? 3 : 1) + 1);
	datum->set_channels(num_channels);
	datum->set_height(cv_img.rows);
	datum->set_width(cv_img.cols);
	datum->clear_label();

	const int label_height = 128;
	const int label_width = 128;
	cv::Mat label_img = cv::imread(label, CV_LOAD_IMAGE_GRAYSCALE);
	cv::resize(label_img, label_img, cv::Size(label_width, label_height));

	for (int h = 0; h < label_height; ++h) {
		for (int w = 0; w < label_width; ++w) {
			int tmp = int(label_img.at < uchar > (h, w));
			if (tmp >= 128) {
				datum->add_label(1);
			} else
				datum->add_label(0);
		}
	}

	datum->clear_data();
	datum->clear_float_data();
	string* datum_string = datum->mutable_data();
	if (is_color) {
		for (int c = 0; c < 3; ++c) {
			for (int h = 0; h < cv_img.rows; ++h) {
				for (int w = 0; w < cv_img.cols; ++w) {
					datum_string->push_back(
							static_cast<char>(cv_img.at < cv::Vec3b > (h, w)[c]));
				}
			}
		}
	} else {  // Faster than repeatedly testing is_color for each pixel w/i loop
		for (int h = 0; h < cv_img.rows; ++h) {
			for (int w = 0; w < cv_img.cols; ++w) {
				datum_string->push_back(
						static_cast<char>(cv_img.at < uchar > (h, w)));
			}
		}
	}

	for (int h = 0; h < cv_img.rows; ++h) {
		for (int w = 0; w < cv_img.cols; ++w) {
			float center_bias = (h - 127.5) * (h - 127.5)
					+ (w - 127.5) * (w - 127.5);
			float tmp = 255 * exp(-center_bias / 10000);
			datum_string->push_back(static_cast<char>(tmp));
		}
	}

	return true;
}

bool ReadImageToDatumSL(const string& filename, const string label,
		const int height, const int width, const bool is_color, Datum* datum) {
	cv::Mat cv_img;
	cv::Mat cmap_img;
	int cv_read_flag =
			(is_color ? CV_LOAD_IMAGE_COLOR : CV_LOAD_IMAGE_GRAYSCALE);

	cv::Mat cv_img_origin = cv::imread(filename, cv_read_flag);
	cv::Mat cmap_img_orgin = cv::imread(label + "_sg.png",
			CV_LOAD_IMAGE_GRAYSCALE);

	if (!cv_img_origin.data) {
		LOG(INFO) << LOG(ERROR) << "Could not open or find file " << filename;
		return false;
	}
	if (height > 0 && width > 0) {
		cv::resize(cv_img_origin, cv_img, cv::Size(width, height));
		cv::resize(cmap_img_orgin, cmap_img, cv::Size(width, height));
	} else {
		cv_img = cv_img_origin;
		cmap_img = cmap_img_orgin;
	}

	int num_channels = ((is_color ? 3 : 1) + 2);
	datum->set_channels(num_channels);
	datum->set_height(cv_img.rows);
	datum->set_width(cv_img.cols);
	datum->clear_label();

	const int label_height = 128;
	const int label_width = 128;
	cv::Mat label_img = cv::imread(label + ".png", CV_LOAD_IMAGE_GRAYSCALE);
	cv::resize(label_img, label_img, cv::Size(label_width, label_height));
	for (int h = 0; h < label_height; ++h) {
		for (int w = 0; w < label_width; ++w) {
			int tmp = int(label_img.at < uchar > (h, w));
//			/LOG(INFO) << tmp;
			if (tmp >= 128) {
				datum->add_label(1);
//				/LOG(INFO) << "1";
			} else
				datum->add_label(0);
		}
	}

	datum->clear_data();
	datum->clear_float_data();
	string* datum_string = datum->mutable_data();
	if (is_color) {
		for (int c = 0; c < 3; ++c) {
			for (int h = 0; h < cv_img.rows; ++h) {
				for (int w = 0; w < cv_img.cols; ++w) {
					datum_string->push_back(
							static_cast<char>(cv_img.at < cv::Vec3b > (h, w)[c]));
				}
			}
		}
	} else {  // Faster than repeatedly testing is_color for each pixel w/i loop
		for (int h = 0; h < cv_img.rows; ++h) {
			for (int w = 0; w < cv_img.cols; ++w) {
				datum_string->push_back(
						static_cast<char>(cv_img.at < uchar > (h, w)));
			}
		}
	}

	for (int h = 0; h < cv_img.rows; ++h) {
		for (int w = 0; w < cv_img.cols; ++w) {
			float center_bias = (h - 127.5) * (h - 127.5)
					+ (w - 127.5) * (w - 127.5);
			float tmp = 255 * exp(-center_bias / 10000);
			datum_string->push_back(static_cast<char>(tmp));
		}
	}


	for (int h = 0; h < cmap_img.rows; ++h) {
		for (int w = 0; w < cmap_img.cols; ++w) {
			datum_string->push_back(
					static_cast<char>(cmap_img.at<unsigned char>(h, w)));
		}
	}

	return true;
}

leveldb::Options GetLevelDBOptions() {
	// In default, we will return the leveldb option and set the max open files
	// in order to avoid using up the operating system's limit.
	leveldb::Options options;
	options.max_open_files = 100;
	return options;
}

// Verifies format of data stored in HDF5 file and reshapes blob accordingly.
template<typename Dtype>
void hdf5_load_nd_dataset_helper(hid_t file_id, const char* dataset_name_,
		int min_dim, int max_dim, Blob<Dtype>* blob) {
	// Verify that the number of dimensions is in the accepted range.
	herr_t status;
	int ndims;
	status = H5LTget_dataset_ndims(file_id, dataset_name_, &ndims);
	CHECK_GE(status, 0) << "Failed to get dataset ndims for " << dataset_name_;
	CHECK_GE(ndims, min_dim);
	CHECK_LE(ndims, max_dim);

	// Verify that the data format is what we expect: float or double.
	std::vector < hsize_t > dims(ndims);
	H5T_class_t class_;
	status = H5LTget_dataset_info(file_id, dataset_name_, dims.data(), &class_,
			NULL);
	CHECK_GE(status, 0) << "Failed to get dataset info for " << dataset_name_;
	CHECK_EQ(class_, H5T_FLOAT) << "Expected float or double data";

	blob->Reshape(dims[0], (dims.size() > 1) ? dims[1] : 1,
			(dims.size() > 2) ? dims[2] : 1, (dims.size() > 3) ? dims[3] : 1);
}

template<>
void hdf5_load_nd_dataset<float>(hid_t file_id, const char* dataset_name_,
		int min_dim, int max_dim, Blob<float>* blob) {
	hdf5_load_nd_dataset_helper(file_id, dataset_name_, min_dim, max_dim, blob);
	herr_t status = H5LTread_dataset_float(file_id, dataset_name_,
			blob->mutable_cpu_data());
	CHECK_GE(status, 0) << "Failed to read float dataset " << dataset_name_;
}

template<>
void hdf5_load_nd_dataset<double>(hid_t file_id, const char* dataset_name_,
		int min_dim, int max_dim, Blob<double>* blob) {
	hdf5_load_nd_dataset_helper(file_id, dataset_name_, min_dim, max_dim, blob);
	herr_t status = H5LTread_dataset_double(file_id, dataset_name_,
			blob->mutable_cpu_data());
	CHECK_GE(status, 0) << "Failed to read double dataset " << dataset_name_;
}

template<>
void hdf5_save_nd_dataset<float>(const hid_t file_id, const string dataset_name,
		const Blob<float>& blob) {
	hsize_t dims[HDF5_NUM_DIMS];
	dims[0] = blob.num();
	dims[1] = blob.channels();
	dims[2] = blob.height();
	dims[3] = blob.width();
	herr_t status = H5LTmake_dataset_float(file_id, dataset_name.c_str(),
	HDF5_NUM_DIMS, dims, blob.cpu_data());
	CHECK_GE(status, 0) << "Failed to make float dataset " << dataset_name;
}

template<>
void hdf5_save_nd_dataset<double>(const hid_t file_id,
		const string dataset_name, const Blob<double>& blob) {
	hsize_t dims[HDF5_NUM_DIMS];
	dims[0] = blob.num();
	dims[1] = blob.channels();
	dims[2] = blob.height();
	dims[3] = blob.width();
	herr_t status = H5LTmake_dataset_double(file_id, dataset_name.c_str(),
	HDF5_NUM_DIMS, dims, blob.cpu_data());
	CHECK_GE(status, 0) << "Failed to make double dataset " << dataset_name;
}

}
// namespace caffe
