#include <iostream>
#include <opencv2/opencv.hpp>
#include <string.h>
#include "NvOnnxParser.h"
#include <cuda_runtime_api.h>
#include <algorithm>
#include <cassert>
#include "common.h"
#include "cuda_utils.h"
#include <torch/torch.h>
#include <torch/script.h>
#include <Eigen/Dense>


//#include "c10/core/Device.h"

#pragma once


class simaRPN
{

public:

	bool gen_template_neck_engine();
	bool gen_search_neck_engine();
	bool gen_head_engine();

	bool load_template_neck_engine();
	bool load_search_neck_engine();
	bool load_head_engine();


	void inference();


	std::vector<ILayer*> backbone(nvinfer1::INetworkDefinition* network, std::map<std::string, nvinfer1::Weights>& weightMap, IPoolingLayer* pool);


public:

	std::string video_path; //non_logo_1920-1080.avi
	std::string wts_name;


private:
	typedef cv::Rect_<double> Rect2d;
	void _image_inference(int frame_id);
	void _MatToTensor_template_neck(cv::Mat img);
	void _MatToTensor_search_neck(cv::Mat img);

	void _TensorToTorch_tensor_cls();
	void _TensorToTorch_tensor_reg();
	void _TensorToTorch_mat_tensor_cls();
	void _TensorToTorch_mat_tensor_reg();





	void _get_box(int frame, double scale_factor);
	void _anchor_generator();
	void _windows_generator();
	torch::Tensor _decode_bbox(double scale_factor);
	void _bbox_clip(int img_h, int img_w);
	auto _bbox_xyxy_to_xywh(torch::Tensor bbox);
	void _cout_output_Tensor(int frame_id);
	cv::Mat _get_crop_frame(cv::Rect2d init_bbox, cv::Mat img, int frame_id);




private:
	cv::Mat _output_img;

	int _input_h = 255;
	int _input_w = 255;
	int _template_input_h = 127;
	int _template_input_w = 127;

	int _engine_mode = 32;
	int _neckOutput_channel = 256;
	int _neckOutput_h = 31;
	int _neckOutput_w = 31;
	int _batch_size = 1;
	float _kalman_score = 0.80;

	int _crop_x_size;
	float _z_size;
	int _crop_z_size;
	std::chrono::system_clock::time_point _inf_start;
	std::chrono::system_clock::time_point _inf_end;

	bool _save_video = false;
	bool _kalman_open = false;

	void *_buffers_template_neck[2];
	void *_buffers_search_neck[2];
	void *_buffers_head[4];

	float *_neck_template_input;
	float *_neck_search_input;
	float *_head_input_srh1;
	float *_head_input_srh2;
	float *_head_input_srh3;

	float *_head_input_tep1;
	float *_head_input_tep2;
	float *_head_input_tep3;


	float *_neck_template_output1;
	float *_neck_template_output2;
	float *_neck_template_output3;

	float *_neck_search_output1;
	float *_neck_search_output2;
	float *_neck_search_output3;


	float *_head_output_cls;
	float *_head_output_reg;

	torch::Tensor _class_score;
	torch::Tensor _reg_score;
	torch::Tensor _all_anchors;
	torch::Tensor _windows;
	typedef cv::Rect_<double> Rect;
	cv::Rect _result_bbox;
	torch::Tensor _final_bbox_matric[10000];
	torch::Tensor _final_bbox;
	torch::Tensor _best_score;


	IRuntime* _runtime_head;
	IRuntime* _runtime_template_neck;
	IRuntime* _runtime_search_neck;

	cudaStream_t _stream_head;
	cudaStream_t _stream_template_neck;
	cudaStream_t _stream_search_neck;

	IExecutionContext* _context_head;
	IExecutionContext* _context_template_neck;
	IExecutionContext* _context_search_neck;

	ICudaEngine* _engine_head;
	ICudaEngine* _engine_template_neck;
	ICudaEngine* _engine_search_neck;


	std::string _head_engine_name = "siamese_head.engine";
	std::string _template_neck_engine_name = "siamese_template_neck.engine";
	std::string _search_neck_engine_name = "siamese_search_neck.engine";

	std::vector<const char*> _input_name = { "input1", "input2", "input3", "input4", "input5", "input6" };
	std::vector<const char*> _output_name = { "output1", "output2", "output3" };

	std::map<std::string, nvinfer1::Weights> _weightMap;
	cv::VideoCapture _video;

};


