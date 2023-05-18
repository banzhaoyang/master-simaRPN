#ifndef SIMARPN_COMMON_H_
#define SIMARPN_COMMON_H_

#include <fstream>
#include <map>
#include <sstream>
#include <vector>
#include <opencv2/opencv.hpp>
#include "NvInfer.h"
#include <Eigen/Dense>


using namespace nvinfer1;

std::map<std::string, nvinfer1::Weights> loadWeights(const std::string file);

ILayer* convBlock(INetworkDefinition *network, std::map<std::string, nvinfer1::Weights>& weightMap, ITensor& input, int outch, int ksize, int s, int g, std::string lname, int num, bool relu_n);

ILayer* convBlock_nonpadding(INetworkDefinition *network, std::map<std::string, nvinfer1::Weights>& weightMap, ITensor& input, int outch, int ksize, int s, int g, std::string lname, int num, bool relu_n);

ILayer* convBlock_dilation(INetworkDefinition *network, std::map<std::string, nvinfer1::Weights>& weightMap, ITensor& input, int outch, int ksize, int s, int g, std::string lname, int num, int pad_dila);

ILayer* neck_convBlock(INetworkDefinition *network, std::map<std::string, nvinfer1::Weights>& weightMap, ITensor& input, int outch, int ksize, int s, int g, std::string lname, int num);

ILayer* head_convBlock(INetworkDefinition *network, std::map<std::string, nvinfer1::Weights>& weightMap, ITensor& input, int outch, int ksize, int s, int g, std::string lname);

ILayer* head_head(INetworkDefinition *network, std::map<std::string, nvinfer1::Weights>& weightMap, ITensor& input, int outch, int ksize, int s, int g, std::string lname);

ILayer* downsample(INetworkDefinition *network, std::map<std::string, nvinfer1::Weights>& weightMap, ITensor& input, int outch, int ksize, int s, int g, std::string lname);

ILayer* downsample_nonpadding(INetworkDefinition *network, std::map<std::string, nvinfer1::Weights>& weightMap, ITensor& input, int outch, int ksize, int s, int g, std::string lname);

ILayer* downsample_dilation(INetworkDefinition *network, std::map<std::string, nvinfer1::Weights>& weightMap, ITensor& input, int outch, int ksize, int s, int g, std::string lname, int pad_dila);

IResizeLayer* upsample_i(INetworkDefinition *network, ITensor& input, bool mode);

IResizeLayer* downsample_i(INetworkDefinition *network, ITensor& input);

IResizeLayer* downsample_dims(INetworkDefinition *network, ITensor& input, int dims_size);

IScaleLayer* addBatchNorm2d(INetworkDefinition *network, std::map<std::string, nvinfer1::Weights>& weightMap, ITensor& input, std::string lname, float eps);

IScaleLayer* addGroupNorm2d(INetworkDefinition *network, std::map<std::string, nvinfer1::Weights>& weightMap, ITensor& input, std::string lname, float eps, int G);

IPoolingLayer* addMaxpool(INetworkDefinition *network, ITensor& input, int ksize, int s);

ILayer* depthwise_correlation(INetworkDefinition *network ,ITensor& x, ITensor& kernel_weight);

ILayer* depthwise_correlation_group256_conv(INetworkDefinition *network, ITensor& x, ITensor& kernel);

IReduceLayer* addVarience(INetworkDefinition *network, ITensor& input);

IConcatenationLayer* gen_x_linspace(INetworkDefinition *network, int feat_size);

IConcatenationLayer* gen_y_linspace(INetworkDefinition *network, int feat_size);

ILayer* CGR_kernel(INetworkDefinition *network, std::map<std::string, nvinfer1::Weights>& weightMap, ITensor& input, int outch, int ksize, int s, int g, std::string lname, int num, int pad);

IElementWiseLayer* kEQUAL(INetworkDefinition* network, IConvolutionLayer *cate_kernel, IPoolingLayer* point_nms);

#endif
