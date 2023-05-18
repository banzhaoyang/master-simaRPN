
#include "common.h" 

std::map<std::string, nvinfer1::Weights> loadWeights(const std::string file) {
	std::cout << "Loading weights: " << file << std::endl;
	std::map<std::string, nvinfer1::Weights> weightMap;

	// Open weights file
	std::ifstream input(file);
	assert(input.is_open() && "Unable to load weight file. please check if the .wts file path is right!!!!!!");

	// Read number of weight blobs
	int32_t count;
	input >> count;
	assert(count > 0 && "Invalid weight map file.");

	while (count--)
	{
		nvinfer1::Weights wt{ nvinfer1::DataType::kFLOAT, nullptr, 0 };
		wt.type = nvinfer1::DataType::kFLOAT;
		uint32_t size;

		// Read name and type of blob
		//std::string name1 = { "head.cls_weight" };
		//std::string name2 = { "head.reg_weight" };
		std::string name;
		input >> name >> std::dec >> size;
		//if (name == name1 || name == name2)
		//{
		//	std::cout << " ";
		//}
		//{
			//uint32_t* val0 = reinterpret_cast<uint32_t*>(malloc(sizeof(val0) * 1));
			//uint32_t* val1 = reinterpret_cast<uint32_t*>(malloc(sizeof(val1) * 1));
			//uint32_t* val2 = reinterpret_cast<uint32_t*>(malloc(sizeof(val2) * 1));
			//std::vector<uint32_t*> val = { val0, val1, val2 };

			//input >> std::hex >> val0[0];
			//input >> std::hex >> val0[1];
			//input >> std::hex >> val0[2];
			//for (int i = 0; i < size; i++)
			//{
			//	wt.values = val[i];
			//	wt.count = 1;
			//	weightMap[name + std::to_string(i)] = wt;
			//}
		//}
		//else {
			// Load blob
		uint32_t* val = reinterpret_cast<uint32_t*>(malloc(sizeof(val) * size));
		for (uint32_t x = 0, y = size; x < y; ++x)
		{
			input >> std::hex >> val[x];
		}
		wt.values = val;

		wt.count = size;
		weightMap[name] = wt;
		//}
	}

	return weightMap;
}


/*
图像上一个5*5的卷积，假设stride也是5，那么可以用5*5=25个1*1的卷积做变换，
然后把图像也每隔stride的像素截取出来，这样图像也变成了25个，
最后在用25个1*1的卷积分别在25个小图上做eltwise(或者matmul），
最后按通道相加起来
*/

ILayer* depthwise_correlation(INetworkDefinition *network, ITensor& x, ITensor& kernel_weight)
{
	std::vector<ISliceLayer*> slice;
	std::vector<ISliceLayer*> kernel_5;

	for (int32_t i = 0; i < 5; i++)
	{
		for (int32_t j = 0; j < 5; j++)
		{
			slice.push_back(network->addSlice(x, Dims3{ 0, j, i }, Dims3{ 256,  25,  25 }, Dims3{ 1, 1, 1 }));
			kernel_5.push_back(network->addSlice(kernel_weight, Dims3{ 0, j, i }, Dims3{ 256,  1,  1 }, Dims3{ 1, 1, 1 }));
		}
	}
	std::vector<IElementWiseLayer*> element;
	for (int32_t i = 0; i < slice.size(); i++)
	{
		element.push_back(network->addElementWise(*slice[i]->getOutput(0), *kernel_5[i]->getOutput(0), ElementWiseOperation::kPROD));
	}

	auto elw = network->addElementWise(*element[0]->getOutput(0), *element[1]->getOutput(0), ElementWiseOperation::kSUM);
	for (int32_t i = 2; i < element.size(); i++)
	{
		elw = network->addElementWise(*elw->getOutput(0), *element[i]->getOutput(0), ElementWiseOperation::kSUM);
	}
	
	return elw;
}

ILayer* depthwise_correlation_group256_conv(INetworkDefinition *network, ITensor& x, ITensor& kernel)
{
	std::vector<ISliceLayer*> slice;
	std::vector<ISliceLayer*> kernel_5;
	ITensor* group_cat[256];

	for (int32_t k = 0; k < 256; k++)
	{
		for (int32_t i = 0; i < 5; i++)
		{
			for (int32_t j = 0; j < 5; j++)
			{
				slice.push_back(network->addSlice(x, Dims3{ k, j, i }, Dims3{ 1,  25,  25 }, Dims3{ 1, 1, 1 }));
				kernel_5.push_back(network->addSlice(kernel, Dims3{ k, j, i }, Dims3{ 1,  1,  1 }, Dims3{ 1, 1, 1 }));
			}
		}
		std::vector<IElementWiseLayer*> element;
		for (int32_t i = 0; i < slice.size(); i++)
		{
			element.push_back(network->addElementWise(*slice[i]->getOutput(0), *kernel_5[i]->getOutput(0), ElementWiseOperation::kPROD));
		}

		auto elw = network->addElementWise(*element[0]->getOutput(0), *element[1]->getOutput(0), ElementWiseOperation::kSUM);
		for (int32_t i = 2; i < element.size(); i++)
		{
			elw = network->addElementWise(*elw->getOutput(0), *element[i]->getOutput(0), ElementWiseOperation::kSUM);
		}
		group_cat[k] = elw->getOutput(0);
		element.clear();
		slice.clear();
		kernel_5.clear();
	}
	IConcatenationLayer* concat = network->addConcatenation(group_cat, 256);
	concat->setAxis(0);
	return concat;
}

ILayer* neck_convBlock(INetworkDefinition *network, std::map<std::string, nvinfer1::Weights>& weightMap, ITensor& input, int outch, int ksize, int s, int g, std::string lname, int num) {
	nvinfer1::Weights emptywts{ DataType::kFLOAT, nullptr, 0 };
	int p = ksize / 2;
	std::string conv[] = { ".convs.0.conv.weight" ,".convs.1.conv.weight" ,".convs.2.conv.weight" };
	std::string bn[] = { ".convs.0.bn" ,".convs.1.bn" ,".convs.2.bn" };
	IConvolutionLayer* conv1 = network->addConvolutionNd(input, outch, DimsHW{ ksize, ksize }, weightMap[lname + conv[num]], emptywts);
	assert(conv1);
	conv1->setStrideNd(DimsHW{ s, s });      // stride 设定卷积步长
	conv1->setPaddingNd(DimsHW{ p, p });     // padding 设定padding步长 加零
	conv1->setNbGroups(g);                   // 设置卷积的组数
	IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), lname + bn[num], 1e-5);

	return bn1;
}

ILayer* convBlock(INetworkDefinition *network, std::map<std::string, nvinfer1::Weights>& weightMap, ITensor& input, int outch, int ksize, int s, int g, std::string lname, int num, bool relu_n) {
	nvinfer1::Weights emptywts{ DataType::kFLOAT, nullptr, 0 };
	int p = ksize / 2;
	std::string conv[] = { ".conv1.weight" ,".conv2.weight" ,".conv3.weight"};
	std::string bn[] = { ".bn1" ,".bn2" ,".bn3"};
	IConvolutionLayer* conv1 = network->addConvolutionNd(input, outch, DimsHW{ ksize, ksize }, weightMap[lname + conv[num]], emptywts);
	assert(conv1);
	conv1->setStrideNd(DimsHW{ s, s });      // stride 设定卷积步长
	conv1->setPaddingNd(DimsHW{ p, p });     // padding 设定padding步长 加零
	conv1->setNbGroups(g);                   // 设置卷积的组数
	IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), lname + bn[num], 1e-5);
	if (relu_n)
	{
		auto relu = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);
		assert(relu);
		return relu;
	}
	return bn1;
}

ILayer* convBlock_nonpadding(INetworkDefinition *network, std::map<std::string, nvinfer1::Weights>& weightMap, ITensor& input, int outch, int ksize, int s, int g, std::string lname, int num, bool relu_n) {
	nvinfer1::Weights emptywts{ DataType::kFLOAT, nullptr, 0 };
	std::string conv[] = { ".conv1.weight" ,".conv2.weight" ,".conv3.weight" };
	std::string bn[] = { ".bn1" ,".bn2" ,".bn3" };
	IConvolutionLayer* conv1 = network->addConvolutionNd(input, outch, DimsHW{ ksize, ksize }, weightMap[lname + conv[num]], emptywts);
	assert(conv1);
	conv1->setStrideNd(DimsHW{ s, s });      // stride 设定卷积步长
	conv1->setNbGroups(g);                   // 设置卷积的组数
	IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), lname + bn[num], 1e-5);
	if (relu_n)
	{
		auto relu = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);
		assert(relu);
		return relu;
	}
	return bn1;
}

ILayer* head_convBlock(INetworkDefinition *network, std::map<std::string, nvinfer1::Weights>& weightMap, ITensor& input, int outch, int ksize, int s, int g, std::string lname) {
	nvinfer1::Weights emptywts{ DataType::kFLOAT, nullptr, 0 };

	IConvolutionLayer* conv1 = network->addConvolutionNd(input, outch, DimsHW{ ksize, ksize }, weightMap[lname + ".conv.weight"], emptywts);
	assert(conv1);
	conv1->setStrideNd(DimsHW{ s, s });      // stride 设定卷积步长
	conv1->setNbGroups(g);                   // 设置卷积的组数
	IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), lname + ".bn", 1e-5);

	auto relu = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);
	assert(relu);

	return relu;
}

ILayer* head_head(INetworkDefinition *network, std::map<std::string, nvinfer1::Weights>& weightMap, ITensor& input, int outch, int ksize, int s, int g, std::string lname) {
	
	IConvolutionLayer* conv = network->addConvolutionNd(input, outch, DimsHW{ ksize , ksize }, weightMap[lname + ".weight"], weightMap[lname + ".bias"]);
	return conv;
}

ILayer* convBlock_dilation(INetworkDefinition *network, std::map<std::string, nvinfer1::Weights>& weightMap, ITensor& input, int outch, int ksize, int s, int g, std::string lname, int num, int pad_dila) {
	nvinfer1::Weights emptywts{ DataType::kFLOAT, nullptr, 0 };
	std::string conv[] = { ".conv1.weight" ,".conv2.weight" ,".conv3.weight" };
	std::string bn[] = { ".bn1" ,".bn2" ,".bn3" };
	IConvolutionLayer* conv1 = network->addConvolutionNd(input, outch, DimsHW{ ksize, ksize }, weightMap[lname + conv[num]], emptywts);
	assert(conv1);
	conv1->setStrideNd(DimsHW{ s, s });                    // stride 设定卷积步长
	conv1->setPaddingNd(DimsHW{ pad_dila, pad_dila });     // padding 设定padding步长 加零
	conv1->setDilation(DimsHW{ pad_dila, pad_dila });      // 设置空洞卷积大小
	conv1->setNbGroups(g);                                 // 设置卷积的组数
	IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), lname + bn[num], 1e-5);

	auto relu = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);
	assert(relu);
	return relu;
}

ILayer* downsample(INetworkDefinition *network, std::map<std::string, nvinfer1::Weights>& weightMap, ITensor& input, int outch, int ksize, int s, int g, std::string lname) {
	nvinfer1::Weights emptywts{ DataType::kFLOAT, nullptr, 0 };
	int p = ksize / 2;

	IConvolutionLayer* conv1 = network->addConvolutionNd(input, outch, DimsHW{ ksize, ksize }, weightMap[lname + ".0.weight"], emptywts);
	assert(conv1);
	conv1->setStrideNd(DimsHW{ s, s });      // stride 设定卷积步长
	conv1->setPaddingNd(DimsHW{ p, p });     // padding 设定padding步长 加零
	conv1->setNbGroups(g);                   // 设置卷积的组数
	IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), lname + ".1", 1e-5);

	//auto relu = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);
	//assert(relu);
	return bn1;
}

ILayer* downsample_nonpadding(INetworkDefinition *network, std::map<std::string, nvinfer1::Weights>& weightMap, ITensor& input, int outch, int ksize, int s, int g, std::string lname) {
	nvinfer1::Weights emptywts{ DataType::kFLOAT, nullptr, 0 };

	IConvolutionLayer* conv1 = network->addConvolutionNd(input, outch, DimsHW{ ksize, ksize }, weightMap[lname + ".0.weight"], emptywts);
	assert(conv1);
	conv1->setStrideNd(DimsHW{ s, s });      // stride 设定卷积步长
	conv1->setNbGroups(g);                   // 设置卷积的组数
	IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), lname + ".1", 1e-5);

	//auto relu = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);
	//assert(relu);
	return bn1;
}

ILayer* downsample_dilation(INetworkDefinition *network, std::map<std::string, nvinfer1::Weights>& weightMap, ITensor& input, int outch, int ksize, int s, int g, std::string lname, int pad_dila) {
	nvinfer1::Weights emptywts{ DataType::kFLOAT, nullptr, 0 };

	IConvolutionLayer* conv1 = network->addConvolutionNd(input, outch, DimsHW{ ksize, ksize }, weightMap[lname + ".0.weight"], emptywts);
	assert(conv1);
	conv1->setStrideNd(DimsHW{ s, s });      // stride 设定卷积步长
	conv1->setPaddingNd(DimsHW{ pad_dila, pad_dila });     // padding 设定padding步长 加零
	conv1->setDilation(DimsHW{ pad_dila, pad_dila });
	conv1->setNbGroups(g);                   // 设置卷积的组数
	IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), lname + ".1", 1e-5);

	//auto relu = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);
	//assert(relu);
	return bn1;
}

IResizeLayer* upsample_i(INetworkDefinition *network, ITensor& input, bool kLINEAR)
{
	IResizeLayer *upSample = network->addResize(input);
	if (kLINEAR)
	{
		upSample->setResizeMode(ResizeMode::kLINEAR);
	}
	else
	{
		upSample->setResizeMode(ResizeMode::kNEAREST);
	}
	float *scales = new float[3];
	scales[0] = 1;     // C
	scales[1] = 2;     // H
	scales[2] = 2;     // W
	upSample->setScales(scales, 3);  // （1,x,x）  number of diminsions
	upSample->setAlignCorners(true); // tips!

	return upSample;
}

IResizeLayer* downsample_i(INetworkDefinition *network, ITensor& input)
{
	IResizeLayer *downSample = network->addResize(input);
	downSample->setResizeMode(ResizeMode::kLINEAR);
	float *scales = new float[3];
	scales[0] = 1;     // C
	scales[1] = 0.5;     // H
	scales[2] = 0.5;     // W
	downSample->setScales(scales, 3);  // （1,x,x）  number of diminsions
	downSample->setAlignCorners(true); // tips!

	return downSample;
}

IResizeLayer* downsample_dims(INetworkDefinition *network, ITensor& input, int dims_size)
{
	IResizeLayer *downSample = network->addResize(input);
	downSample->setResizeMode(ResizeMode::kLINEAR);
	float *scales = new float[3];

	std::vector<Dims3> dim = { Dims3{ 258, 40, 40 } ,Dims3{ 258, 36, 36 } ,Dims3{ 258, 24, 24 } ,Dims3{ 258, 16, 16 } ,Dims3{ 258, 12, 12 } };
	if (dims_size == 40)
	{
		downSample->setOutputDimensions(dim[0]);
	}
	else if(dims_size == 36)
	{
		downSample->setOutputDimensions(dim[1]);
	}
	else if (dims_size == 24)
	{
		downSample->setOutputDimensions(dim[2]);
	}
	else if (dims_size == 16)
	{
		downSample->setOutputDimensions(dim[3]);
	}
	else if (dims_size == 12)
	{
		downSample->setOutputDimensions(dim[4]);
	}
	downSample->setAlignCorners(true); // tips!

	return downSample;
}

IPoolingLayer* addMaxpool(INetworkDefinition *network, ITensor & input, int ksize, int s) 
{
	IPoolingLayer* pool = network->addPoolingNd(input, PoolingType::kMAX, DimsHW{ ksize, ksize });

	pool->setPaddingNd(DimsHW{ 1, 1 });
	if (s == 2)
	{
		pool->setStrideNd(DimsHW{ 2, 2 });
	}
	else if (s == 1)
	{
		pool->setStrideNd(DimsHW{ 1, 1 });
	}
	//pool->setPadding(DimsHW{ 1, 1 });

	return pool;
}

IScaleLayer* addBatchNorm2d(INetworkDefinition *network, std::map<std::string, nvinfer1::Weights>& weightMap, ITensor& input, std::string lname, float eps) {
	float *gamma = (float*)weightMap[lname + ".weight"].values;
	float *beta = (float*)weightMap[lname + ".bias"].values;
	float *mean = (float*)weightMap[lname + ".running_mean"].values;
	float *var = (float*)weightMap[lname + ".running_var"].values;
	int len = weightMap[lname + ".running_var"].count;

	float *scval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
	for (int i = 0; i < len; i++) {
		scval[i] = gamma[i] / sqrt(var[i] + eps);
	}
	nvinfer1::Weights scale{ DataType::kFLOAT, scval, len };

	float *shval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
	for (int i = 0; i < len; i++) {
		shval[i] = beta[i] - mean[i] * gamma[i] / sqrt(var[i] + eps);
	}
	nvinfer1::Weights shift{ DataType::kFLOAT, shval, len };

	float *pval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
	for (int i = 0; i < len; i++) {
		pval[i] = 1.0;
	}
	nvinfer1::Weights power{ DataType::kFLOAT, pval, len };

	weightMap[lname + ".scale"] = scale;
	weightMap[lname + ".shift"] = shift;
	weightMap[lname + ".power"] = power;
	IScaleLayer* scale_1 = network->addScale(input, ScaleMode::kCHANNEL, shift, scale, power);  //为网络添加缩放层
	assert(scale_1);
	return scale_1;
}

IReduceLayer* addVarience(INetworkDefinition *network, ITensor& input)
{
	IReduceLayer* reduce1 = network->addReduce(input, ReduceOperation::kAVG, 6, true);   //axs = 6 是对 w h 所有值做均值 保留通道数
	auto els1 = network->addElementWise(input, *reduce1->getOutput(0), ElementWiseOperation::kSUB);
	auto els2 = network->addElementWise(*els1->getOutput(0), *els1->getOutput(0), ElementWiseOperation::kPROD);
	IReduceLayer* var = network->addReduce(*els2->getOutput(0), ReduceOperation::kAVG, 6, true);
	return var;
}

IScaleLayer* addGroupNorm2d(INetworkDefinition *network, std::map<std::string, nvinfer1::Weights>& weightMap, ITensor& input, std::string lname, float eps, int G = 32) 
{
	float *gamma = (float*)weightMap[lname + ".weight"].values;
	float *beta = (float*)weightMap[lname + ".bias"].values;

	int len = weightMap[lname + ".weight"].count;
	int group_num = len / G;
	Dims input_dim = input.getDimensions();

	std::vector<ITensor *> norm_vec;
	for (int i = 0; i < group_num; i++)
	{
		ISliceLayer * slice_input = network->addSlice(input, Dims3(i * G, 0, 0), Dims3(G, input_dim.d[1], input_dim.d[2]), Dims3(1, 1, 1));
		IReduceLayer* var = addVarience(network, *slice_input->getOutput(0));
		IReduceLayer* mean = network->addReduce(*slice_input->getOutput(0), ReduceOperation::kAVG, 6, true);

		float *eps_val = reinterpret_cast<float*>(malloc(sizeof(float) * G));
		for (int i = 0; i < G; i++)
		{
			eps_val[i] = eps;
		}
		nvinfer1::Weights weps{ DataType::kFLOAT, eps_val, G };
		IConstantLayer* eps_tensor = network->addConstant(Dims3{ G, 1, 1 }, weps);

		//Dims dim = mean->getOutput(0)->getDimensions();
		auto inputSubmean = network->addElementWise(*slice_input->getOutput(0), *mean->getOutput(0), ElementWiseOperation::kSUB);
		auto epsAndvar = network->addElementWise(*var->getOutput(0), *eps_tensor->getOutput(0), ElementWiseOperation::kSUM);
		IElementWiseLayer* normalized = network->addElementWise(*inputSubmean->getOutput(0), *epsAndvar->getOutput(0), ElementWiseOperation::kDIV);
		norm_vec.push_back(normalized->getOutput(0));
	}
	ITensor** t_normalized;
	if (group_num == 4)
	{
		ITensor* tensor_normalized[] = { norm_vec[0] ,norm_vec[1] ,norm_vec[2] ,norm_vec[3] };
		t_normalized = tensor_normalized;
	}
	else if (group_num == 8)
	{
		ITensor* tensor_normalized[] = { norm_vec[0] ,norm_vec[1] ,norm_vec[2] ,norm_vec[3],
										 norm_vec[4] ,norm_vec[5] ,norm_vec[6] ,norm_vec[7] };
		t_normalized = tensor_normalized;
	}
	else if (group_num == 16)
	{
		ITensor* tensor_normalized[] = { norm_vec[0] ,norm_vec[1] ,norm_vec[2] ,norm_vec[3],
										 norm_vec[4] ,norm_vec[5] ,norm_vec[6] ,norm_vec[7],
										 norm_vec[8] ,norm_vec[9] ,norm_vec[10] ,norm_vec[11],
										 norm_vec[12] ,norm_vec[13] ,norm_vec[14] ,norm_vec[15] };
		t_normalized = tensor_normalized;
	}
	auto concat = network->addConcatenation(t_normalized, group_num);
	concat->setAxis(0);


	float* pval3 = reinterpret_cast<float*>(malloc(sizeof(float) * len));
	std::fill_n(pval3, len, 1.0);
	Weights power{ DataType::kFLOAT, pval3, len };
	weightMap[lname + ".power3"] = power;

	IScaleLayer* scale_1 = network->addScale(*concat->getOutput(0), ScaleMode::kCHANNEL, weightMap[lname + ".bias"], weightMap[lname + ".weight"], power);  //为网络添加缩放层
	assert(scale_1);
	return scale_1;
}

ILayer* CGR_kernel(INetworkDefinition *network, std::map<std::string, nvinfer1::Weights>& weightMap, ITensor& input, int outch, int ksize, int s, int g, std::string lname, int num, int pad) 
{
	nvinfer1::Weights emptywts{ DataType::kFLOAT, nullptr, 0 };
	std::string conv[] = { ".conv.weight" };
	std::string bn[] = { ".gn" };
	IConvolutionLayer* conv1 = network->addConvolutionNd(input, outch, DimsHW{ ksize, ksize }, weightMap[lname + conv[num]], emptywts);
	assert(conv1);
	conv1->setStrideNd(DimsHW{ s, s });                    // stride 设定卷积步长
	conv1->setPaddingNd(DimsHW{ pad, pad });               // padding 设定padding步长 加零
	conv1->setNbGroups(g);                                 // 设置卷积的组数
	IScaleLayer* bn1 = addGroupNorm2d(network, weightMap, *conv1->getOutput(0), lname + bn[num], 1e-5);

	auto relu = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);
	assert(relu);
	return relu;
}


IConcatenationLayer* gen_x_linspace(INetworkDefinition *network, int feat_size)
{
	Eigen::VectorXd x_arrange = Eigen::VectorXd::LinSpaced(feat_size, -1, 1);
	Weights lin_x{ DataType::kFLOAT, nullptr, x_arrange.size() };
	float *wgt = reinterpret_cast<float *>(malloc(sizeof(float) * x_arrange.size()));
	for (int i = 0; i < x_arrange.size(); ++i)
	{
		wgt[i] = float(x_arrange[i]);
	}
	lin_x.values = wgt;
	IConstantLayer *d = network->addConstant(Dims3{ 1, 1, feat_size }, lin_x);
	ITensor** vec_x_arrange;
	if (feat_size == 80)
	{
		vec_x_arrange = new ITensor* [80];
	}
	else if (feat_size == 40)
	{
		vec_x_arrange = new ITensor* [40];
	}
	else if (feat_size == 20)
	{
		vec_x_arrange = new ITensor* [20];
	}

	for (int i = 0; i < feat_size; i++)
	{
		vec_x_arrange[i] = d->getOutput(0);
	}
	IConcatenationLayer* concat_X = network->addConcatenation(vec_x_arrange, feat_size);
	concat_X->setAxis(1);

	delete [] vec_x_arrange;
	return concat_X;

}

IConcatenationLayer* gen_y_linspace(INetworkDefinition *network, int feat_size)
{
	Eigen::VectorXd y_arrange = Eigen::VectorXd::LinSpaced(feat_size, -1, 1);
	std::vector<IConstantLayer *> lin_vec_y;
	for (int i = 0; i < feat_size; i++)
	{
		Weights lin_y{ DataType::kFLOAT, nullptr, y_arrange.size() };
		float *wgt = reinterpret_cast<float *>(malloc(sizeof(float) * y_arrange.size()));
		for (int j = 0; j < y_arrange.size(); ++j)
		{
			wgt[j] = float(y_arrange[i]);
		}
		lin_y.values = wgt;
		lin_vec_y.push_back(network->addConstant(Dims3{ 1, 1, feat_size }, lin_y));
	}
	
	ITensor** vec_y_arrange;
	if (feat_size == 80)
	{
		vec_y_arrange = new ITensor* [80];
	}
	else if (feat_size == 40)
	{
		vec_y_arrange = new ITensor* [40];
	}
	else if (feat_size == 20)
	{
		vec_y_arrange = new ITensor* [20];
	}
	for (int i = 0; i < feat_size; i++)
	{
		vec_y_arrange[i] = lin_vec_y[i]->getOutput(0);
	}
	IConcatenationLayer* concat_Y = network->addConcatenation(vec_y_arrange, feat_size);
	concat_Y->setAxis(1);

	delete[] vec_y_arrange;
	return concat_Y;
}

IElementWiseLayer* kEQUAL(INetworkDefinition* network, IConvolutionLayer *cate_kernel, IPoolingLayer* point_nms)
{
	IElementWiseLayer* sub1 = network->addElementWise(*cate_kernel->getOutput(0), *point_nms->getOutput(0), ElementWiseOperation::kSUB);
	Dims sub1_dim = sub1->getOutput(0)->getDimensions();
	IElementWiseLayer* div1 = network->addElementWise(*sub1->getOutput(0), *sub1->getOutput(0), ElementWiseOperation::kDIV);
	int size = sub1_dim.d[0] * sub1_dim.d[1] * sub1_dim.d[2];
	Weights lin_x{ DataType::kFLOAT, nullptr, size };
	float *wgt = reinterpret_cast<float *>(malloc(sizeof(float) * size));
	for (int i = 0; i < size; ++i)
	{
		wgt[i] = 1;
	}
	lin_x.values = wgt;

	IConstantLayer *d1 = network->addConstant(Dims3{ sub1_dim.d[0], sub1_dim.d[1], sub1_dim.d[2] }, lin_x);
	//std::cout << d1->getName() << std::endl;
	//Dims dims1 = d1->getOutput(0)->getDimensions();
	//std::cout << dims1.d[0] << std::endl;
	//std::cout << dims1.d[1] << std::endl;
	//std::cout << dims1.d[2] << std::endl;


	IElementWiseLayer* and1 = network->addElementWise(*d1->getOutput(0), *sub1->getOutput(0), ElementWiseOperation::kSUM);
	IElementWiseLayer* and2 = network->addElementWise(*div1->getOutput(0), *sub1->getOutput(0), ElementWiseOperation::kSUM);
	IElementWiseLayer* sub2 = network->addElementWise(*and1->getOutput(0), *and2->getOutput(0), ElementWiseOperation::kSUM);
	return sub2;
};
