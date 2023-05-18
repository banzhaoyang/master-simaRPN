#include "simaRPN++.h"

#define N 999

using Eigen::MatrixXd;

class Logger : public nvinfer1::ILogger
{
public:
	nvinfer1::ILogger::Severity reportableSeverity;

	explicit Logger(nvinfer1::ILogger::Severity severity = nvinfer1::ILogger::Severity::kINFO) :
		reportableSeverity(severity)
	{
	}

	void log(nvinfer1::ILogger::Severity severity, const char* msg) noexcept override
	{
		if (severity > reportableSeverity)
		{
			return;
		}
		switch (severity)
		{
		case nvinfer1::ILogger::Severity::kINTERNAL_ERROR:
			std::cerr << "INTERNAL_ERROR: ";
			break;
		case nvinfer1::ILogger::Severity::kERROR:
			std::cerr << "ERROR: ";
			break;
		case nvinfer1::ILogger::Severity::kWARNING:
			std::cerr << "WARNING: ";
			break;
		case nvinfer1::ILogger::Severity::kINFO:
			std::cerr << "INFO: ";
			break;
		default:
			std::cerr << "VERBOSE: ";
			break;
		}
		std::cerr << msg << std::endl;
	}
}gLogger;

bool simaRPN::gen_template_neck_engine()
{
	DataType dt = nvinfer1::DataType::kFLOAT;
	IBuilder* builder = createInferBuilder(gLogger);
	//INetworkDefinition* network = builder->createNetworkV2(1U << static_cast<uint32_t> (NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));
	IBuilderConfig* config = builder->createBuilderConfig();
	INetworkDefinition* network = builder->createNetworkV2(0U);

	ITensor* data = network->addInput(_input_name[0], dt, Dims3{ 3, _template_input_h, _template_input_w });
	assert(data);    //检查data
	_weightMap = loadWeights(wts_name);  //加载权重

	auto conv1 = convBlock_nonpadding(network, _weightMap, *data, 64, 7, 2, 1, "backbone", 0, true);
	auto pool = addMaxpool(network, *conv1->getOutput(0), 3, 2);

	//---------------------------------------backbone//backbone----------------------------------------//

	std::vector<ILayer*> out = backbone(network, _weightMap, pool);

	//-------------------------------------------neck//neck--------------------------------------------//

	auto neckout1 = neck_convBlock(network, _weightMap, *out[0]->getOutput(0), 256, 1, 1, 1, "neck", 0);
	auto neckout2 = neck_convBlock(network, _weightMap, *out[1]->getOutput(0), 256, 1, 1, 1, "neck", 1);
	auto neckout3 = neck_convBlock(network, _weightMap, *out[2]->getOutput(0), 256, 1, 1, 1, "neck", 2);

	auto neckout_slice1 = network->addSlice(*neckout1->getOutput(0), Dims3{ 0, 4, 4 }, Dims3{ 256, 7, 7 }, Dims3{ 1, 1, 1 });
	auto neckout_slice2 = network->addSlice(*neckout2->getOutput(0), Dims3{ 0, 4, 4 }, Dims3{ 256, 7, 7 }, Dims3{ 1, 1, 1 });
	auto neckout_slice3 = network->addSlice(*neckout3->getOutput(0), Dims3{ 0, 4, 4 }, Dims3{ 256, 7, 7 }, Dims3{ 1, 1, 1 });

	ITensor* neckout_cat[] = { neckout_slice1->getOutput(0) ,neckout_slice2->getOutput(0) ,neckout_slice3->getOutput(0) };

	auto neckout = network->addConcatenation(neckout_cat, 3);
	neckout->setAxis(0);

	network->markOutput(*neckout->getOutput(0));

	builder->setMaxBatchSize(1);                  // 设置最大batchsize
	config->setMaxWorkspaceSize(16 * (1 << 20));  // 16MB（设置最大工作空间）

	if (_engine_mode == 16) {
		//如果没有全局定时缓存附加到构建器实例，则构建器将创建自己的本地缓存，并在构建器完成时销毁它。可以通过设置构建器标志(BuilderFlag)来关闭全局/本地缓存。
		config->setFlag(BuilderFlag::kFP16);       //BuilderFlag::kFP16   kill FP16
		std::cout << "using engine FP16 ... " << std::endl;
	}
	else if (_engine_mode == 8) {
		//std::cout << "Your platform support int8: " << (builder->platformHasFastInt8() ? "true" : "false") << std::endl;
		//assert(builder->platformHasFastInt8());
		//config->setFlag(BuilderFlag::kINT8);
		//std::cout << "using engine INT8 ... " << std::endl;
		//Int8EntropyCalibrator2* calibrator = new Int8EntropyCalibrator2(1, _input_w, _input_h, "./coco_calib/", "int8calib.table", INPUT_BLOB_NAME);
		//config->setInt8Calibrator(calibrator);
	}
	else if (_engine_mode = 32)
	{
		std::cout << "using engine FP32 ... " << std::endl;
	}
	// 使用Builder对象构建引擎（网络，配置）
	std::cout << "Building engine, please wait for a while..." << std::endl;
	ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
	std::cout << "Build engine successfully!" << std::endl;
	// Don't need the network any more
	network->destroy();
	config->destroy();
	builder->destroy();
	nvinfer1::IHostMemory *seriallizedModel = engine->serialize();
	// save engine1
	assert(seriallizedModel != nullptr);
	std::ofstream p(_template_neck_engine_name, std::ios::binary);

	if (!p)
	{
		std::cerr << "could not open plan output file" << std::endl;
		return false;
	}
	p.write(reinterpret_cast<const char*>(seriallizedModel->data()), seriallizedModel->size());
	seriallizedModel->destroy();
	engine->destroy();
}

bool simaRPN::gen_search_neck_engine() 
{
	DataType dt = nvinfer1::DataType::kFLOAT;
	IBuilder* builder = createInferBuilder(gLogger);
	//INetworkDefinition* network = builder->createNetworkV2(1U << static_cast<uint32_t> (NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));
	IBuilderConfig* config = builder->createBuilderConfig();
	INetworkDefinition* network = builder->createNetworkV2(0U);

	ITensor* data = network->addInput(_input_name[0], dt, Dims3{ 3, _input_h, _input_w });
	assert(data);    //检查data
	//_weightMap = loadWeights(wts_name);  //加载权重

	auto conv1 = convBlock_nonpadding(network, _weightMap, *data, 64, 7, 2, 1, "backbone", 0, true);
	auto pool = addMaxpool(network, *conv1->getOutput(0), 3, 2);

	//---------------------------------------backbone//backbone----------------------------------------//

	std::vector<ILayer*> out = backbone(network, _weightMap, pool);

	//-------------------------------------------neck//neck--------------------------------------------//

	auto neckout1 = neck_convBlock(network, _weightMap, *out[0]->getOutput(0), 256, 1, 1, 1, "neck", 0);
	auto neckout2 = neck_convBlock(network, _weightMap, *out[1]->getOutput(0), 256, 1, 1, 1, "neck", 1);
	auto neckout3 = neck_convBlock(network, _weightMap, *out[2]->getOutput(0), 256, 1, 1, 1, "neck", 2);


	ITensor* neckout_cat[] = { neckout1->getOutput(0) ,neckout2->getOutput(0) ,neckout3->getOutput(0) };

	auto neckout = network->addConcatenation(neckout_cat, 3);
	neckout->setAxis(0);

	network->markOutput(*neckout->getOutput(0));

	builder->setMaxBatchSize(1);                  // 设置最大batchsize
	config->setMaxWorkspaceSize(16 * (1 << 20));  // 16MB（设置最大工作空间）

	if (_engine_mode == 16) {
		//如果没有全局定时缓存附加到构建器实例，则构建器将创建自己的本地缓存，并在构建器完成时销毁它。可以通过设置构建器标志(BuilderFlag)来关闭全局/本地缓存。
		config->setFlag(BuilderFlag::kFP16);       //BuilderFlag::kFP16   kill FP16
		std::cout << "using engine FP16 ... " << std::endl;
	}
	else if (_engine_mode == 8) {
		//std::cout << "Your platform support int8: " << (builder->platformHasFastInt8() ? "true" : "false") << std::endl;
		//assert(builder->platformHasFastInt8());
		//config->setFlag(BuilderFlag::kINT8);
		//std::cout << "using engine INT8 ... " << std::endl;
		//Int8EntropyCalibrator2* calibrator = new Int8EntropyCalibrator2(1, _input_w, _input_h, "./coco_calib/", "int8calib.table", INPUT_BLOB_NAME);
		//config->setInt8Calibrator(calibrator);
	}
	else if (_engine_mode = 32)
	{
		std::cout << "using engine FP32 ... " << std::endl;
	}
	// 使用Builder对象构建引擎（网络，配置）
	std::cout << "Building engine, please wait for a while..." << std::endl;
	ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
	std::cout << "Build engine successfully!" << std::endl;
	// Don't need the network any more
	network->destroy();
	config->destroy();
	builder->destroy();
	nvinfer1::IHostMemory *seriallizedModel = engine->serialize();
	// save engine1
	assert(seriallizedModel != nullptr);
	std::ofstream p(_search_neck_engine_name, std::ios::binary);

	if (!p)
	{
		std::cerr << "could not open plan output file" << std::endl;
		return false;
	}
	p.write(reinterpret_cast<const char*>(seriallizedModel->data()), seriallizedModel->size());
	seriallizedModel->destroy();
	engine->destroy();
	return true;
}

bool simaRPN::gen_head_engine()
{

	DataType dt = nvinfer1::DataType::kFLOAT;
	IBuilder* builder = nvinfer1::createInferBuilder(gLogger);
	//nvinfer1::INetworkDefinition* network = builder->createNetworkV2(1U << static_cast<uint32_t> (nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));
	IBuilderConfig* config = builder->createBuilderConfig();
	INetworkDefinition* network = builder->createNetworkV2(0U);

	ITensor* data_srh = network->addInput(_input_name[0], dt, Dims3{ 3 * _neckOutput_channel, _neckOutput_h, _neckOutput_w });

	ITensor* data_tep = network->addInput(_input_name[1], dt, Dims3{ 3 * _neckOutput_channel, 7, 7 });

	assert(data_srh);
	assert(data_tep);

	_weightMap = loadWeights(wts_name);  //加载权重

	//-------------------------------------slice_neckOutput//slice_neckOutput--------------------------------------//

	auto data_tep1 = network->addSlice(*data_tep, Dims3{ _neckOutput_channel * 0, 0, 0 }, Dims3{ _neckOutput_channel, 7, 7 }, Dims3{ 1, 1, 1 });
	auto data_tep2 = network->addSlice(*data_tep, Dims3{ _neckOutput_channel * 1, 0, 0 }, Dims3{ _neckOutput_channel, 7, 7 }, Dims3{ 1, 1, 1 });
	auto data_tep3 = network->addSlice(*data_tep, Dims3{ _neckOutput_channel * 2, 0, 0 }, Dims3{ _neckOutput_channel, 7, 7 }, Dims3{ 1, 1, 1 });

	auto data_srh1 = network->addSlice(*data_srh, Dims3{ _neckOutput_channel * 0, 0, 0 }, Dims3{ _neckOutput_channel, _neckOutput_h, _neckOutput_w }, Dims3{ 1, 1, 1 });
	auto data_srh2 = network->addSlice(*data_srh, Dims3{ _neckOutput_channel * 1, 0, 0 }, Dims3{ _neckOutput_channel, _neckOutput_h, _neckOutput_w }, Dims3{ 1, 1, 1 });
	auto data_srh3 = network->addSlice(*data_srh, Dims3{ _neckOutput_channel * 2, 0, 0 }, Dims3{ _neckOutput_channel, _neckOutput_h, _neckOutput_w }, Dims3{ 1, 1, 1 });

	//-------------------------------------cls_weight//reg_weight--------------------------------------//

	//int height = 25;
	//int weight = 25;          // 有可能wts文件理的weight和输入给的weight不一样
	//nvinfer1::Dims postDims{ 3,
	//						{3, height, weight},
	//						{nvinfer1::DimensionType::kCHANNEL,
	//						 nvinfer1::DimensionType::kSPATIAL,
	//						 nvinfer1::DimensionType::kSPATIAL} };
	//int size = 3 * weight * height;
	//nvinfer1::Weights clsWt{ nvinfer1::DataType::kFLOAT, nullptr, size };
	//nvinfer1::Weights regWt{ nvinfer1::DataType::kFLOAT, nullptr, size };
	//float* clsWtf = new float[size];
	//float* regWtf = new float[size];
	//for (int i = 0, idx = 0; i < 3; ++i)
	//{
	//	for (int s = 0; s < height; ++s)
	//	{
	//		for (int j = 0; j < weight; ++j, ++idx)
	//		{
	//			if (i == 0) {
	//				clsWtf[idx] = 0.3869f;
	//				regWtf[idx] = 0.2343f;
	//			}
	//			if (i == 1) {
	//				clsWtf[idx] = 0.4622f;
	//				regWtf[idx] = 0.3832f;
	//			}
	//			if (i == 2) {
	//				clsWtf[idx] = 0.1508f;
	//				regWtf[idx] = 0.3825f;
	//			}
	//		}
	//	}
	//}	
	//clsWt.values = clsWtf;
	//regWt.values = regWtf;
	//auto cls_weight = network->addConstant(postDims, clsWt);
	//auto reg_weight = network->addConstant(postDims, regWt);
	//assert(cls_weight != nullptr);
	//assert(reg_weight != nullptr);

	auto cls_weight = network->addConstant(Dims3{ 3, 1, 1 }, _weightMap["head.cls_weight"]);
	auto reg_weight = network->addConstant(Dims3{ 3, 1, 1 }, _weightMap["head.reg_weight"]);
	auto cls_weight_somax = network->addSoftMax(*cls_weight->getOutput(0));
	auto reg_weight_somax = network->addSoftMax(*reg_weight->getOutput(0));

	auto cls_weight_somax1 = network->addSlice(*cls_weight_somax->getOutput(0), Dims3{ 0, 0, 0 }, Dims3{ 1, 1, 1 }, Dims3{ 1, 1, 1 });
	auto cls_weight_somax2 = network->addSlice(*cls_weight_somax->getOutput(0), Dims3{ 1, 0, 0 }, Dims3{ 1, 1, 1 }, Dims3{ 1, 1, 1 });
	auto cls_weight_somax3 = network->addSlice(*cls_weight_somax->getOutput(0), Dims3{ 2, 0, 0 }, Dims3{ 1, 1, 1 }, Dims3{ 1, 1, 1 });

	auto reg_weight_somax1 = network->addSlice(*reg_weight_somax->getOutput(0), Dims3{ 0, 0, 0 }, Dims3{ 1, 1, 1 }, Dims3{ 1, 1, 1 });
	auto reg_weight_somax2 = network->addSlice(*reg_weight_somax->getOutput(0), Dims3{ 1, 0, 0 }, Dims3{ 1, 1, 1 }, Dims3{ 1, 1, 1 });
	auto reg_weight_somax3 = network->addSlice(*reg_weight_somax->getOutput(0), Dims3{ 2, 0, 0 }, Dims3{ 1, 1, 1 }, Dims3{ 1, 1, 1 });


	////---------------------------------------cls_head//cls_head----------------------------------------//

	auto cls_kernel_convs1 = head_convBlock(network, _weightMap, *data_tep1->getOutput(0), 256, 3, 1, 1, "head.cls_heads.0.kernel_convs");
	auto cls_search_convs1 = head_convBlock(network, _weightMap, *data_srh1->getOutput(0), 256, 3, 1, 1, "head.cls_heads.0.search_convs");
	auto cls_correlation_maps1 = depthwise_correlation(network, *cls_search_convs1->getOutput(0), *cls_kernel_convs1->getOutput(0));

	auto cls_head_convs1 = head_convBlock(network, _weightMap, *cls_correlation_maps1->getOutput(0), 256, 1, 1, 1, "head.cls_heads.0.head_convs.0");
	auto cls_head_out1 = head_head(network, _weightMap, *cls_head_convs1->getOutput(0), 10, 1, 1, 1, "head.cls_heads.0.head_convs.1.conv");

	auto cls_socre1 = network->addElementWise(*cls_weight_somax1->getOutput(0), *cls_head_out1->getOutput(0), ElementWiseOperation::kPROD);

	//-↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓-//

	auto cls_kernel_convs2 = head_convBlock(network, _weightMap, *data_tep2->getOutput(0), 256, 3, 1, 1, "head.cls_heads.1.kernel_convs");
	auto cls_search_convs2 = head_convBlock(network, _weightMap, *data_srh2->getOutput(0), 256, 3, 1, 1, "head.cls_heads.1.search_convs");
	auto cls_correlation_maps2 = depthwise_correlation(network, *cls_search_convs2->getOutput(0), *cls_kernel_convs2->getOutput(0));   //402

	auto cls_head_convs2 = head_convBlock(network, _weightMap, *cls_correlation_maps2->getOutput(0), 256, 1, 1, 1, "head.cls_heads.1.head_convs.0");
	auto cls_head_out2 = head_head(network, _weightMap, *cls_head_convs2->getOutput(0), 10, 1, 1, 1, "head.cls_heads.1.head_convs.1.conv");

	IElementWiseLayer* cls_socre2 = network->addElementWise(*cls_weight_somax2->getOutput(0), *cls_head_out2->getOutput(0), ElementWiseOperation::kPROD);
	auto cls_score3 = network->addElementWise(*cls_socre2->getOutput(0), *cls_socre1->getOutput(0), ElementWiseOperation::kSUM);


	//-↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓-//

	auto cls_kernel_convs3 = head_convBlock(network, _weightMap, *data_tep3->getOutput(0), 256, 3, 1, 1, "head.cls_heads.2.kernel_convs");
	auto cls_search_convs3 = head_convBlock(network, _weightMap, *data_srh3->getOutput(0), 256, 3, 1, 1, "head.cls_heads.2.search_convs");
	auto cls_correlation_maps3 = depthwise_correlation(network, *cls_search_convs3->getOutput(0), *cls_kernel_convs3->getOutput(0));

	auto cls_head_convs3 = head_convBlock(network, _weightMap, *cls_correlation_maps3->getOutput(0), 256, 1, 1, 1, "head.cls_heads.2.head_convs.0");
	auto cls_head_out3 = head_head(network, _weightMap, *cls_head_convs3->getOutput(0), 10, 1, 1, 1, "head.cls_heads.2.head_convs.1.conv");

	auto cls_socre4 = network->addElementWise(*cls_weight_somax3->getOutput(0), *cls_head_out3->getOutput(0), ElementWiseOperation::kPROD);
	IElementWiseLayer* cls_score_final = network->addElementWise(*cls_score3->getOutput(0), *cls_socre4->getOutput(0), ElementWiseOperation::kSUM);

	//---------------------------------------reg_head//reg_head----------------------------------------//

	auto reg_kernel_convs1 = head_convBlock(network, _weightMap, *data_tep1->getOutput(0), 256, 3, 1, 1, "head.reg_heads.0.kernel_convs");
	auto reg_search_convs1 = head_convBlock(network, _weightMap, *data_srh1->getOutput(0), 256, 3, 1, 1, "head.reg_heads.0.search_convs");
	auto reg_correlation_maps1 = depthwise_correlation(network, *reg_search_convs1->getOutput(0), *reg_kernel_convs1->getOutput(0));

	auto reg_head_convs1 = head_convBlock(network, _weightMap, *reg_correlation_maps1->getOutput(0), 256, 1, 1, 1, "head.reg_heads.0.head_convs.0");
	auto reg_head_out1 = head_head(network, _weightMap, *reg_head_convs1->getOutput(0), 20, 1, 1, 1, "head.reg_heads.0.head_convs.1.conv");
	
	auto bbox_pred1 = network->addElementWise(*reg_weight_somax1->getOutput(0), *reg_head_out1->getOutput(0), ElementWiseOperation::kPROD);

	//-↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓-//

	auto reg_kernel_convs2 = head_convBlock(network, _weightMap, *data_tep2->getOutput(0), 256, 3, 1, 1, "head.reg_heads.1.kernel_convs");
	auto reg_search_convs2 = head_convBlock(network, _weightMap, *data_srh2->getOutput(0), 256, 3, 1, 1, "head.reg_heads.1.search_convs");
	auto reg_correlation_maps2 = depthwise_correlation(network, *reg_search_convs2->getOutput(0), *reg_kernel_convs2->getOutput(0));

	auto reg_head_convs2 = head_convBlock(network, _weightMap, *reg_correlation_maps2->getOutput(0), 256, 1, 1, 1, "head.reg_heads.1.head_convs.0");
	auto reg_head_out2 = head_head(network, _weightMap, *reg_head_convs2->getOutput(0), 20, 1, 1, 1, "head.reg_heads.1.head_convs.1.conv");

	auto bbox_pred2 = network->addElementWise(*reg_weight_somax2->getOutput(0), *reg_head_out2->getOutput(0), ElementWiseOperation::kPROD);
	auto bbox_pred3 = network->addElementWise(*bbox_pred2->getOutput(0), *bbox_pred1->getOutput(0), ElementWiseOperation::kSUM);

	//-↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓-//

	auto reg_kernel_convs3 = head_convBlock(network, _weightMap, *data_tep3->getOutput(0), 256, 3, 1, 1, "head.reg_heads.2.kernel_convs");
	auto reg_search_convs3 = head_convBlock(network, _weightMap, *data_srh3->getOutput(0), 256, 3, 1, 1, "head.reg_heads.2.search_convs");
	auto reg_correlation_maps3 = depthwise_correlation(network, *reg_search_convs3->getOutput(0), *reg_kernel_convs3->getOutput(0));

	auto reg_head_convs3 = head_convBlock(network, _weightMap, *reg_correlation_maps3->getOutput(0), 256, 1, 1, 1, "head.reg_heads.2.head_convs.0");
	auto reg_head_out3 = head_head(network, _weightMap, *reg_head_convs3->getOutput(0), 20, 1, 1, 1, "head.reg_heads.2.head_convs.1.conv");

	auto bbox_pred4 = network->addElementWise(*reg_weight_somax3->getOutput(0), *reg_head_out3->getOutput(0), ElementWiseOperation::kPROD);
	IElementWiseLayer* bbox_pred_final = network->addElementWise(*bbox_pred4->getOutput(0), *bbox_pred3->getOutput(0), ElementWiseOperation::kSUM);

	cls_score_final->getOutput(0)->setName(_output_name[0]);
	bbox_pred_final->getOutput(0)->setName(_output_name[1]);

	network->markOutput(*cls_score_final->getOutput(0));
	network->markOutput(*bbox_pred_final->getOutput(0));	
	
	builder->setMaxBatchSize(1);                         // 设置最大batchsize
	config->setMaxWorkspaceSize(16 * (1 << 20));         // 16MB（设置最大工作空间）

	if (_engine_mode == 16) {
		//如果没有全局定时缓存附加到构建器实例,则构建器将创建自己的本地缓存,并在构建器完成时销毁它。可以通过设置构建器标志(BuilderFlag)来关闭全局/本地缓存。
		config->setFlag(BuilderFlag::kFP16);       //BuilderFlag::kFP16   kill FP16
		std::cout << "using engine FP16 ... " << std::endl;
	}
	else if (_engine_mode == 8) {
		//std::cout << "Your platform support int8: " << (builder->platformHasFastInt8() ? "true" : "false") << std::endl;
		//assert(builder->platformHasFastInt8());
		//config->setFlag(BuilderFlag::kINT8);
		//std::cout << "using engine INT8 ... " << std::endl;
		//Int8EntropyCalibrator2* calibrator = new Int8EntropyCalibrator2(1, _input_w, _input_h, "./coco_calib/", "int8calib.table", INPUT_BLOB_NAME);
		//config->setInt8Calibrator(calibrator);
	}
	else if (_engine_mode = 32)
	{
		std::cout << "using engine FP32 ... " << std::endl;
	}

	// 使用Builder对象构建引擎（网络，配置）
	std::cout << "Building engine, please wait for a while..." << std::endl;
	ICudaEngine* engine2 = builder->buildEngineWithConfig(*network, *config);
	std::cout << "Build engine successfully!" << std::endl;
	// Don't need the network any more
	//network->destroy();
	nvinfer1::IHostMemory *seriallizedModel2 = engine2->serialize();
	// save engine1
	assert(seriallizedModel2 != nullptr);
	std::ofstream p(_head_engine_name, std::ios::binary);
	if (!p)
	{
		std::cerr << "could not open plan output file" << std::endl;
		return false;
	}
	p.write(reinterpret_cast<const char*>(seriallizedModel2->data()), seriallizedModel2->size());
	seriallizedModel2->destroy();
};

std::vector<ILayer*> simaRPN::backbone(nvinfer1::INetworkDefinition* network, std::map<std::string, nvinfer1::Weights>& weightMap, IPoolingLayer* pool) {
	
	//------------------------------layer1//layer1-----------------------------//

	auto conv2 = convBlock(network, weightMap, *pool->getOutput(0), 64, 1, 1, 1, "backbone.layer1.0", 0, true);
	auto conv3 = convBlock(network, weightMap, *conv2->getOutput(0), 64, 3, 1, 1, "backbone.layer1.0", 1, true);
	auto conv4 = convBlock(network, weightMap, *conv3->getOutput(0), 256, 1, 1, 1, "backbone.layer1.0", 2, false);
	auto downsample1 = downsample_nonpadding(network, weightMap, *pool->getOutput(0), 256, 1, 1, 1, "backbone.layer1.0.downsample");
	auto sum1 = network->addElementWise(*conv4->getOutput(0), *downsample1->getOutput(0), ElementWiseOperation::kSUM);
	auto relu1 = network->addActivation(*sum1->getOutput(0),ActivationType::kRELU);

	auto conv5 = convBlock(network, weightMap, *relu1->getOutput(0), 64, 1, 1, 1, "backbone.layer1.1", 0, true);
	auto conv6 = convBlock(network, weightMap, *conv5->getOutput(0), 64, 3, 1, 1, "backbone.layer1.1", 1, true);
	auto conv7 = convBlock(network, weightMap, *conv6->getOutput(0), 256, 1, 1, 1, "backbone.layer1.1", 2, false);
	auto sum2 = network->addElementWise(*conv7->getOutput(0), *relu1->getOutput(0), ElementWiseOperation::kSUM);
	auto relu2 = network->addActivation(*sum2->getOutput(0), ActivationType::kRELU);


	auto conv8 = convBlock(network, weightMap, *relu2->getOutput(0), 64, 1, 1, 1, "backbone.layer1.2", 0, true);
	auto conv9 = convBlock(network, weightMap, *conv8->getOutput(0), 64, 3, 1, 1, "backbone.layer1.2", 1, true);
	auto conv10 = convBlock(network, weightMap, *conv9->getOutput(0), 256, 1, 1, 1, "backbone.layer1.2", 2, false);
	auto sum3 = network->addElementWise(*conv10->getOutput(0), *relu2->getOutput(0), ElementWiseOperation::kSUM);
	auto relu3 = network->addActivation(*sum3->getOutput(0), ActivationType::kRELU);



	//------------------------------layer2//layer2-----------------------------//

	auto conv11 = convBlock(network, weightMap, *relu3->getOutput(0), 128, 1, 1, 1, "backbone.layer2.0", 0, true);
	auto conv12 = convBlock_nonpadding(network, weightMap, *conv11->getOutput(0), 128, 3, 2, 1, "backbone.layer2.0", 1, true);
	auto conv13 = convBlock(network, weightMap, *conv12->getOutput(0), 512, 1, 1, 1, "backbone.layer2.0", 2, false);
	auto downsample2 = downsample_nonpadding(network, weightMap, *relu3->getOutput(0), 512, 3, 2, 1, "backbone.layer2.0.downsample");
	auto sum4 = network->addElementWise(*downsample2->getOutput(0), *conv13->getOutput(0), ElementWiseOperation::kSUM);
	auto relu4 = network->addActivation(*sum4->getOutput(0), ActivationType::kRELU);


	auto conv14 = convBlock(network, weightMap, *relu4->getOutput(0), 128, 1, 1, 1, "backbone.layer2.1", 0, true);
	auto conv15 = convBlock(network, weightMap, *conv14->getOutput(0), 128, 3, 1, 1, "backbone.layer2.1", 1, true);
	auto conv16 = convBlock(network, weightMap, *conv15->getOutput(0), 512, 1, 1, 1, "backbone.layer2.1", 2, false);
	auto sum5 = network->addElementWise(*conv16->getOutput(0), *relu4->getOutput(0), ElementWiseOperation::kSUM);
	auto relu5 = network->addActivation(*sum5->getOutput(0), ActivationType::kRELU);


	auto conv17 = convBlock(network, weightMap, *relu5->getOutput(0), 128, 1, 1, 1, "backbone.layer2.2", 0, true);
	auto conv18 = convBlock(network, weightMap, *conv17->getOutput(0), 128, 3, 1, 1, "backbone.layer2.2", 1, true);
	auto conv19 = convBlock(network, weightMap, *conv18->getOutput(0), 512, 1, 1, 1, "backbone.layer2.2", 2, false);
	auto sum6 = network->addElementWise(*conv19->getOutput(0), *relu5->getOutput(0), ElementWiseOperation::kSUM);
	auto relu6 = network->addActivation(*sum6->getOutput(0), ActivationType::kRELU);


	auto conv20 = convBlock(network, weightMap, *relu6->getOutput(0), 128, 1, 1, 1, "backbone.layer2.3", 0, true);
	auto conv21 = convBlock(network, weightMap, *conv20->getOutput(0), 128, 3, 1, 1, "backbone.layer2.3", 1, true);
	auto conv22 = convBlock(network, weightMap, *conv21->getOutput(0), 512, 1, 1, 1, "backbone.layer2.3", 2, false);
	auto sum7 = network->addElementWise(*conv22->getOutput(0), *relu6->getOutput(0), ElementWiseOperation::kSUM);
	ILayer* out1 = network->addActivation(*sum7->getOutput(0), ActivationType::kRELU);

	//------------------------------layer3//layer3-----------------------------//

	auto conv23 = convBlock(network, weightMap, *out1->getOutput(0), 256, 1, 1, 1, "backbone.layer3.0", 0, true);
	auto conv24 = convBlock(network, weightMap, *conv23->getOutput(0), 256, 3, 1, 1, "backbone.layer3.0", 1, true);
	auto conv25 = convBlock(network, weightMap, *conv24->getOutput(0), 1024, 1, 1, 1, "backbone.layer3.0", 2, false);
	auto downsample3 = downsample(network, weightMap, *out1->getOutput(0), 1024, 3, 1, 1, "backbone.layer3.0.downsample");
	auto sum8 = network->addElementWise(*downsample3->getOutput(0), *conv25->getOutput(0), ElementWiseOperation::kSUM);
	auto relu8 = network->addActivation(*sum8->getOutput(0), ActivationType::kRELU);


	auto conv26 = convBlock(network, weightMap, *relu8->getOutput(0), 256, 1, 1, 1, "backbone.layer3.1", 0, true);
	auto conv27 = convBlock_dilation(network, weightMap, *conv26->getOutput(0), 256, 3, 1, 1, "backbone.layer3.1", 1, 2);           ////
	auto conv28 = convBlock(network, weightMap, *conv27->getOutput(0), 1024, 1, 1, 1, "backbone.layer3.1", 2,false);
	auto sum9 = network->addElementWise(*conv28->getOutput(0), *relu8->getOutput(0), ElementWiseOperation::kSUM);
	auto relu9 = network->addActivation(*sum9->getOutput(0), ActivationType::kRELU);

	auto conv29 = convBlock(network, weightMap, *relu9->getOutput(0), 256, 1, 1, 1, "backbone.layer3.2", 0, true);
	auto conv30 = convBlock_dilation(network, weightMap, *conv29->getOutput(0), 256, 3, 1, 1, "backbone.layer3.2", 1, 2);
	auto conv31 = convBlock(network, weightMap, *conv30->getOutput(0), 1024, 1, 1, 1, "backbone.layer3.2", 2, false);
	auto sum10 = network->addElementWise(*conv31->getOutput(0), *relu9->getOutput(0), ElementWiseOperation::kSUM);
	auto relu10 = network->addActivation(*sum10->getOutput(0), ActivationType::kRELU);

	auto conv32 = convBlock(network, weightMap, *relu10->getOutput(0), 256, 1, 1, 1, "backbone.layer3.3", 0, true);
	auto conv33 = convBlock_dilation(network, weightMap, *conv32->getOutput(0), 256, 3, 1, 1, "backbone.layer3.3", 1, 2);
	auto conv34 = convBlock(network, weightMap, *conv33->getOutput(0), 1024, 1, 1, 1, "backbone.layer3.3", 2, false);
	auto sum11 = network->addElementWise(*conv34->getOutput(0), *relu10->getOutput(0), ElementWiseOperation::kSUM);
	auto relu11 = network->addActivation(*sum11->getOutput(0), ActivationType::kRELU);

	auto conv35 = convBlock(network, weightMap, *relu11->getOutput(0), 256, 1, 1, 1, "backbone.layer3.4", 0, true);
	auto conv36 = convBlock_dilation(network, weightMap, *conv35->getOutput(0), 256, 3, 1, 1, "backbone.layer3.4", 1, 2);
	auto conv37 = convBlock(network, weightMap, *conv36->getOutput(0), 1024, 1, 1, 1, "backbone.layer3.4", 2, false);
	auto sum12 = network->addElementWise(*conv37->getOutput(0), *relu11->getOutput(0), ElementWiseOperation::kSUM);
	auto relu12 = network->addActivation(*sum12->getOutput(0), ActivationType::kRELU);

	auto conv38 = convBlock(network, weightMap, *relu12->getOutput(0), 256, 1, 1, 1, "backbone.layer3.5", 0,true);
	auto conv39 = convBlock_dilation(network, weightMap, *conv38->getOutput(0), 256, 3, 1, 1, "backbone.layer3.5", 1, 2);
	auto conv40 = convBlock(network, weightMap, *conv39->getOutput(0), 1024, 1, 1, 1, "backbone.layer3.5", 2, false);
	auto sum13 = network->addElementWise(*conv40->getOutput(0), *relu12->getOutput(0), ElementWiseOperation::kSUM);
	ILayer* out2 = network->addActivation(*sum13->getOutput(0), ActivationType::kRELU);

	//-----------------------------------------layer4//layer4------------------------------------------//

	auto conv41 = convBlock(network, weightMap, *out2->getOutput(0), 512, 1, 1, 1, "backbone.layer4.0", 0, true);
	auto conv42 = convBlock_dilation(network, weightMap, *conv41->getOutput(0), 512, 3, 1, 1, "backbone.layer4.0", 1, 2);
	auto conv43 = convBlock(network, weightMap, *conv42->getOutput(0), 2048, 1, 1, 1, "backbone.layer4.0", 2, false);
	auto downsample4 = downsample_dilation(network, weightMap, *out2->getOutput(0), 2048, 3, 1, 1, "backbone.layer4.0.downsample", 2);
	auto sum14 = network->addElementWise(*downsample4->getOutput(0), *conv43->getOutput(0), ElementWiseOperation::kSUM);
	auto relu14 = network->addActivation(*sum14->getOutput(0), ActivationType::kRELU);

	auto conv44 = convBlock(network, weightMap, *relu14->getOutput(0), 512, 1, 1, 1, "backbone.layer4.1", 0, true);
	auto conv45 = convBlock_dilation(network, weightMap, *conv44->getOutput(0), 512, 3, 1, 1, "backbone.layer4.1", 1, 4);           ////
	auto conv46 = convBlock(network, weightMap, *conv45->getOutput(0), 2048, 1, 1, 1, "backbone.layer4.1", 2, false);
	auto sum15 = network->addElementWise(*conv46->getOutput(0), *relu14->getOutput(0), ElementWiseOperation::kSUM);
	auto relu15 = network->addActivation(*sum15->getOutput(0), ActivationType::kRELU);

	auto conv47 = convBlock(network, weightMap, *relu15->getOutput(0), 512, 1, 1, 1, "backbone.layer4.2", 0, true);
	auto conv48 = convBlock_dilation(network, weightMap, *conv47->getOutput(0), 512, 3, 1, 1, "backbone.layer4.2", 1, 4);
	auto conv49 = convBlock(network, weightMap, *conv48->getOutput(0), 2048, 1, 1, 1, "backbone.layer4.2", 2, false);
	auto sum16 = network->addElementWise(*conv49->getOutput(0), *relu15->getOutput(0), ElementWiseOperation::kSUM);
	ILayer* out3 = network->addActivation(*sum16->getOutput(0), ActivationType::kRELU);
	std::vector<ILayer*> out = {out1, out2, out3};
	return out;
}




bool simaRPN::load_template_neck_engine() 
{
	std::ifstream file_template_neck(_template_neck_engine_name, std::ios::binary);
	if (!file_template_neck.good()) {
		std::cerr << "read " << _template_neck_engine_name << " error!" << std::endl;
		return -1;
	}
	char *trtModelStream_temp_neck = nullptr;                       // nullptr 代表空指针，NULL代表数字0
	size_t size1 = 0;                                               // size_t 相当于 unsigned int 无符号整数类型 size_t在主流平台中都是64位 unsigned int 为32位
	file_template_neck.seekg(0, file_template_neck.end);            // 从流end末尾处开始计算的位移
	size1 = file_template_neck.tellg();                             // 返回一个无符号类型的整数，代表流指针位置
	file_template_neck.seekg(0, file_template_neck.beg);            // 从流begin开始处计算的位移，由于没有tellg()函数得到流指针的位置，所以这里是将 流指针回归到开始处
	trtModelStream_temp_neck = new char[size1];
	assert(trtModelStream_temp_neck);
	file_template_neck.read(trtModelStream_temp_neck, size1);       // read(char *buffer,streamsize size) size是要读取的字符数，buffer为内存的一块地址
	file_template_neck.close();

	_neck_template_input = new float[_batch_size * 3 * _template_input_h * _template_input_w];

	_neck_template_output1 = new float[_batch_size * 3 * _neckOutput_channel * 7 * 7];

	_runtime_template_neck = createInferRuntime(gLogger);      //创建要反序列化的运行时对象
	assert(_runtime_template_neck != nullptr);

	_engine_template_neck = _runtime_template_neck->deserializeCudaEngine(trtModelStream_temp_neck, size1, nullptr);
	assert(_engine_template_neck != nullptr);

	_context_template_neck = _engine_template_neck->createExecutionContext();    //创建上下文环境，主要用与inference函数中启动cuda核
	assert(_context_template_neck != nullptr);

	delete[] trtModelStream_temp_neck;
	assert(_engine_template_neck->getNbBindings() == 2);

}

bool simaRPN::load_search_neck_engine() 
{
	std::ifstream file_search_neck(_search_neck_engine_name, std::ios::binary);
	if (!file_search_neck.good()) {
		std::cerr << "read " << _search_neck_engine_name << " error!" << std::endl;
		return -1;
	}
	char *trtModelStream_search_neck = nullptr;                     // nullptr 代表空指针，NULL代表数字0
	size_t size2 = 0;                                               // size_t 相当于 unsigned int 无符号整数类型 size_t在主流平台中都是64位 unsigned int 为32位
	file_search_neck.seekg(0, file_search_neck.end);                // 从流end末尾处开始计算的位移
	size2 = file_search_neck.tellg();                               // 返回一个无符号类型的整数，代表流指针位置
	file_search_neck.seekg(0, file_search_neck.beg);                // 从流begin开始处计算的位移，由于没有tellg()函数得到流指针的位置，所以这里是将 流指针回归到开始处
	trtModelStream_search_neck = new char[size2];
	assert(trtModelStream_search_neck);
	file_search_neck.read(trtModelStream_search_neck, size2);       // read(char *buffer,streamsize size) size是要读取的字符数，buffer为内存的一块地址
	file_search_neck.close();

	_neck_search_input = new float[_batch_size * 3 * _input_h * _input_w];

	_neck_search_output1 = new float[_batch_size * 3 * _neckOutput_channel * _neckOutput_h * _neckOutput_w];

	_runtime_search_neck = createInferRuntime(gLogger);        //创建要反序列化的运行时对象
	assert(_runtime_search_neck != nullptr);

	_engine_search_neck = _runtime_search_neck->deserializeCudaEngine(trtModelStream_search_neck, size2, nullptr);
	assert(_engine_search_neck != nullptr);

	_context_search_neck = _engine_search_neck->createExecutionContext();        //创建上下文环境，主要用与inference函数中启动cuda核
	assert(_context_search_neck != nullptr);

	delete[] trtModelStream_search_neck;
	assert(_engine_search_neck->getNbBindings() == 2);
}

bool simaRPN::load_head_engine() {
	// deserialize the .engine and run inference

	std::ifstream file_head(_head_engine_name, std::ios::binary);
	if (!file_head.good()) {
		std::cerr << "read " << _head_engine_name << " error!" << std::endl;
		return -1;
	}
	char *trtModelStream_head = nullptr;            // nullptr 代表空指针，NULL代表数字0
	size_t size = 0;                                // size_t 相当于 unsigned int 无符号整数类型 size_t在主流平台中都是64位 unsigned int 为32位
	file_head.seekg(0, file_head.end);              // 从流end末尾处开始计算的位移
	size = file_head.tellg();                       // 返回一个无符号类型的整数，代表流指针位置
	file_head.seekg(0, file_head.beg);              // 从流begin开始处计算的位移，由于没有tellg()函数得到流指针的位置，所以这里是将 流指针回归到开始处
	trtModelStream_head = new char[size];
	assert(trtModelStream_head);
	file_head.read(trtModelStream_head, size);      // read(char *buffer,streamsize size) size是要读取的字符数，buffer为内存的一块地址
	file_head.close();

	// prepare input data ---------------------------

	_head_input_srh1 = new float[_batch_size * 3 * _neckOutput_channel * _neckOutput_h * _neckOutput_w];
	_head_input_tep1 = new float[_batch_size * 3 * _neckOutput_channel * 7 * 7];

	_head_output_cls = new float[_batch_size * 10 * 25 * 25];
	_head_output_reg = new float[_batch_size * 20 * 25 * 25];

	_runtime_head = createInferRuntime(gLogger);               //创建要反序列化的运行时对象
	assert(_runtime_head != nullptr);

	_engine_head = _runtime_head->deserializeCudaEngine(trtModelStream_head, size, nullptr);
	assert(_engine_head != nullptr);

	_context_head = _engine_head->createExecutionContext();                      //创建上下文环境，主要用与inference函数中启动cuda核
	assert(_context_head != nullptr);

	delete[] trtModelStream_head;

	assert(_engine_head->getNbBindings() == 4);
}



cv::Mat simaRPN::_get_crop_frame(cv::Rect2d init_bbox, cv::Mat img, int frame_id)
{
	typedef cv::Rect_<double> Rect2d;
	init_bbox = cv::Rect2d( init_bbox.x + init_bbox.width / 2.0, init_bbox.y + init_bbox.height / 2.0, init_bbox.width, init_bbox.height);
	float z_width = init_bbox.width + (init_bbox.width + init_bbox.height) / 2.0;
	float z_height = init_bbox.height + (init_bbox.width + init_bbox.height) / 2.0;
	_crop_z_size = std::round(sqrt(z_height * z_width));
	_z_size = sqrt(z_height * z_width);
	if (frame_id != 0)
	{
		_crop_x_size = std::round(_z_size * (float(_input_h) / float(_template_input_h)));
		_crop_z_size = _crop_x_size;
	}
	cv::Scalar channel_avg = cv::mean(img);
	cv::Mat crop_img;
	cv::Mat out_img;

	int context_xmin = int(init_bbox.x - _crop_z_size / 2.0);
	int context_xmax = int(init_bbox.x + _crop_z_size / 2.0);
	int context_ymin = int(init_bbox.y - _crop_z_size / 2.0);
	int context_ymax = int(init_bbox.y + _crop_z_size / 2.0);
	
	int left_pad = std::max(0, - context_xmin);
	int top_pad = std::max(0, - context_ymin);
	int right_pad = std::max(0, context_xmax - img.cols);
	int bottom_pad = std::max(0, context_ymax - img.rows);

	context_xmin += left_pad;
	context_xmax += left_pad;
	context_ymin += top_pad;
	context_ymax += top_pad;

	if (left_pad || top_pad || right_pad || bottom_pad)
	{
		cv::Mat new_img = cv::Mat::zeros(img.rows + top_pad + bottom_pad, img.cols + left_pad + right_pad, CV_8UC3);
		img(cv::Rect(0, 0, img.cols, img.rows )).copyTo(new_img(cv::Rect(left_pad, top_pad, img.cols, img.rows)));
		if (top_pad)
			new_img(cv::Rect(0, top_pad, img.cols, top_pad)) = channel_avg;                      // channel_avg 为四维向量
		if (bottom_pad)
			new_img(cv::Rect(0, img.rows, img.cols, bottom_pad)) = channel_avg;
		if (left_pad)
			new_img(cv::Rect(0, 0, left_pad, img.rows)) = channel_avg;
		if (right_pad)
			new_img(cv::Rect(img.cols, 0, right_pad, img.rows)) = channel_avg;

		new_img(cv::Rect(context_xmin, context_ymin, context_xmax - context_xmin, context_ymax - context_ymin)).copyTo(crop_img);
		if (frame_id == 0)
		{
			cv::resize(crop_img, out_img, { _template_input_h, _template_input_w }, 1, 1, cv::INTER_LINEAR);
		}
		else
		{
			cv::resize(crop_img, out_img, { _input_h, _input_w }, 1, 1, cv::INTER_LINEAR);
		}

		return out_img;
	}
	else 
	{
		img(cv::Rect(context_xmin, context_ymin, context_xmax - context_xmin, context_ymax - context_ymin)).copyTo(crop_img);
		if (frame_id == 0)
		{
			cv::resize(crop_img, out_img, { _template_input_h, _template_input_w }, 1, 1, cv::INTER_LINEAR);
		}
		else
		{
			cv::resize(crop_img, out_img, { _input_h, _input_w }, 1, 1, cv::INTER_LINEAR);
		}
		return out_img;
	}
};

void simaRPN::_image_inference(int frame_id) 
{
	if (frame_id == 0)
	{
		CUDA_CHECK(cudaMalloc(&_buffers_template_neck[0], _batch_size * 3 * _template_input_h * _template_input_h * sizeof(float)));
		CUDA_CHECK(cudaMalloc(&_buffers_template_neck[1], _batch_size * 3 * _neckOutput_channel * 7 * 7 * sizeof(float)));

		CUDA_CHECK(cudaStreamCreate(&_stream_template_neck));

		//---------------------------------------input---------------------------------------//
		cudaMemcpyAsync(_buffers_template_neck[0], _neck_template_input, _batch_size * 3 * _template_input_h * _template_input_h * sizeof(float), cudaMemcpyHostToDevice, _stream_template_neck);
		
		//-------------------------------------inference-------------------------------------//
		_context_template_neck->enqueue(_batch_size, _buffers_template_neck, _stream_template_neck, nullptr);

		//--------------------------------------output---------------------------------------//
		cudaMemcpyAsync(_neck_template_output1, _buffers_template_neck[1], _batch_size * 3 * _neckOutput_channel * 7 * 7 * sizeof(float), cudaMemcpyDeviceToHost, _stream_template_neck);

		CUDA_CHECK(cudaFree(_buffers_template_neck[0]));
		CUDA_CHECK(cudaFree(_buffers_template_neck[1]));

		cudaStreamDestroy(_stream_template_neck);

		_head_input_tep1 = _neck_template_output1;
	}
	else 
	{
		CUDA_CHECK(cudaMalloc(&_buffers_search_neck[0], _batch_size * 3 * _input_h * _input_w * sizeof(float)));
		CUDA_CHECK(cudaMalloc(&_buffers_search_neck[1], _batch_size * 3 * _neckOutput_channel * _neckOutput_h * _neckOutput_w * sizeof(float)));

		CUDA_CHECK(cudaStreamCreate(&_stream_search_neck));

		//---------------------------------------input---------------------------------------//
		cudaMemcpyAsync(_buffers_search_neck[0], _neck_search_input, _batch_size * 3 * _input_h * _input_w * sizeof(float), cudaMemcpyHostToDevice, _stream_search_neck);

		//-------------------------------------inference-------------------------------------//
		_context_search_neck->enqueue(_batch_size, _buffers_search_neck, _stream_search_neck, nullptr);

		//--------------------------------------output---------------------------------------//
		cudaMemcpyAsync(_neck_search_output1, _buffers_search_neck[1], _batch_size * 3 * _neckOutput_channel * _neckOutput_h * _neckOutput_w * sizeof(float), cudaMemcpyDeviceToHost, _stream_search_neck);

		CUDA_CHECK(cudaFree(_buffers_search_neck[0]));
		CUDA_CHECK(cudaFree(_buffers_search_neck[1]));

		cudaStreamDestroy(_stream_search_neck);

		//-------------------------------------模板匹配-------------------------------------//

		_head_input_srh1 = _neck_search_output1;

		CUDA_CHECK(cudaMalloc(&_buffers_head[0], _batch_size * 3 * _neckOutput_channel * _neckOutput_h * _neckOutput_w * sizeof(float)));
		CUDA_CHECK(cudaMalloc(&_buffers_head[1], _batch_size * 3 * _neckOutput_channel * 7 * 7 * sizeof(float)));

		CUDA_CHECK(cudaMalloc(&_buffers_head[2], 10 * 25 * 25 * sizeof(float)));
		CUDA_CHECK(cudaMalloc(&_buffers_head[3], 20 * 25 * 25 * sizeof(float)));

		CUDA_CHECK(cudaStreamCreate(&_stream_head));

		//---------------------------------------input---------------------------------------//
		cudaMemcpyAsync(_buffers_head[0], _head_input_srh1, _batch_size * 3 * _neckOutput_channel * _neckOutput_h * _neckOutput_w * sizeof(float), cudaMemcpyHostToDevice, _stream_head);
		cudaMemcpyAsync(_buffers_head[1], _head_input_tep1, _batch_size * 3 * _neckOutput_channel * 7 * 7 * sizeof(float), cudaMemcpyHostToDevice, _stream_head);

		//-------------------------------------inference-------------------------------------//
		_context_head->enqueue(_batch_size, _buffers_head, _stream_head, nullptr);
		
		//--------------------------------------output---------------------------------------//
		cudaMemcpyAsync(_head_output_cls, _buffers_head[2], 10 * 25 * 25 * sizeof(float), cudaMemcpyDeviceToHost, _stream_head);
		cudaMemcpyAsync(_head_output_reg, _buffers_head[3], 20 * 25 * 25 * sizeof(float), cudaMemcpyDeviceToHost, _stream_head);

		for (size_t i = 0; i < 4; i++)
		{
			CUDA_CHECK(cudaFree(_buffers_head[i]));
		}
		cudaStreamDestroy(_stream_head);
	}
}

void simaRPN::_MatToTensor_template_neck(cv::Mat img)
{
	int i = 0;
	int b = 0;
	for (int row = 0; row < _template_input_h; ++row)
	{
		uchar* uc_pixel = img.data + row * img.step;      //pr_img.data为首地址，pr_img.step为一整行的步长
		for (int col = 0; col < _template_input_w; ++col)
		{
			_neck_template_input[b * 3 * _template_input_h * _template_input_w + i] = float(uc_pixel[0]);
			_neck_template_input[b * 3 * _template_input_h * _template_input_w + i + _template_input_h * _template_input_w] = float(uc_pixel[1]);
			_neck_template_input[b * 3 * _template_input_h * _template_input_w + i + 2 * _template_input_h * _template_input_w] = float(uc_pixel[2]);
			uc_pixel += 3;      //指针地址位置+3
			//std::cout << b * 3 * _input_h * _input_w / 4 + i << " == " << float(uc_pixel[2]) / 255.0 << ","
			//	<< float(_neck_template_input[b * 3 * _input_h * _input_w / 4 + i]) << std::endl;
			//std::cout << float(_neck_template_input[b * 3 * _input_h * _input_w / 4 + i]);
			++i;
		}
		//std::cout << " \n\n\n " << std::endl;
	}
};

void simaRPN::_MatToTensor_search_neck(cv::Mat img) 
{
	int i = 0;
	int b = 0;
	for (int row = 0; row < _input_h; ++row)
	{
		uchar* uc_pixel = img.data + row * img.step;      //pr_img.data为首地址，pr_img.step为一整行的步长
		for (int col = 0; col < _input_w; ++col)
		{
			_neck_search_input[b * 3 * _input_h * _input_w + i] = float(uc_pixel[0]);
			_neck_search_input[b * 3 * _input_h * _input_w + i + _input_h * _input_w] = float(uc_pixel[1]);
			_neck_search_input[b * 3 * _input_h * _input_w + i + 2 * _input_h * _input_w] = float(uc_pixel[2]);
			uc_pixel += 3;      //指针地址位置+3
			//std::cout << b << "," << b * 3 * _input_h * _input_w + i + 2 * _input_h * _input_w << " == " << float(uc_pixel[2]) << ","
			//	<< float(_neck_search_input[b * 3 * _input_h * _input_w + i + 2 * _input_h * _input_w]) << std::endl;
			++i;
		}
	}
}

void simaRPN::_TensorToTorch_tensor_cls() 
{
	std::vector<torch::Tensor> cls_result;
	std::vector<torch::Tensor> cls_score;
	std::vector<float> cls_score_col;
	int index = 0;

	for (int i = 0; i < 10; i++)
	{
		cls_score.clear();
		for (int row = 0; row < 25; row++)
		{
			cls_score_col.clear();
			for (int col = 0; col < 25; col++)
			{
				//std::cout << _head_output_cls[index + col * (row + 1)] << std::endl;
				cls_score_col.push_back(_head_output_cls[index]);
				++index;
			}
			cls_score.push_back(torch::tensor(cls_score_col, at::kCUDA));
			//std::cout << cls_score_col << std::endl;
		}
		//std::cout << cls_score << std::endl;
		cls_result.push_back(torch::stack({ cls_score }, -1));
	}
	_class_score = torch::stack({ cls_result }, 0);
	_class_score = torch::unsqueeze(_class_score, 0);
	//std::cout << _class_score << std::endl;

};

void simaRPN::_TensorToTorch_tensor_reg()
{
	std::vector<torch::Tensor> reg_result;
	std::vector<torch::Tensor> reg_score;
	std::vector<float> reg_score_col;

	int index = 0;

	for (int i = 0; i < 20; i++)
	{
		reg_score.clear();
		for (int row = 0; row < 25; row++)
		{
			reg_score_col.clear();
			for (int col = 0; col < 25; col++)
			{
				reg_score_col.push_back(_head_output_reg[index]);
				//std::cout << _head_output_reg[index] << std::endl;
				++index;
			}
			reg_score.push_back(torch::tensor(reg_score_col, at::kCUDA));
			//std::cout << cls_score_col << std::endl;
		}
		//std::cout << cls_score << std::endl;
		reg_result.push_back(torch::stack({ reg_score }, -1));
	}
	_reg_score = torch::stack({ reg_result }, 0);
	_reg_score = torch::unsqueeze(_reg_score, 0);
	//std::cout << _reg_score << std::endl;

};

void simaRPN::_TensorToTorch_mat_tensor_cls()
{
	typedef cv::Vec<float, 10> Vec10f;
	int index = 0;
	cv::Mat output_cls = cv::Mat::zeros(25, 25, CV_32FC(10));;
	for (int i = 0; i < 10; i++)
	{
		for (int row = 0; row < 25; row++)
		{
			for (int col = 0; col < 25; col++)
			{
				output_cls.at<Vec10f>(row, col)[i] = _head_output_cls[index];
				++index;
			}
		}
	}
	_class_score = torch::from_blob(output_cls.data, { 25, 25, 10 }).toType(torch::kFloat32);

	_class_score = _class_score.to(at::kCUDA);
	_class_score = _class_score.permute({ 2, 0, 1 });
	_class_score = torch::unsqueeze(_class_score, 0);
}

void simaRPN::_TensorToTorch_mat_tensor_reg()
{
	typedef cv::Vec<float, 20> Vec20f;
	int index = 0;
	cv::Mat output_reg = cv::Mat::zeros(25, 25, CV_32FC(20));;
	for (int i = 0; i < 20; i++)
	{
		for (int row = 0; row < 25; row++)
		{
			for (int col = 0; col < 25; col++)
			{
				output_reg.at<Vec20f>(row, col)[i] = _head_output_reg[index];
				++index;
			}
		}
	}
	_reg_score = torch::from_blob(output_reg.data, { 25, 25, 20 }).toType(torch::kFloat32);
	_reg_score = _reg_score.to(at::kCUDA);
	_reg_score = _reg_score.permute({ 2, 0, 1 });
	_reg_score = torch::unsqueeze(_reg_score, 0);
}

void simaRPN::_cout_output_Tensor(int frame_id) 
{
	if (frame_id == 0)
	{
		std::vector<float*> neck_template_output = { _neck_template_output1, _neck_template_output2, _neck_template_output3 };
		for (int ii = 0; ii < 3; ii++)
		{
			std::cout << " \n\n\n_neck_template_output" << ii + 1 << "\n\n\n" << std::endl;
			int index = 0;
			for (int i = 0; i < 256; i++)
			{
				for (int row = 0; row < 7; row++)
				{
					for (int col = 0; col < 7; col++)
					{
						std::cout << neck_template_output[ii][index] << " ";
						++index;
					}
					std::cout << "" << std::endl;
				}
				std::cout << "\n\n\n" << std::endl;
			}
		}
	}
	else
	{
		std::vector<float*> neck_search_output = { _neck_search_output1, _neck_search_output2, _neck_search_output3 };
		for (int ii = 0; ii < 3; ii++)
		{
			std::cout << " \n\n\n_neck_search_output" << ii + 1 << "\n\n\n" << std::endl;
			int index = 0;
			for (int i = 0; i < 256; i++)
			{
				for (int row = 0; row < 31; row++)
				{
					for (int col = 0; col < 31; col++)
					{
						std::cout << neck_search_output[ii][index] << " ";
						++index;
					}
					std::cout << "" << std::endl;
				}
				std::cout << "\n\n\n" << std::endl;
			}
		}
	}

}

void simaRPN::_anchor_generator() 
{
	std::vector<std::vector<double>> base = { {-52., -16.,  52., 16.},{-44., -20.,  44., 20.}, {-32., -32.,  32., 32.} ,{-20., -44., 20., 44.},{-16., -52., 16., 52.} };
	torch::Tensor base_anchors1 = torch::tensor(base[0], at::device({ at::kCUDA, 0 }));
	torch::Tensor base_anchors2 = torch::tensor(base[1], at::device({ at::kCUDA, 0 }));
	torch::Tensor base_anchors3 = torch::tensor(base[2], at::device({ at::kCUDA, 0 }));
	torch::Tensor base_anchors4 = torch::tensor(base[3], at::device({ at::kCUDA, 0 }));
	torch::Tensor base_anchors5 = torch::tensor(base[4], at::device({ at::kCUDA, 0 }));

	torch::Tensor base_anchors = torch::stack({ base_anchors1 ,base_anchors2 ,base_anchors3 ,base_anchors4, base_anchors5 }, 0);

	torch::Tensor para = torch::tensor({ 8, 8, 25, 25, 2 });
	torch::Scalar feat_h, feat_w = 25;
	torch::Scalar stride_h = 8, stride_w = 8;

	torch::Tensor shift_x = torch::arange(0, 25 * 8, stride_w, at::kCUDA);    //c10::DeviceType::CUDA      c10::DeviceType(1)    c10::Device::Type::CUDA
	torch::Tensor shift_y = torch::arange(0, 25 * 8, stride_h, at::kCUDA);
	auto shift_xx = shift_x.repeat(shift_y.size(0));
	auto shift_yy = shift_y.view({ -1, 1 }).repeat({ 1, shift_x.size(0) }).view(-1);

	auto shifts = torch::stack({ shift_xx, shift_yy, shift_xx, shift_yy }, -1);
	shifts = shifts.type_as(base_anchors);

	shifts = torch::unsqueeze(shifts, 1);
	base_anchors = torch::unsqueeze(base_anchors, 0);

	_all_anchors = base_anchors + shifts;
	_all_anchors = _all_anchors.view({ -1, 4 });
	_all_anchors.select(1, 0) = _all_anchors.select(1, 0) - (para[2] / para[4]) * para[1] + 2 * para[4];
	_all_anchors.select(1, 2) = _all_anchors.select(1, 2) - (para[2] / para[4]) * para[1] + 2 * para[4];
	_all_anchors.select(1, 1) = _all_anchors.select(1, 1) - (para[3] / para[4]) * para[0] + 2 * para[4];
	_all_anchors.select(1, 3) = _all_anchors.select(1, 3) - (para[3] / para[4]) * para[0] + 2 * para[4];
}

void simaRPN::_windows_generator() 
{
	std::vector<double> hanning_h = { 0., 0.01703709, 0.0669873, 0.14644661, 0.25, 0.37059048, 0.5, 0.62940952 ,0.75 ,0.85355339, 0.9330127,  0.98296291, 1.,0.98296291 ,0.9330127,  0.85355339, 0.75,  0.62940952, 0.5, 0.37059048,0.25,0.14644661,0.0669873, 0.01703709,0 };
	std::vector<double> hanning_w = { 0., 0.01703709, 0.0669873, 0.14644661, 0.25, 0.37059048, 0.5, 0.62940952 ,0.75 ,0.85355339, 0.9330127,  0.98296291, 1.,0.98296291 ,0.9330127,  0.85355339, 0.75,  0.62940952, 0.5, 0.37059048,0.25,0.14644661,0.0669873, 0.01703709,0 };

	torch::Tensor hann_h = torch::tensor(hanning_h, at::kCUDA);
	torch::Tensor hann_w = torch::tensor(hanning_w, at::kCUDA);

	std::vector<torch::Tensor> stack_tensor;
	for (size_t i = 0; i < hanning_w.size(); i++)
	{
		stack_tensor.push_back(hann_h * hann_w[i]);
	}
	_windows = torch::cat({ stack_tensor }, -1).repeat(5);
}

torch::Tensor change_ratio(torch::Tensor ratio) 
{
	return torch::max(ratio, 1.0f / ratio);
};

torch::Tensor enlarge_size(torch::Tensor w, torch::Tensor h) 
{
	auto pad = (w + h) * 0.5f;	
	return torch::sqrt((w + pad) * (h + pad));
};

torch::Tensor simaRPN::_decode_bbox(double scale_factor) 
{
	torch::Tensor means = torch::tensor({ 0., 0., 0., 0. }, at::kCUDA);
	means = torch::unsqueeze(means, 0);
	torch::Tensor stds = torch::tensor({ 1., 1., 1., 1. }, at::kCUDA);
	stds = torch::unsqueeze(stds, 0);

	auto denorm_reg_score = _reg_score * stds + means;

	auto dx = denorm_reg_score.select(1, 0);
	auto dy = denorm_reg_score.select(1, 1);
	auto dw = denorm_reg_score.select(1, 2);
	auto dh = denorm_reg_score.select(1, 3);

	dx = torch::unsqueeze(dx, 1);
	dy = torch::unsqueeze(dy, 1);
	dw = torch::unsqueeze(dw, 1);
	dh = torch::unsqueeze(dh, 1);

	auto x1 = _all_anchors.select(1, 0);
	auto y1 = _all_anchors.select(1, 1);
	auto x2 = _all_anchors.select(1, 2);
	auto y2 = _all_anchors.select(1, 3);
	
	auto px = ((x1 + x2) * 0.5).unsqueeze(-1).expand_as(dx);
	auto py = ((y1 + y2) * 0.5).unsqueeze(-1).expand_as(dy);
	auto pw = (x2 - x1).unsqueeze(-1).expand_as(dw);
	auto ph = (y2 - y1).unsqueeze(-1).expand_as(dh);

	auto dx_width = pw * dx;
	auto dy_height = ph * dy;

	double max_ratio = abs(log(0.016));
	
	dw = dw.clamp(-max_ratio, max_ratio);
	dh = dh.clamp(-max_ratio, max_ratio);

	auto gw = pw * dw.exp();
	auto gh = ph * dh.exp();
	auto gx = px + dx_width;
	auto gy = py + dy_height;

	x1 = gx - gw * 0.5;
	y1 = gy - gh * 0.5;
	x2 = gx + gw * 0.5;
	y2 = gy + gh * 0.5;

	auto bboxs_pred = torch::stack({x1, y1, x2, y2}, -1).view(_reg_score.sizes());   //还未转成x,y,w,h

	return bboxs_pred;
}

auto simaRPN::_bbox_xyxy_to_xywh(torch::Tensor bbox) 
{
	std::vector<torch::Tensor> bbox_xyxy = bbox.split(( 1, 1, 1, 1 ), -1);
	auto new_bbox = torch::cat({ (bbox_xyxy[0] + bbox_xyxy[2]) / 2, (bbox_xyxy[1] + bbox_xyxy[3]) / 2 ,bbox_xyxy[2] - bbox_xyxy[0], bbox_xyxy[3] - bbox_xyxy[1] }, -1);
	return new_bbox;
}

void simaRPN::_get_box(int frame_id, double scale_factor)
{
	if (frame_id == 1)
	{
		_anchor_generator();
		_windows_generator();
	}
	int64_t H = _class_score.size(2);
	int64_t W = _class_score.size(3);

	_class_score = _class_score.view({ 2, -1, H, W });
	_class_score = _class_score.permute({ 2, 3, 1, 0 }).contiguous().view({ -1, 2 });

	_class_score = _class_score.softmax({1});
	_class_score = _class_score.select(1, 1);

	_reg_score = _reg_score.view({ 4, -1, H, W });
	_reg_score = _reg_score.permute({ 2, 3, 1, 0 }).contiguous().view({ -1, 4 });

	torch::Tensor bboxs_pred = _decode_bbox(scale_factor);
	bboxs_pred = _bbox_xyxy_to_xywh(bboxs_pred);
	
	auto scale_penalty = change_ratio(
		enlarge_size(bboxs_pred.select(1, 2), bboxs_pred.select(1, 3)) / enlarge_size(                                                    //  ***※※※※***
			torch::tensor(_result_bbox.width, at::kCUDA) * scale_factor, torch::tensor(_result_bbox.height, at::kCUDA) * scale_factor));  // scale_factor

	auto aspect_ratio_penalty = change_ratio(   // 偏差较大
		torch::tensor(float(_result_bbox.width) / float(_result_bbox.height), at::kCUDA) /
		(bboxs_pred.select(1, 2) / bboxs_pred.select(1, 3)));

	auto penalty = torch::exp(-(aspect_ratio_penalty * scale_penalty - 1) * 0.05f);

	auto penalty_score = penalty * _class_score;

	penalty_score = penalty_score * (1 - 0.42f) + _windows * 0.42f;

	auto best_index = torch::argmax(penalty_score);

	_best_score = _class_score[best_index];
	//std::cout << _best_score.item().toFloat() << std::endl;
	auto best_bbox = bboxs_pred[best_index] / scale_factor;

	_final_bbox = torch::zeros_like(best_bbox);
	_final_bbox[0] = best_bbox[0] + torch::tensor(_result_bbox.x, at::kCUDA);
	_final_bbox[1] = best_bbox[1] + torch::tensor(_result_bbox.y, at::kCUDA);

	auto lr = penalty[best_index] * _class_score[best_index] * 0.38f;

	_final_bbox[2] = best_bbox[2] * lr + torch::tensor(_result_bbox.width, at::kCUDA) * (1 - lr);
	_final_bbox[3] = best_bbox[3] * lr + torch::tensor(_result_bbox.height, at::kCUDA) * (1 - lr);

	_final_bbox_matric[frame_id] = _final_bbox;

	float x = _final_bbox[0].item().toFloat();
	float y = _final_bbox[1].item().toFloat();
	float x_pre = _final_bbox_matric[frame_id - 1][0].item().toFloat();
	float y_pre = _final_bbox_matric[frame_id - 1][1].item().toFloat();
	float x_pre_pre;
	float y_pre_pre;
	if (frame_id >= 2)
	{
		x_pre_pre = _final_bbox_matric[frame_id - 2][0].item().toFloat();
		y_pre_pre = _final_bbox_matric[frame_id - 2][1].item().toFloat();
	}

	//if (abs(x - x_pre) > 2 * abs(x_pre - x_pre_pre) || abs(y - y_pre) > 2 * abs(y_pre - y_pre_pre))
	//{
	//	_kalman_filter(frame_id);
	//}

	//if (_best_score.item().toFloat() <= _kalman_score && _kalman_open)
	//{
	//	_kalman_filter(frame_id);
	//}
}

void simaRPN::_bbox_clip(int img_h, int img_w) 
{
	_final_bbox[0] = _final_bbox[0].clamp(0., img_w);
	_final_bbox[1] = _final_bbox[1].clamp(0., img_h);
	_final_bbox[2] = _final_bbox[2].clamp(10., img_w);
	_final_bbox[3] = _final_bbox[3].clamp(10., img_h);
}

void simaRPN::inference() 
{
	std::vector<float> time_list;
	std::vector<float> time_inf_list;

	cv::VideoCapture video(0);   //
	if (video.isOpened())
	{
		std::cout << "---------------informations--------------" << std::endl;
		std::cout << "width =        " << video.get(cv::CAP_PROP_FRAME_WIDTH) << std::endl;
		std::cout << "height =       " << video.get(cv::CAP_PROP_FRAME_HEIGHT) << std::endl;
		std::cout << "frame_nums =   " << video.get(cv::CAP_PROP_FPS) << std::endl;
		std::cout << "total_frames = " << video.get(cv::CAP_PROP_FRAME_COUNT) << std::endl;
		std::cout << "read the video ,please wait ..." << std::endl;
	}
	if (_save_video)
	{
		cv::VideoWriter writer;
		std::string save_path = video_path.replace(video_path.size() - 8, 8 ,"result.avi");
		int coder = cv::VideoWriter::fourcc('X', 'V', 'I', 'D');
		cv::Size size = { int(video.get(cv::CAP_PROP_FRAME_WIDTH)), int(video.get(cv::CAP_PROP_FRAME_HEIGHT)) };
		double fps = video.get(cv::CAP_PROP_FPS);
		writer.open(save_path, coder, fps, size);   //save_path, coder, fps, size
	}
	int frame_id = 0;
	cv::Mat frame;
	char string[20];
	char string1[20];
	float best_score;
	while (video.isOpened())
	{
		auto start = std::chrono::system_clock::now();
		video >> frame;

		if (frame.empty())
		{
			std::cout << "input image is NULL ..." << std::endl;
			break;
		}

		if (frame_id == 0)
		{
			_result_bbox = cv::selectROI(frame, false, false);
			//_result_bbox = cv::Rect(cv::Point2d{ 390,412 }, cv::Point2d{ 430, 441 });
			//_result_bbox = cv::Rect(cv::Point2d{ 1120, 210 }, cv::Point2d{ 1153, 242 });
			//_result_bbox = cv::Rect(cv::Point2d{ 1044, 245 }, cv::Point2d{ 1054, 255 });
			cv::Mat crop_frame = _get_crop_frame(_result_bbox, frame, frame_id);
			_MatToTensor_template_neck(crop_frame);
			_image_inference(frame_id);
			//_cout_output_Tensor(frame_id);
			best_score = -1;
		}
		else
		{
			cv::Mat crop_frame = _get_crop_frame(_result_bbox, frame, frame_id);  // 0.0009s
			_MatToTensor_search_neck(crop_frame);

			_inf_start = std::chrono::system_clock::now();
			_image_inference(frame_id);
			_inf_end = std::chrono::system_clock::now();
			//_cout_output_Tensor(frame_id);
			//_test_uniform();

			_TensorToTorch_mat_tensor_cls();    // →0.0008s
			_TensorToTorch_mat_tensor_reg();    //   ↑
			float scale_factor = 127.0f / _z_size;
			_get_box(frame_id, scale_factor);    // 0.000174937s
			_bbox_clip(frame.rows, frame.cols);   // 0.0s
			_final_bbox = torch::round(_final_bbox);
			//std::cout << std::setprecision(20) << _final_bbox << std::endl;
			best_score = _best_score.item().toFloat();

		}
		if (best_score == -1)
		{
			torch::Tensor one = torch::tensor({ (float)_result_bbox.x, (float)_result_bbox.y, (float)_result_bbox.width, (float)_result_bbox.height }, at::kCUDA);
			_final_bbox_matric[frame_id] = one;
			_result_bbox = _result_bbox;
		}
		else
		{
			if (best_score <= _kalman_score && best_score >= 0 && _kalman_open)
			{
			}
			else
			{
				_result_bbox.x = _final_bbox[0].item().toFloat();
				_result_bbox.y = _final_bbox[1].item().toFloat();
				_result_bbox.width = _final_bbox[2].item().toFloat();
				_result_bbox.height = _final_bbox[3].item().toFloat();
			}
		}
		frame_id++;

		cv::rectangle(frame, _result_bbox, cv::Scalar(0, 255, 0), 2);
		//auto s = std::chrono::system_clock::now();
		//cv::Point pt[1][7];
		//int x = _result_bbox.x;
		//int y = _result_bbox.y;
		//int width = _result_bbox.width;
		//pt[0][0] = cv::Point(x + width / 2, _result_bbox.y - 10);
		//pt[0][1] = cv::Point(x + width / 2 - 20, y - 10 - sqrt(3) * 20);
		//pt[0][2] = cv::Point(x + width / 2 - 10, y - 10 - sqrt(3) * 20);
		//pt[0][3] = cv::Point(x + width / 2 - 10, y - 20 - sqrt(3) * 20);
		//pt[0][4] = cv::Point(x + width / 2 + 10, y - 20 - sqrt(3) * 20);
		//pt[0][5] = cv::Point(x + width / 2 + 10, y - 10 - sqrt(3) * 20);
		//pt[0][6] = cv::Point(x + width / 2 + 20, y - 10 - sqrt(3) * 20);
		//const cv::Point* ppt[1] = { pt[0] };
		//int npt[] = { 7 };
		//cv::Mat dst;
		//frame.copyTo(dst);
		//cv::fillPoly(dst, ppt, npt, 1, cv::Scalar(0, 255, 0));
		//cv::addWeighted(dst, 0.5, frame, 0.5, 0, dst);
		//auto e = std::chrono::system_clock::now();
		//std::cout << (std::chrono::duration_cast<std::chrono::milliseconds>(e - s).count()) / 1000.0 << "s" << std::endl;

		std::vector<std::vector<cv::Point>> contours;
		std::vector<cv::Point> contour;

		int x = _result_bbox.x;
		int y = _result_bbox.y;
		int width = _result_bbox.width;
		contour.push_back(cv::Point(x + width / 2, _result_bbox.y - 10));
		contour.push_back(cv::Point(x + width / 2 - 20, y - 10 - sqrt(3) * 20));
		contour.push_back(cv::Point(x + width / 2 - 10, y - 10 - sqrt(3) * 20));
		contour.push_back(cv::Point(x + width / 2 - 10, y - 20 - sqrt(3) * 20));
		contour.push_back(cv::Point(x + width / 2 + 10, y - 20 - sqrt(3) * 20));
		contour.push_back(cv::Point(x + width / 2 + 10, y - 10 - sqrt(3) * 20));
		contour.push_back(cv::Point(x + width / 2 + 20, y - 10 - sqrt(3) * 20));
		contours.push_back(contour);

		std::vector<cv::Mat> mvs;

		cv::split(frame, mvs);

		for (int c = 0; c < contours.size() && 3 == mvs.size(); c++)
		{
			cv::drawContours(mvs[1], contours, c, cv::Scalar(255, 0, 255), -1);
			cv::drawContours(mvs[0], contours, c, cv::Scalar(255, 0, 0), 2);
			cv::drawContours(mvs[1], contours, c, cv::Scalar(0, 255, 0), 2);
			cv::drawContours(mvs[2], contours, c, cv::Scalar(0, 0, 255), 2);
		}
		cv::merge(mvs, frame);

		auto end = std::chrono::system_clock::now();

		std::cout << "frame:" << "\t" << frame_id << "\t" << "|  total >> " <<
			(std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()) / 1000.0 << " s " << " \t" <<
			"|  inf >> " << (std::chrono::duration_cast<std::chrono::milliseconds>(_inf_end - _inf_start)).count() / 1000.0 << " s " << std::endl;
		time_list.push_back((std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()) / 1000.0);
		time_inf_list.push_back((std::chrono::duration_cast<std::chrono::milliseconds>(_inf_end - _inf_start).count()) / 1000.0);

		double fps = 1.0 / ((std::chrono::duration_cast<std::chrono::milliseconds>(end - start)).count() / 1000.0);
		sprintf(string, "%.2f", fps);
		sprintf(string1, "%.2f", best_score);

		std::string fpsString = "FPS:";
		std::string scoreString = "";
		fpsString += string;
		scoreString += string1;
		putText(frame, fpsString, cv::Point(30, 80), 5, 4, cv::Scalar(255, 0, 0));
		if (frame_id >= 1) {
			putText(frame, scoreString, cv::Point(_result_bbox.x, _result_bbox.y - 70), 5, 2, cv::Scalar(0, 0, 255), 2);
		}
		cv::namedWindow("Tracking target", cv::WINDOW_AUTOSIZE);
		cv::imshow("Tracking target", frame);
		cv::waitKey(1);
	}
	double sum1 = std::accumulate(std::begin(time_list), std::end(time_list), 0.0);
	double mean1 = sum1 / time_list.size(); 
	double sum2 = std::accumulate(std::begin(time_inf_list), std::end(time_inf_list), 0.0);
	double mean2 = sum2 / time_inf_list.size(); 
	std::cout << "total time: " << "\t" << sum1 << std::endl;
	std::cout << "average time: " << "\t" << mean1 << std::endl;
	std::cout << "inferenve total time: " << "\t" << sum2 << std::endl;
	std::cout << "inferenve average time: " << "\t" << mean2 << std::endl;
	std::cout << "average FPS: " << "\t" << 1.0 / mean1 << std::endl;

};	

