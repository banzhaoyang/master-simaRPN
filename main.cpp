#include "simaRPN++.h"



int main()
{
	//simaRPN sima;
	//sima.test_engine();
	//sima.test_load_engine();
	//sima.test_inference();
	//sima.test_backbone_inference();

	simaRPN sima;
	sima.video_path = "0";
	sima.wts_name = "F:/visualObject/master-simaRPN++/wts/siamese_rpn_r50_20e_lasot_20220420_181845-dd0f151e.wts";
	sima.gen_template_neck_engine();
	sima.gen_search_neck_engine();
	sima.gen_head_engine();
	std::cout << " gen engine successful!\n";
	//sima.video_path = "D:/vspy_obj/simaRPN++-master/simaRPN++/demo/non_logo_1920-1080.avi";
	//sima.video_path = "D:/vspy_obj/yolov7/yolo_deepstream/tensorrt_yolov7/imgs/test.mp4";

	sima.load_template_neck_engine();
	sima.load_search_neck_engine();
	sima.load_head_engine();
	sima.inference();

	system("pause");
	return 0;
}

