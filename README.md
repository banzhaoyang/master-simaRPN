# simaRPN++

The Pytorch implementation is [open-mmlab/mmtracking](https://github.com/open-mmlab/mmtracking).
## environment

### win10  vs2017 cuda11.4 cudnn7.6.5 tensorrt8.5.1.7 libtorch1.9.0 opencv4.5.0 eigen 3.3.9

- For simaRPN++ , download .pth from [https://download.openmmlab.com/mmtracking/sot/siamese_rpn/siamese_rpn_r50_1x_lasot/siamese_rpn_r50_20e_lasot_20220420_181845-dd0f151e.pth]

## How to Run, simaRPN++ as example

1. Using gen_tws.py convert .pth to .wts

```
// download https://download.openmmlab.com/mmtracking/sot/siamese_rpn/siamese_rpn_r50_1x_lasot/siamese_rpn_r50_20e_lasot_20220420_181845-dd0f151e.pth
cp {master-simaRPN++}
python gen_wts.py
// a file 'simaRPN++.wts' will be generated.
```


2. cmake
```
mkdir build && cd build
cmake .. -DCMAKE_GENERATOR_PLATFORM=x64
双击击master-simarpn++.sln
右键项目->调试->环境
PATH=D:\libtorch_release\lib;%PATH% $(LocalDebuggerEnvironment)
链接器->命令行
/INCLUDE:"?ignore_this_library_placeholder@@YAHXZ" 
```

3. 推理效果见 [https://www.bilibili.com/video/BV1GL4y1z7SU/?spm_id_from=333.999.0.0&vd_source=1b511fc030684d71384798f0c288619e]