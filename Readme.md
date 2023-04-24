# try-tflm
演示如把Tensorflow的mobilenet模型，转化为tflite模型，并运行在esp32s3上。
# esp32s3
esp32s3的cpu集成了若干加速指令，用以提高ai推理或傅里叶变换，点积，矩阵，向量等的计算速度。  
这部分指令本来需要通过汇编调用，但乐鑫通过
[esp-dsp](https://github.com/espressif/esp-dsp)，
[esp-nn](https://github.com/espressif/esp-nn),
提供了c语言加速接口，使得调用更加方便。
# TFLM 和 esp-nn
TFLM（TensorFlow Lite Micro）是谷歌为嵌入式设备开发的神经网络推理引擎，用TensorFlow开发的模型能够转化为TFLite格式，
而TFLM能在单片机上运行TFLite格式的模型，但需注意，TFLM并非完整支持TFLite，如果模型包含了不支持的算子，TFLM无法运行。    
TFLM全部由标准的c++实现，有良好的移植性。乐鑫通过esp-nn深度适配了TFLM，使得TFLM能够利用s3的加速指令更快的进行运算，以获得数倍的性能优势。

# 项目结构
本项目参考乐鑫[官方tflm示例](https://github.com/espressif/tflite-micro-esp-examples)进行构建，通过组件管理器引用该项目内的esp-nn和tflm源码。  
# 模型转换
Tensorflow提供了mobilenet在imagenet数据集上的预训练模型。简单的说，这个预训练模型学会了能够分辨1000种物体的图片。
我们将其转化为tflite格式，转换方式见 https://www.kaggle.com/code/legendtech/mobilenet-convert 。