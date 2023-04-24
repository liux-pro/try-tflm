# 把tflite文件转换为头文件
f_read = open("mobilenet_v1_0.25_128.tflite", "rb")
tflite_bytes = f_read.read()

templet = "#pragma once\nalignas(8) const unsigned char model_tflite[] = {{{}}};"

s = templet.format(",".join([hex(x) for x in tflite_bytes]))

f = open("model_tflite.h", "w")
f.write(s)
f.close()
