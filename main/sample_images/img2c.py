import cv2
import numpy as np
# 将图片转换为c语言数组

width = 128
height = 128

img = cv2.imread("tiger.jpeg")
img = cv2.resize(img, (width, height))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# img = (img.copy().astype("int32")-128).astype("int8")


img_bytes = []

for x in range(width):
    for y in range(height):
        img_bytes.append(int(img[x, y][0]))
        img_bytes.append(int(img[x, y][1]))
        img_bytes.append(int(img[x, y][2]))

s = "unsigned char img[]={"
s = s + ",".join([hex(x) for x in img_bytes])
s = s + "};"

f = open("../img.c", "w")
f.write(s)
f.close()
