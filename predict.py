'''
predict.py有几个注意点
1、无法进行批量预测，如果想要批量预测，可以利用os.listdir()遍历文件夹，利用Image.open打开图片文件进行预测。
2、如果想要保存，利用r_image.save("img.jpg")即可保存。
3、如果想要获得框的坐标，可以进入detect_image函数，读取top,left,bottom,right这四个值。
4、如果想要截取下目标，可以利用获取到的top,left,bottom,right这四个值在原图上利用矩阵的方式进行截取。
'''
from PIL import Image
import cv2
import numpy as np

from yolo import YOLO

import os, sys

os.chdir(sys.path[0])

yolo = YOLO()

while True:
    img = input('Input image filename:')
    try:
        image = Image.open(img)      
    except:
        print('Open Error! Try again!')
        continue
    else:
        r_image = yolo.detect_image(image)
        r_image.save("gaiⅣ.jpg")
        r_image.show()

# def test_txt():
#     sets=[('2021', 'test')]
#     srcimage = "img_test/jpg/"
#     path = "img_test/img_test.txt"
#     for year, image_set in sets:
#         image_ids = open(path)
#         list_file = open('img_test/%s_%s.txt'%(year, image_set), 'w')
#         for image_id in image_ids:
#             image_id = str(int(image_id)).zfill(5)
#             list_file.write(srcimage+'%s.jpg'%(image_id))
#             list_file.write('\n')
#         list_file.close()

# def predict():
#     if not os.path.exists("img_test/predict"):
#         os.makedirs("img_test/predict")
#     with open("img_test/2021_test.txt") as f:
#         lines = f.readlines()
#         for line in lines:
#             suffix = line.split('.')[-2].split('/')[-1]
#             line = line.split()
#             #print(line[0])
#             img = Image.open(line[0]).convert("L").convert("RGB")
#             r_image = yolo.detect_image(img)
#             #r_image.show()
#             r_image.save("img_test/predict/"+suffix+'.jpg')

# #输出图片路径
# test_txt()
# #输出预测结果
# predict()