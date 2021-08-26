import cv2
import numpy as np
from PIL import Image
from os import getcwd
import os
import matplotlib.pyplot as plt

import random 
random.seed(0)

import sys

os.chdir(sys.path[0])

# def rename(path, num):
#     file_names = os.listdir(path)
#     c = num

#     # 随机获取一张图片的格式
#     # f_first = file_names[0]
#     # suffix = f_first.split('.')[-1]  # 图片文件的后缀

#     for file in file_names:
#         os.rename(os.path.join(path, file), os.path.join(path, '{:0>5d}.{}'.format(c, 'jpg')))
#         c += 1

def test_id(srcimage, saveBasePath):
    train_percent=1

    temp_img = os.listdir(srcimage)
    total_img = []
    for i in temp_img:
        if i.endswith(".jpg"):  # 根据srcimage图片名修改  eg. jpg png and so on.
            total_img.append(i)

    num=len(total_img)  
    list=range(num)  
    tr=int(num*train_percent)  
    train=random.sample(list,tr)  

    ftrain = open(os.path.join(saveBasePath,'img_test.txt'), 'w') 

    for i  in list:  
        name=total_img[i][:-4]+'\n'  
        if i in train:  
            ftrain.write(name)  

    ftrain.close()

srcimage = "img_test/jpg"
#rename(srcimage, 0)
saveBasePath = "img_test"
test_id(srcimage, saveBasePath)