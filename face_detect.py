#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 18:19:00 2019

@author: kevin
"""
from PIL import Image 
import cv2
import os

'''filename = '/Users/kevin/Desktop/photo/Alisson Becker/Alisson Becker face -pes -fifa6.jpg'
img = cv2.imread(filename)
shape = img.shape
print(img.ndim)
face_cascade = cv2.CascadeClassifier('/anaconda3/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml')
if img.ndim == 3:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
else:
    gray = img #if语句：如果img维度为3，说明不是灰度图，先转化为灰度图gray，如果不为3，也就是2，原图就是灰度图

faces = face_cascade.detectMultiScale(gray, 1.3, 5)#1.3和5是特征的最小、最大检测窗口，它改变检测结果也会改变
result = []
for (x,y,width,height) in faces:
    result.append((max(0,x-0.3*width),max(0,y-0.3*height),min(shape[1],x+1.3*width),min(shape[0],y+1.3*height)))
print(result)
try:
    Image.open(filename).crop(result[0]).save('/Users/kevin/Desktop/1.jpg')
except:
    os.remove('/Users/kevin/Desktop/1.jpg')
    Image.open(filename).crop(result[0]).save('/Users/kevin/Desktop/1.png')'''


'''def detectFaces(image_name):
    img = cv2.imread(image_name)
    face_cascade = cv2.CascadeClassifier('/anaconda3/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml')
    if (img.ndim == 3 | img.ndim == 4):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img #if语句：如果img维度为3，说明不是灰度图，先转化为灰度图gray，如果不为3，也就是2，原图就是灰度图

    faces = face_cascade.detectMultiScale(gray, 1.1, 8)#1.3和5是特征的最小、最大检测窗口，它改变检测结果也会改变
    result = []
    for (x,y,width,height) in faces:
        result.append((x,y,x+width,y+height))
    return result


def saveFaces(input_img_path,output_img_path):
    for player in os.listdir(input_img_path):
        os.makedirs(output_img_path+'/'+player)
        for pic in os.listdir(input_img_path+'/'+player):
            try:
                faces = detectFaces(input_img_path+'/'+player+'/'+pic)
                if faces:
                    #将人脸保存在save_dir目录下。
                    #Image模块：Image.open获取图像句柄，crop剪切图像(剪切的区域就是detectFaces返回的坐标)，save保存。
                    count = 0
                    for (x1,y1,x2,y2) in faces:
                        file_name = os.path.join(output_img_path+'/'+player,player+str(count)+".jpg")
                        Image.open(input_img_path+'/'+player+'/'+pic).crop((x1,y1,x2,y2)).save(file_name)
                        count+=1
            except:
                pass'''

def saveFaces(input_img_path,output_img_path):
    i = 1
    for player in os.listdir(input_img_path):
        os.makedirs(output_img_path+'/'+player)
        if (player=='.DS_Store'):
            pass
        for pic in os.listdir(input_img_path+'/'+player):
            try:
                img = cv2.imread(input_img_path+'/'+player+'/'+pic)
                shape = img.shape
                face_cascade = cv2.CascadeClassifier('/anaconda3/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml')
                if img.ndim == 3:
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                else:
                    gray = img #if语句：如果img维度为3，说明不是灰度图，先转化为灰度图gray，如果不为3，也就是2，原图就是灰度图
                
                faces = face_cascade.detectMultiScale(gray, 1.15, 5)#1.3和5是特征的最小、最大检测窗口，它改变检测结果也会改变
                result = []
                for (x,y,width,height) in faces:
                    w = max(width,height)
                    if ((1.6*width>127) & (1.6*height>127)):
                        result.append((max(0,x-0.3*w),max(0,y-0.3*w),min(shape[1],x+1.3*w),min(shape[0],y+1.3*w)))
                print(result)
                try:
                    Image.open(input_img_path+'/'+player+'/'+pic).crop(result[0]).resize((128,128)).save(output_img_path+'/'+player+'/'+player+str(i)+'.jpg')
                except:
                    os.remove(output_img_path+'/'+player+'/'+player+str(i)+'.jpg')
                    img = Image.open(input_img_path+'/'+player+'/'+pic)
                    r, g, b, a = img.split()
                    img = Image.merge("RGB", (r, g, b))
                    img.convert('RGB').crop(result[0]).resize((128,128)).save(output_img_path+'/'+player+'/'+player+str(i)+'.jpg')
                    #Image.open(input_img_path+'/'+player+'/'+pic).crop(result[0]).save(output_img_path+'/'+player+'/'+player+str(i)+'.png')
                i += 1
            except:
                pass

saveFaces('/Users/kevin/Desktop/photo','/Users/kevin/Desktop/photoface2')

#saveFaces('/Users/kevin/Desktop/photo2','/Users/kevin/Desktop/photo3')









