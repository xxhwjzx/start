# -- encoding:utf-8 --

# from imutils import paths
# imagepaths=list(paths.list_images('./datasets/animals'))
# print(imagepaths)

# import numpy as np
# import matplotlib.pyplot as plt
# np.random.seed(19680801)
#
#
# N = 100
# r0 = 0.6
# x = 0.9 * np.random.rand(N)
# y = 0.9 * np.random.rand(N)
# area = (20 * np.random.rand(N))**2  # 0 to 10 point radii
# print(area)
# c = np.sqrt(area)
# print(c)
# r = np.sqrt(x * x + y * y)
# area1 = np.ma.masked_where(r < r0, area)
# area2 = np.ma.masked_where(r >= r0, area)
# plt.scatter(x, y, s=area1, marker='^', c=c)
# plt.scatter(x, y, s=area2, marker='o', c=c)
# # Show the boundary between the regions:
# theta = np.arange(0, np.pi / 2, 0.01)
# plt.plot(r0 * np.cos(theta), r0 * np.sin(theta))
#
# plt.show()

# from imutils import paths
# image=list(paths.list_images('./datasets/animals'))
# print(image)

# from tensorflow.examples.tutorials.mnist import input_data
# import tensorflow as tf
# file='./datasets/MNIST'
# mnist=input_data.read_data_sets(file,one_hot=True)
# trainx=mnist.train.images
# testx=mnist.test.images
# trainy=mnist.train.labels
# testy=mnist.test.labels
# # print(trainx.shape)
# trainx=trainx.reshape((trainx.shape[0],28,28,1))
# testx=testx.reshape((testx.shape[0],28,28,1))
# print(trainy)
# print(trainy)





import os
from imutils import paths
import cv2


# figname=os.path.altsep.join(['./output','1.jpg'])
# print(figname)
#
# listname=list(paths.list_images('./datasets'))
# print(listname)
# print(os.path.altsep.join(['./output','{}.png'.format(os.getpid())]))

# fig1=os.path.altsep.join(['./output','{}.png'.format(os.getpid())])
# fig2=cv2.imread('./output/3animals.png')
# cv2.imwrite(fig1,fig2)
# cv2.imshow('image',cv2.imread(fig1))
# cv2.waitKey(0)

import argparse
import os

# ap=argparse.ArgumentParser()
# ap.add_argument('-o','--output',required=True,
#                 help='path to the output directory')
# args=vars(ap.parse_args())
# # print(args['output'])
# figpath=os.path.altsep.join([args['output'],'1.png'])
# print(figpath)


# -- encoding:utf-8 --
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import os
import cv2
# for imagePath in sorted(list(paths.list_images('./datasets/SMILEsmileD/SMILEs/positives/positives7'))):
#     image=cv2.imread(imagePath)
#     # print(image)
#     gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
#     # print(gray)
#     gray=cv2.resize(gray,(28,28))
#     gray=gray.astype('float')/255.0
#     # print(gray)
#     gray=img_to_array(gray)
#     # print(gray)
#     gray=np.expand_dims(gray,axis=0)
#
#     # gray=imutils.resize(gray,width=28)
#     model=load_model('./output/smilev1.hdf5')
#     (notsmiling,smiling)=model.predict(gray)[0]
#     print(smiling)



# ap=argparse.ArgumentParser()
# ap.add_argument('-c','--cascade',required=True,
#                 help='path to where the face cascade resides')
# ap.add_argument('-m','--model',required=True,
#                 help='path to pre-trained smile detector CNN')
# ap.add_argument('-v','--video',help='path to the (optional) video file ')
# args=vars(ap.parse_args())
#
# detector=cv2.CascadeClassifier(args['cascade'])
# model=load_model(args['model'])
# if not args.get('video',False):
#     camera=cv2.VideoCapture(0)
#
# else:
#     camera=cv2.VideoCapture(args['video'])
#
# while True:
#     (grabbed,frame)=camera.read()
#     if args.get('videoo') and not grabbed:
#         break
#
#     frame=imutils.resize(frame,height=300)
#     (h,w)=frame.shape[:2]
#     center=(w//2,h//2)
#     M=cv2.getRotationMatrix2D(center,-90,1)
#     frame=cv2.warpAffine(frame,M,(w,h))
#     gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
#     frameClone=frame.copy()
#     rects=detector.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(30,30),flags=cv2.CASCADE_SCALE_IMAGE)
#     for (fx,fy,fw,fh) in rects:
#         roi=gray[fy:fy+fh,fx:fx+fw]
#         roi=cv2.resize(roi,(28,28))
#         roi=roi.astype('float')/255.0
#         roi=img_to_array(roi)
#         roi=np.expand_dims(roi,axis=0)#################这里为什么要扩一维度，试验一下
#         (notsmiling,smiling)=model.predict(roi)[0]
#         label='Smiling' if smiling>notsmiling else 'Not Smiling'
#         cv2.putText(frameClone,label,(fx,fy-10),cv2.FONT_HERSHEY_SIMPLEX,0.45,(0,0,255),2)
#         cv2.rectangle(frameClone,(fx,fy),(fx+fw,fy+fh),(0,0,255),2)
#     cv2.imshow('face',frameClone)
#     if cv2.waitKey(1) & 0xFF ==ord('q'):
#         break
#
# camera.release()
# cv2.destroyAllWindows()


# frame = cv2.imread('./datasets/chepai/3.jpg')
# # print(frame.shape)
# print(frame.shape)
# cv2.imshow('image',frame)
# cv2.waitKey(0)

# (h, w) = frame.shape[:2]  # 获取高h 宽w
# padh=int((w-h)/2.0)
# padw=int((h-w)/2.0)
#
# if h>w:
#     frame=cv2.copyMakeBorder(frame,0,0,padw,padw,cv2.BORDER_REPLICATE)
# else:
#     frame=cv2.copyMakeBorder(frame,padh,padh,0,0,cv2.BORDER_REPLICATE)
# (h, w) = frame.shape[:2]  # 获取高h 宽w
# center = (w // 2, h // 2)  # 获取图像中心
# M = cv2.getRotationMatrix2D(center,0, 1)  # 实例化cv2旋转函数，第一个参数是中心，第二个参数是角度，第三个参数是缩放大小
# frame = cv2.warpAffine(frame, M, (w, h),borderValue=1000)  # 进行旋转 并将旋转后的图片赋值给frame
# cv2.imshow('image',frame)
# cv2.waitKey(0)

# E:\AI\deep_learning_for_computer_vision\start>python extract_features.py --dataset ./datasets/animals/ --output ./datasets/animals/features.hdf5



# imagePaths=list(paths.list_images('./datasets/animals/'))
# labels=[p.split(os.path.sep)[-2].split(os.path.altsep)[-1] for p in imagePaths]
# print(labels)
from sklearn.preprocessing import LabelEncoder
import random
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
imagePaths = list(paths.list_images('./datasets/animals/'))

# random.shuffle(imagePaths)
print(imagePaths)
image = load_img(imagePaths[0], target_size=(224, 224))
print(image)
image = img_to_array(image)
print(image)
print(image.shape)
image = np.expand_dims(image, axis=0)
print(image)
print(image.shape)
image = imagenet_utils.preprocess_input(image)
print(image)
# labels = [p.split(os.path.sep)[-2].split(os.path.altsep)[-1] for p in imagePaths]
#
# print(labels)
# le = LabelEncoder()
# labels = le.fit_transform(labels)
# print(le.classes_)