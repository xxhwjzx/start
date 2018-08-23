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

ap=argparse.ArgumentParser()
ap.add_argument('-o','--output',required=True,
                help='path to the output directory')
args=vars(ap.parse_args())
# print(args['output'])
figpath=os.path.altsep.join([args['output'],'1.png'])
print(figpath)

