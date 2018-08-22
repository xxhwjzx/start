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

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
file='./datasets/MNIST'
mnist=input_data.read_data_sets(file,one_hot=True)
trainx=mnist.train.images
testx=mnist.test.images
trainy=mnist.train.labels
testy=mnist.test.labels
# print(trainx.shape)
trainx=trainx.reshape((trainx.shape[0],28,28,1))
testx=testx.reshape((testx.shape[0],28,28,1))
# print(trainy)
# print(trainy)
