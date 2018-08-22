# -- encoding:utf-8 --
import numpy as np
import cv2

labels=['dog','cat','panda']
np.random.seed(1)

W=np.random.randn(3,3072)
b=np.random.randn(3)

# print(W)
# print(b)

orig=cv2.imread('timg.jpg')
image=cv2.resize(orig,(32,32)).flatten()
print(image)
scores=W.dot(image)+b
print(scores)

for (label,score) in zip(labels,scores):
    print('[INFO]{}:{:.2f}'.format(label,score))

cv2.putText(orig,'Label:{}'.format(labels[np.argmax(scores)]),(100,100),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),5)
cv2.imshow('image',orig)
cv2.waitKey(0)
