# -- encoding:utf-8 --
from keras.applications import ResNet50
from keras.applications import InceptionV3
from keras.applications import Xception
from keras.applications import VGG16
from keras.applications import VGG19
from keras.applications import imagenet_utils
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
import numpy as np
import imutils
import argparse
import tensorflow as tf
import freetype
import cv2
import os

ap=argparse.ArgumentParser()
ap.add_argument('-i','--image',required=True,
                help='path to the input image')
ap.add_argument('-model','--model',type=str,default='vgg16',
                help='name of pretrained networks to use')
args=vars(ap.parse_args())


MODELS={'vgg16':VGG16,'vgg19':VGG19,'inception':InceptionV3,'xception':Xception,
        'resnet':ResNet50}

if args['model'] not in MODELS.keys():
    raise AssertionError("The --model command line argument should be a key in the 'MODELS' dictionary")
inputshape=(224,224)
preprocess=imagenet_utils.preprocess_input
if args['model'] in ('inception','xception'):
    inputshape=(299,299)
    preprocess=preprocess_input

print('[INFO] loading{}...'.format(args['model']))
Network=MODELS[args['model']]
model=Network(weights='imagenet')

print('[INFO] loading and pre-processing image...')
image=load_img(args['image'],target_size=inputshape)

image=img_to_array(image)
image=np.expand_dims(image,axis=0)
#这里加了一维 bias
image=preprocess(image)

print("[INFO] classifying image with'{}'...".format(args['model']))
preds=model.predict(image)
# print('打印preds:',preds)
p=imagenet_utils.decode_predictions(preds)
# print('打印P:',p)
# print('打印P[0]:',p[0])
# print('打印P[0][0]:',p[0][0])
for (i,(imagenetID,label,prob)) in enumerate(p[0]):
    print('{}.{}.{:.2f}%'.format(i+1,label,prob*100))
orig=cv2.imread(args['image'])
orig=imutils.resize(orig,width=500)
# orig=cv2.imread(orig)
# ft=freetype
(imagenetID,label,prob)=p[0][0]

cv2.putText(orig,'This picture is :{}'.format(label),(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)
cv2.putText(orig,'{} has preds is:{:.2f}% '.format(label,prob*100),(10,60),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)
cv2.imshow('classification',orig)
# filename=
# cv2.imwrite()
cv2.waitKey(0)

#python imagenet_pretrained.py --image ./mypicture/1.jpg --model inception