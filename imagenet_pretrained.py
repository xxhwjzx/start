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
import argparse
import cv2

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
p=imagenet_utils.decode_predictions

