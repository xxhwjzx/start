# -- encoding:utf-8 --
from pyimagepreprocess.datasets.Simpledatasetloader import datasetLoader
from pyimagepreprocess.preprocessing.simplepreprocessor import Simplepreprocessor
from pyimagepreprocess.preprocessing.imagetoarraypreprocessor import ImagetoArrayPreprocessor
from keras.models import load_model
from imutils import paths
import numpy as np
import argparse
import cv2

ap=argparse.ArgumentParser()
ap.add_argument('-d','--dataset',required=True,
                help='path to input dataset')
ap.add_argument('-m','--model',required=True,
                help='path to pre-trained model')
args=vars(ap.parse_args())

classlabels=['cat','dog','panda']

print('[INFO] sampling images...')
imagepaths=np.array(list(paths.list_images(args['dataset'])))
# imagepaths=np.array(list(paths.list_images('./datasets/animals')))
print(imagepaths)
idxs=np.random.randint(0,len(imagepaths),size=(2,))
imagepaths=imagepaths[idxs]
print(imagepaths)

sp=Simplepreprocessor(32,32)
apl=ImagetoArrayPreprocessor()

sdl=datasetLoader(preprocessor=[sp,apl])
(data,labels)=sdl.load(imagepaths)
#
data=data.astype('float')/255.0
print(data)
print('[INFO] loading pre-trained network...')
model=load_model(args['model'])

print('[INFO] predicting...')

# preds=model.predict(data,batch_size=32).argmax(axis=1)
preds=model.predict(data).argmax(axis=1)
print(preds)
for (i,imagepath) in enumerate(imagepaths):
    image=cv2.imread(imagepath)
    cv2.putText(image,'label:{}'.format(classlabels[preds[i]]),(10,30),
                cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
    cv2.imshow('Image',image)
    cv2.waitKey(0)