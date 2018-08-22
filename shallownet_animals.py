# -- encoding:utf-8 --
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from pyimagepreprocess.preprocessing.imagetoarraypreprocessor import ImagetoArrayPreprocessor
from pyimagepreprocess.preprocessing.simplepreprocessor import Simplepreprocessor
from pyimagepreprocess.datasets.Simpledatasetloader import datasetLoader
from pyimagepreprocess.nn.conv.shallownet import ShallowNet
from keras.optimizers import SGD
import numpy as np
import matplotlib.pyplot as plt
import argparse
from imutils import paths

ap=argparse.ArgumentParser()
ap.add_argument('-d','--dataset',required=True,
                help='path to input dataset')
args=vars(ap.parse_args())

print('[INFO] loading images...')
imagePaths=list(paths.list_images(args['dataset']))

sp=Simplepreprocessor(32,32)
iap=ImagetoArrayPreprocessor()
sdl=datasetLoader(preprocessor=[sp,iap])
(data,labels)=sdl.load(imagePaths,verbose=500)
data=data.astype('float')/255.0
(trainx,testx,trainy,testy)=train_test_split(data,labels,test_size=0.25,random_state=42)
trainy=LabelBinarizer().fit_transform(trainy)
testy=LabelBinarizer().fit_transform(testy)

print('[INFO] compiling model...')
opt=SGD(0.005)
model=ShallowNet.build(width=32,height=32,depth=3,classes=3)
model.compile(loss='categorical_crossentropy',optimizer=opt,
              metrics=['accuracy'])
print('[INFO] training network...')
H=model.fit(trainx,trainy,validation_data=(testx,testy),epochs=100,batch_size=32,verbose=1)

print('[INFO] evaluating network...')
predictions=model.predict(testx,batch_size=32)
print(classification_report(testy.argmax(axis=1),predictions.argmax(axis=1),target_names=['cat','dot','panda']))


plt.style.use('ggplot')
plt.figure()
plt.plot(np.arange(0,100),H.history['loss'],label='train_loss')
plt.plot(np.arange(0,100),H.history['val_loss'],label='val_loss')
plt.plot(np.arange(0,100),H.history['acc'],label='train_acc')
plt.plot(np.arange(0,100),H.history['val_acc'],label='val_acc')
plt.title('training loss and accuracy')
plt.xlabel('epoch')
plt.ylabel('loss/accuracy')
plt.legend()
plt.show()
