# -- encoding:utf-8 --
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from pyimagepreprocess.preprocessing.imagetoarraypreprocessor import ImagetoArrayPreprocessor
from pyimagepreprocess.preprocessing.aspectawarepreprocessor import AspectAwarePreprocessor
from pyimagepreprocess.Simpledatasetloader import datasetLoader
from pyimagepreprocess.nn.conv.minivggnet import MiniVGGNet
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

ap=argparse.ArgumentParser()
ap.add_argument('-d','--dataset',required=True,
                help='path to input dataset')
args=vars(ap.parse_args())

print('[INFO] loading images...')
imagePaths=list(paths.list_images(args['dataset']))
# imagePaths=list(paths.list_images('./datasets/flower/17flowers/jpg'))
# print(imagePaths)
classname=[pt.split(os.path.sep)[-2] for pt in imagePaths]
classname=[str(x) for x in np.unique(classname)]
# print(classname)

aap=AspectAwarePreprocessor(64,64)
iap=ImagetoArrayPreprocessor()
sdl=datasetLoader(preprocessor=[aap,iap])
(data,labels)=sdl.load(imagePaths,verbose=500)
data=data.astype('float')/255.0
(trainx,testx,trainy,testy)=train_test_split(data,labels,test_size=0.25,random_state=42)
le=LabelBinarizer()
trainy=le.fit_transform(trainy)
testy=le.transform(testy)
aug=ImageDataGenerator(rotation_range=30,width_shift_range=0.1,height_shift_range=0.1,shear_range=0.2,
                       zoom_range=0.2,horizontal_flip=True,fill_mode='nearest')
print('[INFO] compiling model...')
opt=SGD(lr=0.05)
model=MiniVGGNet.build(width=64,height=64,depth=3,classes=len(classname))
model.compile(optimizer=opt,loss='categorical_crossentropy',metrics=['accuracy'])
print('[INFO] training network...')
H=model.fit_generator(aug.flow(trainx,trainy,batch_size=32),validation_data=(testx,testy),steps_per_epoch=len(trainx)//32,epochs=100,verbose=1)

print('[INFO] evaluating network...')
predictions=model.predict(testx,batch_size=32)
print(classification_report(testy.argmax(axis=1),predictions.argmax(axis=1),target_names=classname))

plt.style.use('ggplot')
plt.figure()
plt.plot(np.arange(0,100),H.history['loss'],label='train_loss')
plt.plot(np.arange(0,100),H.history['val_loss'],label='val_loss')
plt.plot(np.arange(0,100),H.history['acc'],label='acc')
plt.plot(np.arange(0,100),H.history['val_acc'],label='val_acc')
plt.title('Training Loss and Accuracy')
plt.xlabel('Epoch')
plt.ylabel('loss/acc')
plt.legend()
plt.show()
