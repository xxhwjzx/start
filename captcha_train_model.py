# -- encoding:utf-8 --
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.preprocessing.image import img_to_array
from keras.optimizers import SGD
from pyimagepreprocess.nn.conv.lenet import LeNet
from pyimagepreprocess.utils.captchahelper import preprocess
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2
import os


ap=argparse.ArgumentParser()
ap.add_argument('-d','--dataset',required=True,
                help='path to input dataset')
ap.add_argument('-m','--model',required=True,
                help='path to output model')
args=vars(ap.parse_args())

data=[]
labels=[]

for imagePath in paths.list_images(args['dataset']):
    image=cv2.imread(imagePath)
    image=cv2.cvtColor(imagePath,cv2.COLOR_BGR2GRAY)
    image=preprocess(image,28,28)
    image=img_to_array(image)
    data.append(image)
    label=imagePath.split(os.path.altsep)[-2]
    labels.append(label)


data=np.array(data,dtype='float')/255.0
labels=np.array(labels)

(trainx,testx,trainy,testy)=train_test_split(data,labels,test_size=0.25,random_state=42)
lb=LabelBinarizer()
trainy=lb.fit_transform(trainy)
testy=lb.transform(testy)

model=LeNet.build(width=28,height=28,depth=1,classes=9)
opt=SGD(lr=0.01)
model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])

print('[INFO] training network...')
H=model.fit(trainx,trainy,validation_data=(testx,testy),batch_size=32,epochs=15,verbose=1)
print('[INFO] evaluating network...')
predictions=model.predict(testx,batch_size=32)
print(classification_report(testy.argmax(axis=1),predictions.argmax(axis=1),target_names=lb.classes_))

print('[INFO] serializing network...')
model.save(args['model'])

plt.style.use('ggplot')
plt.figure()
plt.plot(np.arange(0,15),H.history['loss'],label='train_loss')
plt.plot(np.arange(0,15),H.history['val_loss'],label='val_loss')
plt.plot(np.arange(0,15),H.history['acc'],label='acc')
plt.plot(np.arange(0,15),H.history['val_acc'],label='val_acc')
plt.title('Training Loss and Accuracy')
plt.xlabel('Epoch')
plt.ylabel('loss/acc')
plt.legend()
plt.show()

