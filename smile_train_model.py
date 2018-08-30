# -- encoding:utf-8 --
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.preprocessing.image import img_to_array
from keras.utils import np_utils
from pyimagepreprocess.nn.conv.lenet import LeNet
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import imutils
import cv2

ap=argparse.ArgumentParser()
ap.add_argument('-d','--dataset',required=True,
                help='path to input dataset of faced')
ap.add_argument('-m','--model',required=True,
                help='path to output model')
args=vars(ap.parse_args())

data=[]
labels=[]

for imagePath in sorted(list(paths.list_images(args['dataset']))):
    image=cv2.imread(imagePath)
    image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    image=imutils.resize(image,width=28)
    image=img_to_array(image)
    data.append(image)
    label=imagePath.split(os.path.altsep)[-1].split(os.path.sep)[-3]
    # print(label)
    label='smiling' if label=='positives' else 'not_smiling'
    labels.append(label)


data=np.array(data,dtype='float')/255.0
labels=np.array(labels)

le=LabelEncoder().fit(labels)
print('letransform',le.transform(labels))
labels=np_utils.to_categorical(le.transform(labels),2)

classtotals=labels.sum(axis=0)
classweight=classtotals.max()/classtotals

(trainx,testx,trainy,testy)=train_test_split(data,labels,test_size=0.20,stratify=labels,random_state=42)

print('[INFO] compiling model....')
model=LeNet.build(28,28,1,2)
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

print('[INFO] training network...')
H=model.fit(trainx,trainy,validation_data=(testx,testy),class_weight=classweight,batch_size=64,epochs=15,verbose=1)

print('[INFO] evaluating network...')
predictions=model.predict(testx,batch_size=64)
print(classification_report(testy.argmax(axis=1),predictions.argmax(axis=1),target_names=le.classes_))

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

#E:\AI\deep_learning_for_computer_vision\start>python smile_train_model.py --dataset ./datasets/SMILEsmileD --model ./output/smilev1.hdf5

