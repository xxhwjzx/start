# -- encoding:utf-8 --
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer
from keras.layers.core import Dense
from keras.optimizers import SGD
from keras.models import Sequential
from keras.datasets import cifar10
import numpy as np
import matplotlib.pyplot as plt
import argparse

ap=argparse.ArgumentParser()
ap.add_argument('-o','--output',required=True,
                help='path to the output loss/accuracy plot')
args=vars(ap.parse_args())

print('[INFO]  loading cifar10 data...')
((trainx,trainy),(testx,testy))=cifar10.load_data()
trainx=trainx.astype('float')/255
testx=testx.astype('float')/255
trainx=trainx.reshape((trainx.shape[0],3072))
testx=testx.reshape((testx.shape[0],3072))

lb=LabelBinarizer()
trainy=lb.fit_transform(trainy)
testy=lb.transform(testy)

labelnemes=['airplane','automobile','bird','cat',
            'deer','dog','frog','horse','ship','truck']

model=Sequential()
model.add(Dense(1024,input_shape=(3072,),activation='relu'))
model.add(Dense(512,activation='relu'))
model.add(Dense(10,activation='softmax'))

print('[INFO] training network...')
sgd=SGD(0.01)
model.compile(optimizer=sgd,loss='categorical_crossentropy',
              metrics=['accuracy'])
H=model.fit(trainx,trainy,validation_data=(testx,testy),epochs=10,
          batch_size=32)

print('[INFO] evaluating network...')
predictions=model.predict(testx,batch_size=32)
print(classification_report(testy.argmax(axis=1),
                            predictions.argmax(axis=1),
                            target_names=labelnemes))

plt.style.use('ggplot')
plt.figure()
plt.plot(np.arange(0,10),H.history['loss'],label='train_loss')
plt.plot(np.arange(0,10),H.history['val_loss'],label='val_loss')
plt.plot(np.arange(0,10),H.history['acc'],label='train_acc')
plt.plot(np.arange(0,10),H.history['val_acc'],label='val_acc')
plt.title('training loss and accuracy')
plt.xlabel('epoch')
plt.ylabel('loss/accuracy')
plt.legend()
plt.savefig(args['output'])