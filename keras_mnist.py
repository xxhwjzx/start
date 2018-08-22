# -- encoding:utf-8 --
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn import datasets
from keras.layers.core import Dense
from keras.models import Sequential
from keras.optimizers import SGD
import numpy as np
import argparse
import matplotlib.pyplot as plt

ap=argparse.ArgumentParser()
ap.add_argument('-o','--output',required=True,
                help='path to the output loss/accuracy plot')
args=vars(ap.parse_args())

print('[INFO] loading MNIST (full) dataset...')
# dataset=datasets.fetch_mldata('MNIST Original')
dataset=datasets.load_digits()
data=dataset.data.astype('float')/255
(trainx,testx,trainy,testy)=train_test_split(data,
                                             dataset.target,test_size=0.25)
lb=LabelBinarizer()
trainy=lb.fit_transform(trainy)
testy=lb.transform(testy)

model=Sequential()
model.add(Dense(16,input_shape=(64,),activation='sigmoid'))
model.add(Dense(16,activation='sigmoid'))
model.add(Dense(10,activation='softmax'))

print('[INFO] training network...')
sgd=SGD(0.01)
model.compile(loss='categorical_crossentropy',optimizer=sgd,
              metrics=['accuracy'])
H=model.fit(trainx,trainy,validation_data=(testx,testy),
            epochs=40000,batch_size=128)
print('[INFO] evaluating network...')
predictions=model.predict(testx,batch_size=128)
print(classification_report(testy.argmax(axis=1),predictions.argmax(axis=1),
                            target_names=[str(x) for x in lb.classes_]))

plt.style.use('ggplot')
plt.figure()
plt.plot(np.arange(0,100),H.history['loss'],label='train_loss')
plt.plot(np.arange(0,100),H.history['val_loss'],label='val_loss')
plt.plot(np.arange(0,100),H.history['acc'],label='train_acc')
plt.plot(np.arange(0,100),H.history['val_acc'],label='val_acc')
plt.title('Training LOSS and accuracy')
plt.xlabel('Epoch')
plt.ylabel('loss/accuracy')
plt.legend()
plt.savefig(args['output'])
