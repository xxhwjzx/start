# -- encoding:utf-8 --

import matplotlib
matplotlib.use('Agg')

from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from pyimagepreprocess.nn.conv.minivggnet import MiniVGGNet
from keras.callbacks import LearningRateScheduler
from keras.optimizers import SGD
from keras.datasets import cifar10
import matplotlib.pyplot as plt
import numpy as np
import argparse

def step_decay(epoch):
    initAlpha=0.01
    factor=0.25
    dropevery=5
    alpha=initAlpha*(factor**((epoch+1)/dropevery))
    return float(alpha)

ap=argparse.ArgumentParser()
ap.add_argument('-o','--output',required=True,
                help='path to the output loss/acc plot')
args=vars(ap.parse_args())

print('[INFO] loading cifar10 data...')
((trainx,trainy),(testx,testy))=cifar10.load_data()
trainx=trainx.astype('float')/255.0
testx=testx.astype('float')/255.0
le=LabelBinarizer()
trainy=le.fit_transform(trainy)
testy=le.transform(testy)

labelnames=['airplane','automobile','bird','cat','deer',
            'dog','frog','horse','ship','truck']

callbacks=[LearningRateScheduler(step_decay)]

model=MiniVGGNet.build(32,32,3,10)
opt=SGD(lr=0.01,momentum=0.9,nesterov=True)
model.compile(optimizer=opt,loss='categorical_crossentropy',metrics=['accuracy'])
H=model.fit(trainx,trainy,validation_data=(testx,testy),batch_size=64,epochs=40,callbacks=callbacks,verbose=1)

print('[INFO] evaluating network...')
predictions=model.predict(testx,batch_size=64)
print(classification_report(testy.argmax(axis=1),predictions.argmax(axis=1),target_names=labelnames))

plt.style.use('ggplot')
plt.figure()
plt.plot(np.arange(0,40),H.history['loss'],label='train_loss')
plt.plot(np.arange(0,40),H.history['val_loss'],label='val_loss')
plt.plot(np.arange(0,40),H.history['acc'],label='train_acc')
plt.plot(np.arange(0,40),H.history['val_acc'],label='val_acc')
plt.title('training loss and accuracy on cifar-10')
plt.xlabel('epoch')
plt.ylabel('loss/acc')
plt.legend()
plt.savefig(args['output'])


