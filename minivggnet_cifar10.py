# -- encoding:utf-8 --
import matplotlib
matplotlib.use('Agg')

from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from pyimagepreprocess.nn.conv.minivggnet import MiniVGGNet
from keras.optimizers import SGD
from keras.datasets import cifar10
import matplotlib.pyplot as plt
import numpy as np
import argparse

ap=argparse.ArgumentParser()
ap.add_argument('-o','--output',required=True,
                help='path to the output loss/acc plot')
args=vars(ap.parse_args())


print('[INFO] loading cifar-10 dataset...')
((trainx,trainy),(testx,testy))=cifar10.load_data()
trainx=trainx.astype('float')/255.0
testx=testx.astype('float')/255.0

# print(testy[0])
lb=LabelBinarizer()
trainy=lb.fit_transform(trainy)
testy=lb.transform(testy)
# print(testy[0])
# print(lb.classes_)

labelnames=['airplane','automobile','bird','cat','deer',
            'dog','frog','horse','ship','truck']

print('[INFO] compiling model...')
opt=SGD(lr=0.01,decay=0.01/40,momentum=0.9,nesterov=True)
model=MiniVGGNet.build(width=32,height=32,depth=3,classes=10)
model.compile(optimizer=opt,loss='categorical_crossentropy',metrics=['accuracy'])
print('[INFO] training network...')

H=model.fit(trainx,trainy,validation_data=(testx,testy),batch_size=64,epochs=4,verbose=1)
print('[INFO] evaluating network...')
predictions=model.predict(testx,batch_size=64)
print(classification_report(testy.argmax(axis=1),predictions.argmax(axis=1),target_names=labelnames))

plt.style.use('ggplot')
plt.figure()
plt.plot(np.arange(0,4),H.history['loss'],label='train_loss')
plt.plot(np.arange(0,4),H.history['val_loss'],label='val_loss')
plt.plot(np.arange(0,4),H.history['acc'],label='train_acc')
plt.plot(np.arange(0,4),H.history['val_acc'],label='val_acc')
plt.title('training loss and accuracy on cifar-10')
plt.xlabel('epoch')
plt.ylabel('loss/acc')
plt.legend()
plt.savefig(args['output'])