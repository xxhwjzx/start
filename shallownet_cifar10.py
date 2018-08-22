# -- encoding:utf-8 --
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.optimizers import SGD
from keras.datasets import cifar10
from pyimagepreprocess.nn.conv.shallownet import ShallowNet
import matplotlib.pyplot as plt
import numpy as np

print('[INFO] loading CIFAR10...')
((trainx,trainy),(testx,testy))=cifar10.load_data()
trainx=trainx.astype('float')/255
testx=testx.astype('float')/255

lb=LabelBinarizer()
trainy=lb.fit_transform(trainy)
testy=lb.transform(testy)

labelnemes=['airplane','automobile','bird','cat',
            'deer','dog','frog','horse','ship','truck']

print('[INFO] compiling model...')
opt=SGD(0.01)
model=ShallowNet.build(width=32,height=32,depth=3,classes=10)
model.compile(optimizer=opt,metrics=['accuracy'],loss='categorical_crossentropy')

print('[INFO] training network...')
H=model.fit(trainx,trainy,validation_data=(testx,testy),epochs=40,batch_size=32,verbose=1)

print('[INFO] evaluating network...')
predictions=model.predict(testx,batch_size=32)
print(classification_report(testy.argmax(axis=1),predictions.argmax(axis=1),target_names=labelnemes))

plt.style.use('ggplot')
plt.figure()
plt.plot(np.arange(0,40),H.history['loss'],label='train_loss')
plt.plot(np.arange(0,40),H.history['val_loss'],label='val_loss')
plt.plot(np.arange(0,40),H.history['acc'],label='train_acc')
plt.plot(np.arange(0,40),H.history['val_acc'],label='val_acc')
plt.title('training loss and accuracy')
plt.xlabel('epoch')
plt.ylabel('loss/accuracy')
plt.legend()
plt.show()