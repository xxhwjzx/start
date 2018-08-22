# -- encoding:utf-8 --
from pyimagepreprocess.nn.conv.lenet import LeNet
from keras.optimizers import SGD
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import datasets
from keras import backend as K
import matplotlib.pyplot as plt
import numpy as np



from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

print('[INFO] accessing Mnist...')
file='./datasets/MNIST'
mnist=input_data.read_data_sets(file,one_hot=True)
data=mnist.train.images
# testx=mnist.test.images
target=mnist.train.labels
# testy=mnist.test.labels
# print(trainx.shape)
# data=trainx.reshape((trainx.shape[0],28,28,1))
# testx=testx.reshape((testx.shape[0],28,28,1))


# dataset=datasets.fetch_mldata('MNIST Original')
# data=dataset.data

if K.image_data_format()=='channels_first':
    data=data.reshape(data.shape[0],1,28,28)

else:
    data = data.reshape(data.shape[0], 28, 28,1)

(trainx,testx,trainy,testy)=train_test_split(data,target.astype('int'),test_size=0.25,random_state=42)
# le=LabelBinarizer()
# trainy=le.fit_transform(trainy)
# testy=le.transform(testy)

opt=SGD(0.05)
model=LeNet.build(28,28,1,10)
model.compile(optimizer=opt,loss='categorical_crossentropy',metrics=['accuracy'])
H=model.fit(trainx,trainy,validation_data=(testx,testy),batch_size=16,epochs=5,verbose=1)

print('[INFO] evaluating network...')

predictions=model.predict(testx,batch_size=16)
print(classification_report(testy.argmax(axis=1),predictions.argmax(axis=1)))

plt.style.use('ggplot')
plt.figure()
plt.plot(np.arange(0,5),H.history['loss'],label='train_loss')
plt.plot(np.arange(0,5),H.history['val_loss'],label='val_loss')
plt.plot(np.arange(0,5),H.history['acc'],label='train_acc')
plt.plot(np.arange(0,5),H.history['val_acc'],label='val_acc')
plt.title('training loss and accuracy')
plt.xlabel('epoch')
plt.ylabel('loss/accuracy')
plt.legend()
plt.show()