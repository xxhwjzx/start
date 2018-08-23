# -- encoding:utf-8 --
from sklearn.preprocessing import LabelBinarizer
from keras.datasets import cifar10
from pyimagepreprocess.nn.conv.minivggnet import MiniVGGNet
from sklearn.model_selection import train_test_split
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
import argparse
import os

ap=argparse.ArgumentParser()
ap.add_argument('-w','--weights',required=True,
                help='path to weights directory')
args=vars(ap.parse_args())

((trainx,trainy),(testx,testy))=cifar10.load_data()
trainx=trainx.astype('float')/255.0
testx=testx.astype('float')/255.0
le=LabelBinarizer()
trainy=le.fit_transform(trainy)
testy=le.transform(testy)

model=MiniVGGNet.build(32,32,3,10)
opt=SGD(lr=0.01,momentum=0.9,nesterov=True)
model.compile(optimizer=opt,loss='categorical_crossentropy',metrics=['accuracy'])

fname=os.path.altsep.join([args['weights'],'weights-{epoch:03d}-{val_loss:.4f}.hdf5'])
checkpoint=ModelCheckpoint(fname,monitor='val_loss',mode='min',save_best_only=True,verbose=1)
callbacks=[checkpoint]
print('[INFO] training network...')
H=model.fit(trainx,trainy,validation_data=(testx,testy),batch_size=64,epochs=40,callbacks=callbacks,verbose=1)


