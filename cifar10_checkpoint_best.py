# -- encoding:utf-8 --
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from keras.optimizers import SGD
from pyimagepreprocess.nn.conv.minivggnet import MiniVGGNet
from keras.datasets import cifar10
from keras.callbacks import ModelCheckpoint
import argparse
import os

ap=argparse.ArgumentParser()
ap.add_argument('-w','--weight',required=True,
                help='path to the best weight file ')
args=vars(ap.parse_args())

print('[INFO] loading cifar-10 data...')
((trainx,trainy),(testx,testy))=cifar10.load_data()
trainx=trainx.astype('float')/255.0
testx=testx.astype('float')/255.0

lb=LabelBinarizer()
trainy=lb.fit_transform(trainy)
testy=lb.transform(testy)

model=MiniVGGNet.build(32,32,3,10)
opt=SGD(lr=0.01,momentum=0.9,nesterov=True)
model.compile(optimizer=opt,loss='categorical_crossentropy',metrics=['accuracy'])

checkpoint=ModelCheckpoint(args['weight'],monitor='val_loss',save_best_only=True,verbose=1)
callbacks=[checkpoint]

H=model.fit(trainx,trainy,validation_data=(testx,testy),epochs=40,batch_size=64,callbacks=callbacks,verbose=1)
