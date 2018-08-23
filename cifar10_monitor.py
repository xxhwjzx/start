# -- encoding:utf-8 --
import matplotlib
matplotlib.use('Agg')

from callbacks.trainingmonitor import TrainningMonitor
from sklearn.preprocessing import LabelBinarizer
from pyimagepreprocess.nn.conv.minivggnet import MiniVGGNet
from keras.optimizers import SGD
from keras.datasets import cifar10
import argparse
import os

ap=argparse.ArgumentParser()
ap.add_argument('-o','--output',required=True,
                help='path to the output directory')
args=ap.parse_args()

print('[INFO] process ID:{}'.format((os.getpid())))
#返回当前进程ID,当同时运行多个程序时候，可以看到那个id的运行效果不好
print('[INFO] loading cifar10 data...')
((trainx,trainy),(testx,testy))=cifar10.load_data()
trainx=trainx.astype('float')/255.0
testx=testx.astype('float')/255.0

lb=LabelBinarizer()
trainy=lb.fit_transform(trainy)
testy=lb.transform(testy)

labelnames=['airplane','automobile','bird','cat','deer',
            'dog','frog','horse','ship','truck']

print('[INFO] compiling model...')
opt=SGD(lr=0.01,momentum=0.9,nesterov=True)
model=MiniVGGNet.build(32,32,3,10)
model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])
figpath=os.path.sep.join([args['output'],'{}.png'.format(os.getpid())])
#getpid()  Return the current process id.
jsonPath=os.path.sep.join([args['output'],'{}.json'.format(os.getpid())])
callbacks=[TrainningMonitor(figpath,jsonPath=jsonPath)]

print('[INFO] training network...')
model.fit(trainx,trainy,validation_data=(testx,testy),batch_size=64,epochs=100,
          callbacks=callbacks,verbose=1)

