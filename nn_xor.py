# -- encoding:utf-8 --
from pyimagepreprocess.nn.neuralnetwork import Neuralnetwork
import numpy as np

X=np.array([[0,0],[0,1],[1,0],[1,1]])
# y=np.array([[0],[1],[1],[1]])
# y=np.array([[0],[0],[0],[1]])
y=np.array([[0],[1],[1],[0]])
# y=np.array([[1],[0],[0],[1]])

NN=Neuralnetwork([2,10,1])
NN.fit(X,y,epochs=40000)
for (x,target) in zip(X,y):
    pred=NN.predict(x)[0][0]
    step=1 if pred>0.5 else 0
    print('[INFO] data={},ground-truth={},pred={:.4f},step={}'.format(x,target,pred,step))