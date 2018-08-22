# -- encoding:utf-8 --
from pyimagepreprocess.nn.perceptron import Perceptron
import numpy as np

x=np.array([[0,0],[0,1],[1,0],[1,1]])

# y=np.array([[0],[1],[1],[1]])
# y=np.array([[0],[0],[0],[1]])
y=np.array([[0],[1],[1],[0]])
# y=np.array([[1],[0],[0],[1]])

print('[INFO] training perceptron...')
p=Perceptron(x.shape[1],alpha=0.1)
p.fit(x,y,epochs=20)
print('[INFO] testing perceptron...')

for (x,target) in zip(x,y):
    pred=p.predict(x)
    print('[INFO] data={},ground-truth={},pred={}'.format(x,target,pred))