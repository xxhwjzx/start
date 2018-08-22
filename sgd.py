# -- encoding:utf-8 --
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import argparse
import numpy as np

def sigmoid_activation(x):
    return 1.0/(1+np.exp(-x))

def predict(x,w):
    preds=sigmoid_activation(x.dot(w))
    preds[preds<=0.5]=0
    preds[preds>0.5]=1
    return preds

def next_batch(x,y,batchsize):
    for i in np.arange(0,x.shape[0],batchsize):
        yield (x[i:i+batchsize],y[i:i+batchsize])

ap=argparse.ArgumentParser()
ap.add_argument('-e','--epochs',type=float,default=100,help='# of epochs')
ap.add_argument('-a','--alpha',type=float,default=0.01,help='learning rate')
ap.add_argument('-b','--batch_size',type=int,default=32,help='size of SGD mini-batch')
args=vars(ap.parse_args())

(x,y)=make_blobs(n_samples=100,n_features=2,centers=2,cluster_std=1.5,random_state=1)
y=y.reshape(y.shape[0],1)
x=np.c_[x,np.ones((x.shape[0]))]
(trainx,testx,trainy,testy)=train_test_split(x,y,test_size=0.25,random_state=2)
w=np.random.randn(x.shape[1],1)
losses=[]

for epoch in np.arange(0,args['epochs']):
    epochloss=[]
    for (batchx,batchy) in next_batch(x,y,args['batch_size']):
        preds=sigmoid_activation(batchx.dot(w))
        error=preds-batchy
        epochloss.append(np.sum(error**2))
        gradient=batchx.T.dot(error)
        w+=-args['alpha']*gradient
    loss=np.average(epochloss)
    losses.append(loss)
    if epoch==0 or (epoch+1)%5==0:
        print('[INFO] epoch={},loss={:.7f}'.format(int(epoch+1),loss))
print('[INFO] evaluating...')
preds=predict(testx,w)
print(classification_report(testy,preds))

plt.style.use('ggplot')
plt.figure()
plt.title('Data')
plt.scatter(testx[:,0],testx[:,1],marker='o',c=testy.reshape(testy.shape[0],),s=30)

plt.style.use('ggplot')
plt.figure()
plt.plot(np.arange(0,args['epochs']),losses)
plt.title('Training loss')
plt.xlabel('Epoch')
plt.ylabel('loss')
plt.show()
