# -- encoding:utf-8 --
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np
import argparse

def sigomid_activation(x):
    return 1.0/(1+np.exp(-x))

def predict(X,W):
    preds=sigomid_activation(X.dot(W))
    preds[preds<=0.5]=0
    preds[preds>0.5]=1
    return preds

ap=argparse.ArgumentParser()
ap.add_argument('-e','--epochs',type=float,default=100,help='# of epochs')
ap.add_argument('-a','--alpha',type=float,default=0.01,help='learning rate')
args=vars(ap.parse_args())

(X,y)=make_blobs(n_samples=100,n_features=2,centers=2,cluster_std=1.5,random_state=1)
print(X)
print(y)
y=y.reshape(y.shape[0],1)

X=np.c_[X,np.ones((X.shape[0]))]
print(X.shape[0])
print(X)
(trainx,testx,trainy,testy)=train_test_split(X,y,test_size=0.5,random_state=42)

print('[INFO] training...')
W=np.random.randn(X.shape[1],1)
losses=[]

for epoch in np.arange(0,args['epochs']):
    preds=sigomid_activation(trainx.dot(W))
    error=preds-trainy
    loss=np.sum(error**2)
    losses.append(loss)

    gradient=trainx.T.dot(error)
    W+=-args['alpha']*gradient
    if epoch==0 or (epoch%5)==0:
        print('[INFO] epoch={},loss={:.7f}'.format(int(epoch+1),loss))

print('[INFO] evaluating...')
preds=predict(testx,W)
print(classification_report(testy,preds))


plt.style.use('ggplot')
plt.figure()
plt.title('Data')
plt.scatter(testx[:,0],testx[:,1],marker='o',c=testy.reshape(testy.shape[0],),s=30)
# print(testx[:,0])
# print(testx[:,1])
# print(testy)
# print(testy.reshape(testy.shape[0],))
plt.style.use('ggplot')
plt.figure()
plt.plot(np.arange(0,args['epochs']),losses)
plt.title('Training Loss')
plt.xlabel('Epochs #')
plt.ylabel('Loss')
plt.show()