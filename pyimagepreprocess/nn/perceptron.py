# -- encoding:utf-8 --
import numpy as np

class Perceptron:
    def __init__(self,N,alpha=0.1):

        self.w=np.random.randn(N+1)/np.sqrt(N)
        self.alpht=alpha

    def step(self,x):
        return 1 if x>0 else 0

    def fit(self,x,y,epochs=10):
        x=np.c_[x,np.ones((x.shape[0]))]
        for epoch in np.arange(0,epochs):
            for (xi,target) in zip(x,y):
                p=self.step(np.dot(xi,self.w))
                if p!=target:
                    error=p-target
                    self.w+=-self.alpht*error*xi

    def predict(self,x,addBias=True):
        x=np.atleast_2d(x)
        if addBias:
            x=np.c_[x,np.ones((x.shape[0]))]
        return self.step(np.dot(x,self.w))

# x=np.array([[0,0],[0,1],[1,0],[1,1]])
# y=np.atleast_2d(x)
# print(y)
# y=np.array([[0],[1],[1],[1]])
# w=np.random.randn(x.shape[1])/np.sqrt(x.shape[1])
#
# print(np.dot(x,w))
#
# def step(x):
#     return 1 if x.all() > 0 else 0
# for (xi,yi) in zip(x,y):
#     z=step(np.dot(xi,w))
#     print(z)

