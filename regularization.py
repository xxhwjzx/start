# -- encoding:utf-8 --
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from pyimagepreprocess.simplepreprocessor import Simplepreprocessor
from pyimagepreprocess.Simpledatasetloader import datasetLoader
from imutils import paths
import argparse

ap=argparse.ArgumentParser()
ap.add_argument('-d','--dataset',default='./datasets/animals',help='path to input dataset')
args=vars(ap.parse_args())

print('[INFO]loading images...')
imagepaths=list(paths.list_images(args['dataset']))
sp=Simplepreprocessor(32,32)
sdl=datasetLoader(preprocessor=[sp])
(data,labels)=sdl.load(imagepaths,verbose=500)
data=data.reshape((data.shape[0]),3072)
le=LabelEncoder()
labels=le.fit_transform(labels)

(trainx,testx,trainy,testy)=train_test_split(data,labels,test_size=0.25,random_state=3)

for r in (None,'l1','l2'):
    print("[INFO] training model with '{}' penalty'".format(r))
    model=SGDClassifier(loss='log',penalty=r,max_iter=10,learning_rate='constant',eta0=0.01,
                        random_state=6)
    model.fit(trainx,trainy)
    acc=model.score(testx,testy)
    print("[INFO] '{}'penalty accuracy:{:.2f}%".format(r,acc*100))