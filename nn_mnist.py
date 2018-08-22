# -- encoding:utf-8 --
from pyimagepreprocess.nn.neuralnetwork import Neuralnetwork
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import datasets

print('[INFO] loading MNIST datasets...')
digits=datasets.load_digits()
data=digits.data.astype('float')
data=(data-data.min())/(data.max()-data.min())
print('[INFO] samples:{},dim:{}'.format(data.shape[0],data.shape[0]))

(trainx,testx,trainy,testy)=train_test_split(data,digits.target,test_size=0.25)

trainy=LabelBinarizer().fit_transform(trainy)
testy=LabelBinarizer().fit_transform(testy)

print('[INFO] training network...')
nn=Neuralnetwork([trainx.shape[1],32,16,10])
print('[INFO] {}'.format(nn))
nn.fit(trainx,trainy,epochs=20000)
print('[INFO] evaluating network...')
predictions=nn.predict(testx)
predictions=predictions.argmax(axis=1)
print(classification_report(testy.argmax(axis=1),predictions))