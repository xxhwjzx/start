# -- encoding:utf-8 --
from pyimagepreprocess.nn.conv.lenet import LeNet
from keras.utils import plot_model
import os

os.environ["PATH"] += os.pathsep + 'G:/Graphviz2.38/bin/'

model = LeNet.build(28, 28, 1, 10)
plot_model(model, to_file='lenet1.png', show_shapes=True)
