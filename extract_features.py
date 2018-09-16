# -- encoding:utf-8 --
#导入各种包
from keras.applications import VGG16
from keras.applications import ResNet50
from keras.applications import imagenet_utils
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from sklearn.preprocessing import LabelEncoder
from pyimagepreprocess.io.hdf5datasetwriter import HDF5DatasetWriter
from imutils import paths
import numpy as np
import progressbar
import argparse
import random
import os
#args命令
ap = argparse.ArgumentParser()
ap.add_argument('-d', '--dataset', required=True,
                help='path to input dataset')
ap.add_argument('-o', '--output', required=True,
                help='path to output HDF5 file')
ap.add_argument('-b', '--batch_size', type=int, default=32,
                help='batch size of images to be passed through network')
ap.add_argument('-s', '--buffer_size', type=int, default=1000,
                help='size of feature extraction buffer')
args = vars(ap.parse_args())

bs = args['batch_size']
print('[INFO] loading images...')

#导入图片文件为列表
#['./datasets/animals/cats\\cats_00001.jpg', './datasets/animals/cats\\cats_00002.jpg',
#  './datasets/animals/cats\\cats_00003.jpg'] 样式
imagePaths = list(paths.list_images(args['dataset']))
#随即打乱文件
random.shuffle(imagePaths)
#提取标签
labels = [p.split(os.path.sep)[-2].split(os.path.altsep)[-1] for p in imagePaths]
# print(labels)
#实例化二值类
le = LabelEncoder()
#把标签全部转换成1、2、3..
labels = le.fit_transform(labels)
#载入网络
print('[INFO] loading network...')
#特征提取模型选用ResNet50的imagenet的特征 不包含头部全连接
model = ResNet50(weights='imagenet', include_top=False)
#数据用HDF5写入 第一个是数据的维度
dataset = HDF5DatasetWriter((len(imagePaths), 512 * 7 * 7), args['output'], dataKey='feature',
                            bufSize=args['buffer_size'])
#存储标签 这里是['cats' 'dogs' 'panda']
dataset.storeClassLabels(le.classes_)
#构建进度条小部件
widgets = ['Extracting Features:', progressbar.Percentage(), ' ', progressbar.Bar(), ' ', progressbar.ETA()]
#进度条bar构建
pbar = progressbar.ProgressBar(maxval=len(imagePaths), widgets=widgets).start()

#构建批次
for i in np.arange(0, len(imagePaths), bs):
    #批次文件构建 此处一个bs为32
    batchPaths = imagePaths[i:i + bs]
    # 批次标签构建 此处一个bs为32
    batchLabels = labels[i:i + bs]
    #构建一个空的图片列表
    batchImages = []

    for (j, imagePath) in enumerate(batchPaths):
        #把图片以目标格式载入到内存
        image = load_img(imagePath, target_size=(224, 224))
        #把图片变成数组
        image = img_to_array(image)
        #在首位增加一个维度
        image = np.expand_dims(image, axis=0)
        #
        image = imagenet_utils.preprocess_input(image)
        batchImages.append(image)

    batchImages = np.vstack(batchImages)
    features = model.predict(batchImages, batch_size=bs)
    features = features.reshape((features.shape[0], 512 * 7 * 7))
    dataset.add(features, batchLabels)
    pbar.update(i)

dataset.close()
pbar.finish()

# E:\AI\deep_learning_for_computer_vision\start>python extract_features.py --dataset ./datasets/animals/ --output ./datasets/animals/features.hdf5
