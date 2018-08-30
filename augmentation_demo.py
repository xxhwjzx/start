# -- encoding:utf-8 --
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
import numpy as np
import argparse

ap=argparse.ArgumentParser()
ap.add_argument('-i','--image',required=True,
                help='path to the input image')
ap.add_argument('-o','--output',required=True,
                help='path to output directory to store augmentation example')
ap.add_argument('-p','--prefix',type=str,default='image',
                help='output filename prefix')
args=vars(ap.parse_args())

print('[INFO] loading example image...')
image=load_img(args['image'])
image=img_to_array(image)
image=np.expand_dims(image,axis=0)
aug=ImageDataGenerator(rotation_range=15,width_shift_range=0.1,height_shift_range=0.1,
                       shear_range=0.2,zoom_range=0.2,horizontal_flip=False,fill_mode='nearest')
total=0
print('[INFO] generating images...')
imageGen=aug.flow(image,batch_size=1,save_to_dir=args['output'],
                  save_prefix=args['prefix'],save_format='jpg')
for image in imageGen:
    total+=1
    if total==10:
        break




# featurewise_center：布尔值，使输入数据集去中心化（均值为0）, 按feature执行。
# samplewise_center：布尔值，使输入数据的每个样本均值为0。
# featurewise_std_normalization：布尔值，将输入除以数据集的标准差以完成标准化, 按feature执行。
# samplewise_std_normalization：布尔值，将输入的每个样本除以其自身的标准差。
# zca_whitening：布尔值，对输入数据施加ZCA白化。
# rotation_range：整数，数据提升时图片随机转动的角度。随机选择图片的角度，是一个0~180的度数，取值为0~180。
# width_shift_range：浮点数，图片宽度的某个比例，数据提升时图片随机水平偏移的幅度。
# height_shift_range：浮点数，图片高度的某个比例，数据提升时图片随机竖直偏移的幅度。 
# height_shift_range和width_shift_range是用来指定水平和竖直方向随机移动的程度，这是两个0~1之间的比例。
# shear_range：浮点数，剪切强度（逆时针方向的剪切变换角度）。是用来进行剪切变换的程度。
# zoom_range：浮点数或形如[lower,upper]的列表，随机缩放的幅度，若为浮点数，则相当于[lower,upper] = [1 - zoom_range, 1+zoom_range]。用来进行随机的放大。
# channel_shift_range：浮点数，随机通道偏移的幅度。
# fill_mode：‘constant’，‘nearest’，‘reflect’或‘wrap’之一，当进行变换时超出边界的点将根据本参数给定的方法进行处理
# cval：浮点数或整数，当fill_mode=constant时，指定要向超出边界的点填充的值。
# horizontal_flip：布尔值，进行随机水平翻转。随机的对图片进行水平翻转，这个参数适用于水平翻转不影响图片语义的时候。
# vertical_flip：布尔值，进行随机竖直翻转。
# rescale: 值将在执行其他处理前乘到整个图像上，我们的图像在RGB通道都是0~255的整数，这样的操作可能使图像的值过高或过低，所以我们将这个值定为0~1之间的数。
# preprocessing_function: 将被应用于每个输入的函数。该函数将在任何其他修改之前运行。该函数接受一个参数，为一张图片（秩为3的numpy array），并且输出一个具有相同shape的numpy array
# data_format：字符串，“channel_first”或“channel_last”之一，代表图像的通道维的位置。该参数是Keras 1.x中的image_dim_ordering，“channel_last”对应原本的“tf”，“channel_first”对应原本的“th”。以128x128的RGB图像为例，“channel_first”应将数据组织为（3,128,128），而“channel_last”应将数据组织为（128,128,3）。该参数的默认值是~/.keras/keras.json中设置的值，若从未设置过，则为“channel_last”。
