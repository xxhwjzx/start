# -- encoding:utf-8 --
from keras.preprocessing.image import img_to_array

class ImagetoArrayPreprocessor:
    def __init__(self,dataFormat=None):
        self.dataFormat=dataFormat

    def preprocessor(self,image):
        return img_to_array(image,data_format=self.dataFormat)