# -- encoding:utf-8 --
import cv2

class Simplepreprocessor:
    def __init__(self,wight,height,inter=cv2.INTER_AREA):
        self.weight=wight
        self.height=height
        self.inter=inter

    def preprocessor(self,image):
        return cv2.resize(image,(self.weight,self.height),interpolation=self.inter)

