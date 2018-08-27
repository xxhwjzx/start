# -- encoding:utf-8 --
import imutils
import cv2

def preprocess(image,width,height):
    # cv2.imshow('image',image)
    # cv2.waitKey(0)
    (h,w)=image.shape[:2]
    if w>h:
        image=imutils.resize(image,width=width)
        # print(image.shape)
        # cv2.imshow('image', image)
        # cv2.waitKey(0)

    else:
        image=imutils.resize(image,height=height)
        # print(image.shape)
        # cv2.imshow('image', image)
        # cv2.waitKey(0)

    padw=int((width-image.shape[1])/2.0)
    padh=int((height-image.shape[0])/2.0)
    image=cv2.copyMakeBorder(image,padh,padh,padw,padw,
                             cv2.BORDER_REPLICATE)
    # cv2.imshow('image', image)
    # cv2.waitKey(0)
    # print(image.shape)
    image=cv2.resize(image,(width,height))
    return image
# image=cv2.imread('E:/AI/deep_learning_for_computer_vision/start/captcha_breaker/dataset/1/000001.png')
# print(image)
# pre=preprocess(image,width=64,height=64)