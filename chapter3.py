# -- encoding:utf-8 --
# import cv2
# image=cv2.imread('timg.jpg')
# print(image.shape)
# cv2.imshow('image',image)
# cv2.waitKey(0)


def bin(n):
    if n==0:
        return 1
    elif n==1:
        return 1
    else:
        return bin(n-1)+bin(n-2)

print(bin(5))