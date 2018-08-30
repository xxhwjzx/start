# -- encoding:utf-8 --
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import os
import cv2
ap=argparse.ArgumentParser()
ap.add_argument('-c','--cascade',required=True,
                help='path to where the face cascade resides')
ap.add_argument('-m','--model',required=True,
                help='path to pre-trained smile detector CNN')
ap.add_argument('-v','--video',help='path to the (optional) video file ')
args=vars(ap.parse_args())

detector=cv2.CascadeClassifier(args['cascade'])
model=load_model(args['model'])
if not args.get('video',False):
    camera=cv2.VideoCapture(0)

else:
    camera=cv2.VideoCapture(args['video'])

while True:
    (grabbed,frame)=camera.read()
    if args.get('videoo') and not grabbed:
        break

    frame=imutils.resize(frame,height=500)   #保持纵横比，根据高度修改图像像素
    (h,w)=frame.shape[:2]                    #获取高h 宽w
    center=(w//2,h//2)                       #获取图像中心
    M=cv2.getRotationMatrix2D(center,0,1)    #实例化cv2旋转函数，第一个参数是中心，第二个参数是角度，第三个参数是缩放大小
    frame=cv2.warpAffine(frame,M,(w,h))      #进行旋转 并将旋转后的图片赋值给frame
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    frameClone=frame.copy()
    rects=detector.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(30,30),flags=cv2.CASCADE_SCALE_IMAGE)
    for (fx,fy,fw,fh) in rects:
        roi=gray[fy:fy+fh,fx:fx+fw]
        roi=cv2.resize(roi,(28,28))
        roi=roi.astype('float')/255.0
        roi=img_to_array(roi)
        roi=np.expand_dims(roi,axis=0)
        (notsmiling,smiling)=model.predict(roi)[0]
        print('没笑：{},笑了：{}'.format(notsmiling,smiling))
        label='Smiling' if smiling>notsmiling else 'Not Smiling'
        cv2.putText(frameClone,label,(fx,fy-10),cv2.FONT_HERSHEY_SIMPLEX,0.45,(0,0,255),2)
        cv2.rectangle(frameClone,(fx,fy),(fx+fw,fy+fh),(0,0,255),2)
    cv2.imshow('face',frameClone)
    if cv2.waitKey(1) & 0xFF ==ord('q'):
        break

camera.release()
cv2.destroyAllWindows()

#E:\AI\deep_learning_for_computer_vision\start>python smile_detect.py --cascade ./face/haarcascade_frontalface_default.xml --model ./output/smilev1.hdf5 --video./mypicture/mymp43.mp4