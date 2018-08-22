# -- encoding:utf-8 --
from skimage.exposure import rescale_intensity
import numpy as np
import argparse
import cv2

def convolve(image,K):
    (ih,iw)=image.shape[:2]
    (kh,kw)=K.shape[:2]

    pad=(kw-1)//2
    image=cv2.copyMakeBorder(image,pad,pad,pad,pad,cv2.BORDER_REPLICATE)
    output=np.zeros((ih,iw),dtype='float')
    for y in np.arange(pad,ih+pad):
        for x in np.arange(pad,iw+pad):
            roi=image[y-pad:y+pad+1,x-pad:x+pad+1]
            k=(roi*K).sum()
            output[y-pad,x-pad]=k
    output=rescale_intensity(output,in_range=(0,255))
    output=(output*255).astype('uint8')
    return output

ap=argparse.ArgumentParser()
ap.add_argument('-i','--image',required=True,
                help='path to the image')
args=vars(ap.parse_args())

smallBlur=np.ones((7,7),dtype='float')*(1.0/(7*7))
largeBlur=np.ones((21,21),dtype='float')*(1.0/(21*21))   #模糊

sharpen=np.array((
    [0,-1,0],
    [-1,5,-1],
    [0,-1,0]
),dtype='int')            #锐化


laplacian=np.array((
    [0,1,0],
    [1,-4,1],
    [0,1,0]
),dtype='int')      #发现轮廓

sobelx=np.array((
    [-1,0,1],
    [-2,0,2],
    [-1,0,1]
),dtype='int')      #X轴扩展

sobely=np.array((
    [-1,-2,-1],
    [0,0,0],
    [1,2,1]
),dtype='int')      #y轴扩展

emboss=np.array((
    [-2,-1,0],
    [-1,1,1],
    [0,1,2]
),dtype='int')    #浮饰


kernelBank=(('small_blur',smallBlur),
            ('large_blur',largeBlur),
            ('sharpen',sharpen),
            ('laplacian',laplacian),
            ('sobel_x',sobelx),
            ('sobel_y',sobely),
            ('emboss',emboss))


image=cv2.imread(args['image'])
gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

for (kernelname,K) in kernelBank:
    print('[INFO] applying{}kernel'.format(kernelname))
    convolveoutput=convolve(gray,K)
    opencvoutput=cv2.filter2D(gray,-1,K)
    i=0
    cv2.imshow('original',gray)
    cv2.imwrite('./output/原图_{}.png'.format(i),gray)
    cv2.imshow('{}-convole'.format(kernelname),convolveoutput)
    cv2.imwrite('./output/convole_{}_{}.png'.format(kernelname,i), convolveoutput)
    cv2.imshow('{}-opencv'.format(kernelname),opencvoutput)
    cv2.imwrite('./output/opencvoutput_{}_{}.png'.format(kernelname,i), opencvoutput)
    cv2.waitKey(0)
    cv2.destroyAllWindows()