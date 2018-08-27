# -- encoding:utf-8 --
import argparse
import requests
import time
import os

ap=argparse.ArgumentParser()
ap.add_argument('-o','--output',required=True,
                help='path to output directory of images')
ap.add_argument('-n','--num-images',type=int,default=500,
                help='# of images to download')
args=vars(ap.parse_args())

url='https://www.e-zpassny.com/vector/jcaptcha.do'