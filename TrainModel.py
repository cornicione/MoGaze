import sys, os
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
import numpy as np
import mxnet as mx
import cv2
from collections import namedtuple
Batch = namedtuple('Batch', ['data'])
epoch = 9
imgSize = 128
model_prefix = './zrn_landmark87_ResNet50'
ctx = mx.cpu()
path='./crop_img/image.txt'
# path_inverse = './checkpoints/inverse.txt'
def get_data(image):
    image = image.transpose(2, 0, 1)
    image = np.reshape(image, (1, image.shape[0], image.shape[1], image.shape[2]))
    return image


