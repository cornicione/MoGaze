
#coding=utf-8
from __future__ import division
import math
import os
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.utils.data import Dataset,DataLoader
from imgaug import augmenters as iaa
import torch
import cv2
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
from torch.nn import SmoothL1Loss
import time
import gc
from numpy.linalg import norm,inv
from torch.autograd import Variable

####### add KeyPtNet() for keypoints regression
#######

class KeyPtNet(nn.Module):

    def __init__(self):
        super(KeyPtNet, self).__init__()

        #
        self.face_kpt = resnext50(4,64)
        self.eye_kpt = resnext50(4,64)


    def forward(self,img_face,img_eye):
        return_dict = {}
        head_pt = self.face_kpt(img_face)
        eye_pt = self.eye_kpt(img_eye)

        return_dict['head_pt'] = head_pt
        return_dict['eye_pt'] = eye_pt

        return return_dict


####### add KeyPtDataset for keypoints dataset

class KeyPtDataset(Dataset):
    def __init__(self,data_dir,mode,transform = None ):
        """
        Args:
        data_dir (string): Directory with all the images.
        mode (string): train/val/test subdirs.
        transform (callable, optional): Optional transform to be applied
        on a sample.
        """
        self.data_dir = data_dir
        self.transform = transform
        self.mode = mode
        self.img_list = os.listdir(os.path.join(data_dir,"head_pt"))
        if self.mode == "train":
            self.head_pt_label = self.load_kpt(os.path.join(data_dir,"head_pt.txt"))
            self.eye_pt_label = self.load_kpt(os.path.join(data_dir,"eye_pt.txt"))


    def load_kpt(self,filename):

        #load keypoints of faces
        ret = {}
        with open(filename,'r') as kptfile:
            while True :
                line = kptfile.readline()
                if not line:
                    break
                img_filename = line.strip("\n")
                src_point = []
                p_count = int(line.strip("\n"))
                for j in range(p_count):

                    # resize image (896,896)----(224,224)

                    x = float(kptfile.readline().strip("\n")) * 0.25
                    y = float(kptfile.readline().strip("\n")) * 0.25
                    src_point.append((x,y))
                ret[img_filename] = src_point
        return ret


    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):

        head_image = cv2.imread(os.path.join(self.data_dir, "head", self.img_list[idx]), cv2.IMREAD_GRAYSCALE)

        # 头部图像还包含了大量背景区域,需要做居中裁剪
        #224 *224
        mid_x, mid_y = head_image.shape[0] // 2, head_image.shape[1] // 2
        head_image = head_image[mid_x - 112:mid_x + 112, mid_y - 112:mid_y + 112]
        leye_image = cv2.imread(os.path.join(self.data_dir, "l_eye", self.img_list[idx]), cv2.IMREAD_GRAYSCALE)
        reye_image = cv2.imread(os.path.join(self.data_dir, "r_eye", self.img_list[idx]), cv2.IMREAD_GRAYSCALE)
        eye_image = leye_image if np.random.rand() < 0.5 else reye_image

        head_image = image_normalize(head_image)
        eye_image = image_normalize(eye_image)
        if self.mode == "train":
            head_pt = self.head_pt_label[self.img_list[idx]]
            eye_pt = self.eye_pt_label[self.img_list[idx]]

            sample = {'img_name': self.img_list[idx], 'head_image': head_image, 'eye_image': eye_image,
                      'head_pt': head_pt, 'eye_pt': eye_pt}
        else:
            sample = {'img_name': self.img_list[idx], 'head_image': head_image, 'eye_image': eye_image}

        return sample