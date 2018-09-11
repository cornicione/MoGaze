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



def NameList(file_dir, out_dir, txt_name):
    dirs = os.listdir(file_dir)
    dirs.sort()

    with open(out_dir + txt_name,'w') as f:
        for d in dirs:
            if d.endswith(".png"):
                f.writelines(file_dir+d)
                f.write('\n')
    f.close()

    print("Give the namelist done!")


def ReadFile(file_path, remove_pattern='\n'):
    with open(file_path) as f:
        lines = f.readlines()
    f.close()

    line = []
    for l in lines:
        line.append(l.split(remove_pattern)[0])
    lines = line

    return lines

def NamesSelect(ordered_names,selected_num = None):
    num = len(ordered_names)
    new_index = np.random.permutation(range(num))
    if selected_num is None:
        selected_num = num

    shuffle_names = []
    for n in new_index[:selected_num]:
        shuffle_names.append(ordered_names[n])

    return shuffle_names

def plot_line(img,label,flag):
    R = 200
    color = [(0, 0, 255), (0, 255, 255), (0, 255, 0), (255, 255, 0), (255, 0, 0), (255, 0, 255)]
    cx, cy = int(img.shape[0]/2),int(img.shape[1]/2)
    for index,lola in enumerate([label]):
        yaw,pitch =lola
        yaw = yaw/180 * math.pi
        pitch = pitch/180 *math.pi

        x = int(R *(math.sin(yaw)))
        y = int(R *(math.cos(yaw) * math.sin(pitch)))
        img = cv2.line(img,(cx,cy),(cx-x,cy-y),color[flag],2)
    return img


def Draw(show_img,gt=None,pre=None):
    show_head = False
    show_eye = False
    show_gaze = True

    if show_head:
        head_lola = gt['head_pose']
        cv2.putText(show_img,"head gt:{:.3f} {:.3f}".format(head_lola[0],head_lola[1]),
                    (40,20),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,255,0),1)
        draw_img = plot_line(show_img,gt['head_pose'],0)

        if not pre_label is None:
            pre_head_lola = pre['head_pose']
            cv2.putText(show_img, "head pre:{:.3f} {:.3f}".format(head_lola[0], head_lola[1]),
                        (40, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 1)
            draw_img = plot_line(show_img, gt['head_pose'], 0)


    if show_eye:
        eye_lola = gt['eye_pose']
        cv2.putText(show_img,"eye:{:.3f} {:.3f}".format(eye_lola[0],eye_lola[1]),
                    (40,40),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(255,0,0),1)
        draw_img = plot_line(show_img, gt['eye_pose'], 1)


    if show_gaze:
        gaze_lola = gt['gaze_lola']
        cv2.putText(show_img,"gaze:{:.3f} {:.3f}".format(gaze_lola[0],gaze_lola[1]),
                    (40,80),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,0,255),1)
        draw_img = plot_line(show_img, gt['gaze_lola'], 2)

    cv2.imshow("draw",draw_img)
    cv2.waitKey()

def ReadGazeTxt(gt_txt):
    ret = {}
    with open(gt_txt,"r") as f:
        while True :
            line = f.readline()
            if not line:
                break
            line = line.strip("\n")+".png"

            lo = float(f.readline().strip("\n"))
            la = float(f.readline().strip("\n"))
            ret[line] = np.array([lo,la],dtype=np.float32)
    return ret

def ReadTestResult(result_txt):
    ret = {}

    result_lines= ReadFile(result_txt)
    for ind in range(0,len(result_lines),4):
        name = result_lines[ind]
        head_pose = np.array([float(result_lines[ind + 1].split(' ')[0]),
                              float(result_lines[ind + 1].split(' ')[1])],dtype=np.float32)
        eye_pose = np.array([float(result_lines[ind + 2].split(' ')[0]),
                  float(result_lines[ind + 2].split(' ')[1])], dtype=np.float32)
        gaze_lola = np.array([float(result_lines[ind + 3].split(' ')[0]),
                  float(result_lines[ind + 3].split(' ')[1])], dtype=np.float32)
        ret[name] = [head_pose,eye_pose,gaze_lola]

    return ret




data_dir = '/home/momenta/Desktop/'
model_dir = "/home/momenta/Desktop/ckpt/"
result_dir = data_dir + 'Result/'
train_data_path = data_dir+'train/'
test_data_path = data_dir+'test/'
Show_gt = False
Test_Valid = False
Test_Test = True
Visual = True



if Show_gt:
    head_gt = ReadGazeTxt(data_dir+"head_label.txt")
    eye_gt = ReadGazeTxt(data_dir+"eye_label.txt")
    gaze_gt = ReadGazeTxt(data_dir+"gaze_label.txt")
    train_names = NamesSelect(ReadFile(data_dir + 'train_names.txt'),1000)

    for ind,img_path in enumerate(train_names):
        pic_name = img_path.split('/')[-1]
        img = cv2.imread(img_path)

        pre_hgt = head_gt[pic_name]
        pre_egt = eye_gt[pic_name]
        pre_ggt = gaze_gt[pic_name]

        pre_gt = {}
        pre_gt['head_pose'] = pre_hgt
        pre_gt['eye_pose'] = pre_egt
        pre_gt['gaze_lola'] = pre_ggt

        # if pre_gt['gaze_lola'][1] > 0:
        #     pos_gaze_angle.append(pic_name)

        if Visual and pre_gt['gaze_lola'][1] > 0:
            Draw(img,pre_gt)

        print("Now to the {}th pic, name is {}".format(ind,pic_name))


'''
    gaze_gt = ReadGazeTxt(result_dir + 'test_pred_09-09_14:43_2000.txt')
    test_names = ReadFile(data_dir + "test_names.txt")
    for ind,img_path in enumerate(test_names):
        pic_name = img_path.split('/')[-1]
        img = cv2.imread(img_path)

        pre_ggt = gaze_gt[pic_name]
        pre_gt = {}
        pre_gt['gaze_lola'] = pre_ggt

        if Visual:
            Draw(img,pre_gt)

        print("Now to the {}th pic, name is {}".format(ind,pic_name))
'''


if Test_Test:
    pre = ReadTestResult(result_dir + "TestResult_09-10_10:37_train_42000_0909_crop200.txt")
    test_names = ReadFile(data_dir + 'test_names.txt')
    for ind,img_path in enumerate(test_names):
        pic_name = img_path.split('/')[-1]
        img = cv2.imread(img_path)

        pre_label = {}
        pre_label['head_pose'] = pre[pic_name.split('.png')[0]][0]
        pre_label['eye_pose'] = pre[pic_name.split('.png')[0]][1]
        pre_label['gaze_lola'] = pre[pic_name.split('.png')[0]][2]

        if Visual:
            Draw(img,pre_label)


if Test_Valid:
    pre = ReadTestResult(result_dir + "ValidResult_09-10_09:21_train_60000_0909_crop200.txt")
    valid_names = ReadFile(data_dir + 'ValidNames.txt')

    head_gt = ReadGazeTxt(data_dir+"head_label.txt")
    eye_gt = ReadGazeTxt(data_dir+"eye_label.txt")
    gaze_gt = ReadGazeTxt(data_dir+"gaze_label.txt")

    for ind,img_path in enumerate(valid_names):
        pic_name = img_path.split('/')[-1]
        img = cv2.imread(img_path)

        hgt = head_gt[pic_name]
        egt = eye_gt[pic_name]
        ggt = gaze_gt[pic_name]

        gt_label = {}
        gt_label['head_pose'] = hgt
        gt_label['eye_pose'] = egt
        gt_label['gaze_lola'] = ggt

        pre_label = {}
        pre_label['head_pose'] = pre[pic_name.split('.png')[0]][0]
        pre_label['eye_pose'] = pre[pic_name.split('.png')[0]][1]
        pre_label['gaze_lola'] = pre[pic_name.split('.png')[0]][2]

        if Visual:
            Draw(img,gt_label,pre_label)

print("done!")







'''
## write validation set
valid_names = NamesSelect(ReadFile(data_dir + 'train_names.txt'),5000)

with open(data_dir + "ValidNames.txt","w") as f:
    for v in valid_names:
        f.writelines(v.split('/')[-1])
        f.write('\n')
f.close()

'''

