import torch
from torch.autograd import Variable
import numpy as np
import math
import mxnet as mx

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


def NumpyCal(head_pose,eye_pose):
    head_lo = head_pose[0]
    head_la = head_pose[1]
    eye_lo = eye_pose[0]
    eye_la = eye_pose[1]

    cA = math.cos(head_lo / 180 * np.pi)
    sA = math.sin(head_lo / 180 * np.pi)
    cB = math.cos(head_la / 180 * np.pi)
    sB = math.sin(head_la / 180 * np.pi)
    cC = math.cos(eye_lo / 180 * np.pi)
    sC = math.sin(eye_lo / 180 * np.pi)
    cD = math.cos(eye_la / 180 * np.pi)
    sD = math.sin(eye_la / 180 * np.pi)

    g_x = - cA * sC * cD + sA * sB * sD - sA * cB * cC * cD
    g_y = cB * sD + sB * cC * cD
    g_z = sA * sC * cD + cA * sB * sD - cA * cB * cC * cD
    gaze_lo = math.atan2(-g_x, -g_z) * 180.0 / np.pi
    gaze_la = -math.asin(g_y) * 180.0 / np.pi
    gaze_lola = np.array([gaze_lo,gaze_la]).astype(np.float32)
    return gaze_lola


##
# 1. write head pred to txt (1 lines)
# 2. get eye pred from txt (4 lines)
##


head_gt = ReadGazeTxt('./data/train/head_label.txt')
eye_gt = ReadGazeTxt('./data/train/eye_label.txt')
gaze_gt = ReadGazeTxt('./data/train/gaze_label.txt')

for name,hgt in head_gt.items():
    egt = eye_gt[name]
    ggt = gaze_gt[name]
    gcal = NumpyCal(hgt,egt)
    print gcal
    print ggt

print "done"