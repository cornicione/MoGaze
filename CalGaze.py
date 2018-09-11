import torch
from torch.autograd import Variable
import numpy as np
import math
import mxnet as mx



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

# pred_head_txt = ""
# pred_eye_txt = ""
# valid_name_path = ""


head = np.array([10., 20.]).astype(np.float32)
eye = np.array([15., 25.]).astype(np.float32)
gaze_numpy = NumpyCal(head,eye)

print "done"