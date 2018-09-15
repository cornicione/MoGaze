
import os
import math
import numpy as np



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


def lola2vec(lola):
    lo = lola[0]
    la = lola[1]
    vec = np.zeros(3,dtype = np.float64)
    vec[1] = np.sin(-la/180.0*np.pi)
    rate = (np.sin(-lo/180.0*np.pi))**2
    sign = -1 if lo > 0 else 1
    vec[0] = sign*np.sqrt((1.0-vec[1]**2)*rate)
    vec[2] = -np.sqrt((1.0-vec[1]**2)*(1-rate))
    return vec


def vec2lola(vec):
    vec = vec / np.linalg.norm(vec)
    lo = -np.arcsin(vec[0]/np.sqrt(vec[0]*vec[0]+vec[2]*vec[2]))*180.0/np.pi
    la = -np.arcsin(vec[1]) * 180.0 / np.pi
    return np.array([lo,la],dtype = np.float64)


def decompose_eye_lola(head, gaze):
    # head: longitude&latitude of head posture
    # gaze: direction vector of gaze
    ho = head[0] * np.pi / 180
    ha = head[1] * np.pi / 180
    R = np.zeros((3, 3), dtype=np.float64)
    R[0, 0] = np.cos(ho)
    R[0, 1] = np.sin(ho) * np.sin(ha)
    R[0, 2] = np.sin(ho) * np.cos(ha)

    R[1, 0] = 0
    R[1, 1] = np.cos(ha)
    R[1, 2] = -np.sin(ha)

    R[2, 0] = -np.sin(ho)
    R[2, 1] = np.cos(ho) * np.sin(ha)
    R[2, 2] = np.cos(ho) * np.cos(ha)

    gaze = lola2vec(gaze)
    gaze = np.linalg.inv(R).dot(gaze)

    gaze = gaze / np.linalg.norm(gaze)
    # import pdb;pdb.set_trace()
    eye_lo = np.arcsin(-gaze[0] / np.sqrt(1 - gaze[1] * gaze[1])) / np.pi * 180
    eye_la = np.arcsin(gaze[1]) / np.pi * 180
    return np.array([eye_lo, eye_la], dtype=np.float64)


head_gt = ReadGazeTxt('./data/train/head_label.txt')
eye_gt = ReadGazeTxt('./data/train/eye_label.txt')
gaze_gt = ReadGazeTxt('./data/train/gaze_label.txt')

for name,hgt in head_gt.items():
    ggt = gaze_gt[name]
    egt = eye_gt[name]
    ggt = gaze_gt[name]
    eye_lola = decompose_eye_lola(hgt,ggt)

    print egt


print "done"