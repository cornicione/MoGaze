#coding=utf-8

import sys
sys.path.append('/Users/momo/Desktop/DATA/')
import numpy as np
import cv2
import h5py
import scipy.io as sio
from util import ReadFile
from math import atan2,degrees,radians


'''
    STEP: 1.Gen_txt() : 先生成txt
          2.根据txt生成hdf5
'''


def RestTest():

    l_2w = ReadFile(data_path + 'train_2w.txt')
    l_all = ReadFile(data_path + 'list.txt')

    te_out = open(data_path + 'test_rest.txt', 'w')

    tr_imgs_path = []
    te_imgs_path = []

    for ind, line in enumerate(l_2w):
        tr_imgs_path.append(line.split(' ')[0].split('/')[-1].split('.')[0])
    print str(len(tr_imgs_path))

    for l in l_all:
        if l+'_128' not in tr_imgs_path:
            im_path = img_path + l + '_128.jpg'
            p3_path = para_path + l + '_pose.mat'

            Euler = sio.loadmat(p3_path)['Euler_Para'][0]
            pitch = degrees(Euler[0])
            yaw = degrees(Euler[1])
            roll = degrees(Euler[2])
            roll = roll - align_degree[prefix.index(l)] - disturb_degree[prefix.index(l)]

            te_imgs_path.append(l)
            te_out.writelines(im_path + ' ' + str(pitch) + ' ' + str(yaw) + ' ' + str(roll))
            te_out.write('\n')

    print str(len(te_imgs_path))

    te_out.close()



def Gen_txt():
    '''
        生成Euler回归txt的list
    '''

    pre_list = ReadFile(data_path + 'list_96.txt')

    tr_num = 36000
    te_num = 1000

    tr_out = open(data_path + '96_train_%d.txt' %(tr_num), 'w')
    te_out = open(data_path + '96_test_%d.txt' %(te_num), 'w')
    re_out = open(data_path + '96_test_rest.txt', 'w')

    tr_start_ind = np.random.randint(0, len(pre_list) - tr_num - te_num)

    new_ind = np.random.permutation(len(pre_list))
    tr_ind = new_ind[tr_start_ind: tr_start_ind+tr_num]
    te_ind = new_ind[tr_start_ind+tr_num: tr_start_ind+tr_num+te_num]


    for ti in tr_ind:
        p = pre_list[ti]
        im_path = img_path + p + '_128.jpg'
        p3_path = para_path + p + '_pose.mat'

        Euler = sio.loadmat(p3_path)['Euler_Para'][0]
        pitch = degrees(Euler[0])
        yaw = degrees(Euler[1])
        roll = degrees(Euler[2])

# ***** validation roll *********
#         img = cv2.imread(im_path.replace('/input_96/', '/img/').replace('_128.jpg', '.jpg'))
#         cv2.imshow("ori_img", img)
#
#         small_img_ro = cv2.imread(im_path)
#
#         M = cv2.getRotationMatrix2D((img.shape[1] / 2, img.shape[0] / 2), roll, 1)
#         big_img_ro = cv2.warpAffine(img, M, (img.shape[0], img.shape[1]))
#
#         cv2.putText(small_img_ro, str(roll), (40, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
#                     (200, 20, 200), 1)
#
#         cv2.imshow("ori_img_ro", big_img_ro)
#         cv2.imshow("small_img_ro", small_img_ro)
#         cv2.waitKey()
# *********************************
        tr_out.writelines(im_path + ' ' + str(pitch) + ' ' + str(yaw) + ' ' + str(roll))
        tr_out.write('\n')
    tr_out.close()

    for te in te_ind:
        p = pre_list[te]
        im_path = img_path + p + '_128.jpg'
        p3_path = para_path + p + '_pose.mat'

        Euler = sio.loadmat(p3_path)['Euler_Para'][0]
        pitch = degrees(Euler[0])
        yaw = degrees(Euler[1])
        roll = degrees(Euler[2])

        te_out.writelines(im_path + ' ' + str(pitch) + ' ' + str(yaw) + ' ' + str(roll))
        te_out.write('\n')
    te_out.close()

    cou = 0
    for s in new_ind:
        if s not in tr_ind and s not in te_ind:
            p = pre_list[s]
            im_path = img_path + p + '_128.jpg'
            p3_path = para_path + p + '_pose.mat'

            Euler = sio.loadmat(p3_path)['Euler_Para'][0]
            pitch = degrees(Euler[0])
            yaw = degrees(Euler[1])
            roll = degrees(Euler[2])

            re_out.writelines(im_path + ' ' + str(pitch) + ' ' + str(yaw) + ' ' + str(roll))
            re_out.write('\n')
        else:
            cou +=1
    re_out.close()
    print " the rest number of samples : " + str(len(pre_list)-cou)
    print " = " + str(len(pre_list)-len(tr_ind)-len(te_ind))



def WriteHdf5(setname):

    IMAGE_SIZE = (128, 128)
    MEAN_VALUE = 0

    filename = data_path + setname + '.txt'

    lines = ReadFile(filename)
    np.random.shuffle(lines)

    sample_size = len(lines)
    imgs = np.zeros((sample_size, 3,) + IMAGE_SIZE, dtype=np.float32)
    poses = np.zeros((sample_size,3), dtype=np.float32)

    h5_filename = '{}.h5'.format(setname)
    with h5py.File(h5_filename, 'w') as h:
        for i, line in enumerate(lines):
            image_name, pitch, yaw, roll = line.split()
            img = cv2.imread(image_name).astype(np.float32)
            img = img.transpose(2,0,1)

            img = img.reshape((1,)+img.shape)
            img -= MEAN_VALUE
            imgs[i] = img
            poses[i] = [float(pitch), float(yaw), float(roll)]
            if (i+1) % 1000 == 0:
                print('processed {} images!'.format(i+1))
        h.create_dataset('data', data=imgs)
        h.create_dataset('pose', data=poses)

    print "save h5 file"
    with open('{}_h5.txt'.format(setname), 'w') as f:
        f.write(h5_filename)


def ReadHdf5(h5_name):

    h5_path = data_path + h5_name
    h = h5py.File(h5_path, 'r')
    h.keys()
    imgs = h['data'][:]
    poses = h['pose'][:]
    h.close()
    return imgs,poses


def SplitTxt():

    lines = ReadFile(data_path + 'test_3w6.txt')
    new_ind = np.random.permutation(len(lines)).tolist()
    new_lines = []
    for ind in new_ind:
        new_lines.append(lines[ind])
    tr_num = 20000
    tr_lines = new_lines[:tr_num]
    te_lines = new_lines[tr_num:]

    out_tr = open(data_path + 'train_rest_2w.txt','w')
    out_te = open(data_path + 'test_rest_1w6.txt','w')

    for tr in tr_lines:
        out_tr.writelines(tr)
        out_tr.write('\n')
    out_tr.close()
    for te in te_lines:
        out_te.writelines(te)
        out_te.write('\n')
    out_te.close()



data_path = '/Users/momo/Desktop/Euler/Data/'
img_path = data_path + 'input_96/'
para_path = data_path + 'pose_para/'



#WriteHdf5('train_rest_2w')
#ReadHdf5('test_1000.h5')
