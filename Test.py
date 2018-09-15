# coding: utf-8

import os,cv2,sys
sys.path.append('/Users/momo/caffe/python')
sys.path.append('/Users/momo/Desktop/DATA/')
import caffe
import numpy as np
from util import ReadFile


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

def GetDataForCaffe(img):
    img_caffe = img.transpose((2,0,1))
    img_caffe = img_caffe.reshape((1,img_caffe.shape[0],img_caffe.shape[1],img_caffe.shape[2]))
    return img_caffe

def TestAImage(img,net):
    dst_size = 128
    if img.shape[0] != dst_size or img.shape[1] != dst_size:
        img = cv2.resize(img, (dst_size, dst_size))

    image = GetDataForCaffe(img)
    net.blobs['data'].reshape(*(image.shape))
    forward_kwargs = {'data': image.astype(np.float32)}
    blobs_out = net.forward(**forward_kwargs)

    euler = net.blobs['poselayer'].data[0].tolist()

    return euler

model_dir = '/Users/momo/Desktop/Euler/Model/'
prototxt = model_dir + 'resSmallV2_noBN_Ridge_bigyawpitch_731.prototxt'
caffemodel = prototxt.replace('prototxt','caffemodel')
net = caffe.Net(prototxt, caffemodel, caffe.TEST)

imgSize = 128
train_names = ReadFile('/Users/momo/Desktop/MoGaze/data/train/head/img.txt')
train_names = np.random.permutation(train_names)

'''
#test_names = ReadFile('/Users/momo/Desktop/MoGaze/data/train/head/img.txt')

f = open('./test_head_lo.txt','w')
for name in train_names:
    ori_img = cv2.imread(name)
    crop_img = ori_img[ori_img.shape[0] // 2 - 200:ori_img.shape[0] // 2 + 200,
               ori_img.shape[1] // 2 - 200:ori_img.shape[1] // 2 + 200]

    euler = TestAImage(crop_img, net)
    euler = np.asarray([euler[1], euler[0]]).astype(np.float32)

    f.writelines(name.split('/')[-1].strip('.png') + '\n')
    f.writelines("{:.3f}".format(euler[0]))

f.close()


'''


#'''
Display = True
gts = ReadGazeTxt('./data/train/eye_label.txt')

loerr = []
laerr = []

for name in train_names:

    # if not int(name.split('/')[-1].split('.png')[0])>60000:
    #     continue
    print name
    l_img = cv2.imread(name.replace('head','l_eye'))
    r_img = cv2.imread(name.replace('head', 'r_eye'))

    ori_img = l_img if np.random.rand()>0.5 else r_img
    if ori_img is None:
        continue
    prefix = name.split('/')[-1]
    head_gt = gts[prefix]

    eye_image_center = [ori_img.shape[0]//2,ori_img.shape[1]//2]
    crop_half_w = 30
    crop_img = ori_img[:ori_img.shape[0],
               max(eye_image_center[1] - crop_half_w,0):min(eye_image_center[1] + crop_half_w,ori_img.shape[1])]

    resize_img = cv2.resize(crop_img, (224, 224))
## **** face ****
    # crop_img = ori_img[ori_img.shape[0] // 2 - 200:ori_img.shape[0] // 2 + 200,
    #            ori_img.shape[1] // 2 - 200:ori_img.shape[1] // 2 + 200]

    # euler = TestAImage(crop_img,net)
    # euler = np.asarray([euler[1],euler[0]]).astype(np.float32)
    #
    # loerr.append(head_gt[0] - euler[0])
    # laerr.append(head_gt[1] - euler[1])


    if Display:
        # cv2.putText(resize_img, name.split('/')[-1],
        #             (10, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 255), 1)
        # cv2.putText(crop_img, "pre {:.3f} {:.3f}".format(euler[0],euler[1]),
        #             (20, 70), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)
        cv2.putText(resize_img, "gt {:.3f}".format(head_gt[0]),
                    (10, 40), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)
        cv2.putText(resize_img, "{:.3f}".format(head_gt[1]),
                    (10, 60), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)
#        cv2.imshow("result", resize_img)
        cv2.imshow(name,resize_img)
        cv2.waitKey()

#    print "name {} with euler {}".format(name.split('/')[-1], euler)

print np.mean(loerr)
print np.mean(laerr)


# 训练集上平均误差la:2度,lo:23度
print "done"

#'''