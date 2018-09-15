
import cv2,os
import math
import numpy as np
import mxnet as mx
import sys
sys.path.append('/Users/momo/caffe/python')
from collections import namedtuple


ctx = mx.cpu()
Batch = namedtuple('Batch', ['data'])


def get_data(image):
    image = image.transpose(2, 0, 1)
    image = np.reshape(image, (1, image.shape[0], image.shape[1], image.shape[2]))
    return image


def load_model(model_prefix,epoch):
    fcnxs, fcnxs_args, fcnxs_auxs = mx.model.load_checkpoint(model_prefix, epoch)
    fcnxs_args = {k: v.as_in_context(ctx) for k, v in fcnxs_args.items()}
    fcnxs_auxs = {k: v.as_in_context(ctx) for k, v in fcnxs_auxs.items()}
    mod = mx.mod.Module(symbol=fcnxs, context=ctx, label_names=('l2_label',))
    del fcnxs_args['data']
    del fcnxs_args['l2_label']
    mod.bind(for_training=False,
             data_shapes=[('data', (1, 3, 128, 128))],
             force_rebind=True)
    mod.set_params(fcnxs_args, fcnxs_auxs,allow_missing=True)
    return mod



def test(mod, test_names):

    for i in range(0, len(test_names)):
        ori_img = cv2.imread(head_path + test_names[i])

        # crop_half_w = 24
        # crop_half_h = 32
        # crop_img = ori_img[ori_img.shape[0] // 2 - crop_half_h :ori_img.shape[0] // 2 + crop_half_h,
        #            ori_img.shape[1] // 2 - crop_half_w :ori_img.shape[1] // 2 + crop_half_w]
        crop_half_w = 60
        crop_img = ori_img[:ori_img.shape[0],
                            ori_img.shape[1] // 2 - crop_half_w :ori_img.shape[1] // 2 + crop_half_w]
        print crop_img.shape
        cv2.imshow("eye", crop_img)

        img_resize = cv2.resize(crop_img, (224, 224))
        cv2.imshow("eye resize",img_resize)
        cv2.waitKey()

        img_test = get_data(img_resize)

        mod.forward(Batch([mx.nd.array(img_test)]))
        output_outer = mod.get_outputs()[0].asnumpy()

        output_outer = output_outer * float(128)
        output_outer = np.reshape(output_outer, (2, -1))

        for j in range(output_outer.shape[1]):
            cv2.circle(img_resize, (int(output_outer[0, j]),
                                    int(output_outer[1, j])), 1, (0, 255, 0), 2)

        cv2.imshow('img', img_resize)
        cv2.waitKey(0)
        print i



width = 128
height = 128

#mod = load_model('/Users/momo/Desktop/FaceAlignment137/Model/fuyu_V1_in137', 1600)
eye_path = '/Users/momo/Desktop/MoGaze/data/train/l_eye/'
train_names = [n for n in os.listdir(eye_path) if n.endswith('.png')]

for t in train_names:
    ori_img = cv2.imread(eye_path + t)

    eye_image_center = [ori_img.shape[0]//2,ori_img.shape[1]//2]
    crop_half_w = 60
    crop_img = ori_img[:ori_img.shape[0],
               max(eye_image_center[1] - crop_half_w,0):min(eye_image_center[1] + crop_half_w,ori_img.shape[1])]

    cv2.imshow("eye", crop_img)

    img_resize = cv2.resize(crop_img, (224, 224))
    cv2.imshow("eye resize", img_resize)
    cv2.waitKey()

#test(mod, train_names)