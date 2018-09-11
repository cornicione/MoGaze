#coding = utf-8

import os
import cv2
import mxnet as mx
import numpy as np
from numpy import *
from collections import namedtuple

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.externals import joblib


def ReadFile(file_path, remove_pattern='\n'):
    '''
    :param remove_pattern: if Windows, pattern = '\r'
    '''
    with open(file_path) as f:
        lines = f.readlines()
    f.close()

    line = []
    for l in lines:
        line.append(l.split(remove_pattern)[0])
    lines = line

    return lines


def get_data(image):
    image = image.transpose(2, 0, 1)
    image = np.reshape(image, (1, image.shape[0], image.shape[1], image.shape[2]))
    return image

def LoadMxModel(model_prefix,epoch):

    fcnxs, fcnxs_args, fcnxs_auxs = mx.model.load_checkpoint(model_prefix, epoch)
    fcnxs_args = {k: v.as_in_context(ctx) for k, v in fcnxs_args.items()}
    fcnxs_auxs = {k: v.as_in_context(ctx) for k, v in fcnxs_auxs.items()}
    mod = mx.mod.Module(symbol=fcnxs, context=ctx, label_names=('l2_label',))
#    mx.viz.plot_network(fcnxs).view()
    del fcnxs_args['data']
    del fcnxs_args['l2_label']
    mod.bind(for_training=False,
             data_shapes=[('data', (1, 3, imgSize, imgSize))],
             force_rebind=True)
    mod.set_params(fcnxs_args, fcnxs_auxs)


    return mod

def LoadFeaExtractor(model_prefix,epoch):

    fcnxs, fcnxs_args, fcnxs_auxs = mx.model.load_checkpoint(model_prefix, epoch)
    fcnxs_args = {k: v.as_in_context(ctx) for k, v in fcnxs_args.items()}
    fcnxs_auxs = {k: v.as_in_context(ctx) for k, v in fcnxs_auxs.items()}
    mod = mx.mod.Module(symbol=fcnxs, context=ctx, label_names=('l2_label',))
    #    mx.viz.plot_network(fcnxs).view()
    del fcnxs_args['data']
    del fcnxs_args['l2_label']
    mod.bind(for_training=False,
             data_shapes=[('data', (1, 3, imgSize, imgSize))],
             force_rebind=True)
    mod.set_params(fcnxs_args, fcnxs_auxs, allow_missing=True)

    internals = mod.symbol.get_internals()
    print(internals.list_outputs())
    fea_symbol = internals['_plus23_output']

    feature_extractor = mx.mod.Module(symbol=fea_symbol,
                                      context=ctx,
                                      label_names=None)
    feature_extractor.bind(for_training=False,
                           data_shapes=[('data', (1, 3, imgSize, imgSize))])

    feature_extractor.set_params(fcnxs_args,
                                 fcnxs_auxs)

    return feature_extractor

def GetFeaandGt(train_names, model, Gt, start_index, extract_num):

    if start_index + extract_num > len(train_names):
        extract_num = len(train_names)-start_index
        print("[Get fea] out of the length of train_names")

    features = np.zeros((extract_num, 512))
    labels = np.zeros((extract_num, 2))

    for ind, name in enumerate(train_names[start_index:start_index + extract_num]):
        ori_img = cv2.imread(name)  # filename : original image (~900)
        crop_img = ori_img[ori_img.shape[0] // 2 - 200:ori_img.shape[0] // 2 + 200,
                   ori_img.shape[1] // 2 - 200:ori_img.shape[1] // 2 + 200,
                   ...]
        input_img_1 = cv2.resize(crop_img, (imgSize, imgSize))
        image = get_data(input_img_1)

        model.forward(Batch([mx.nd.array(image)]))
        feature = mod.get_outputs()[0].asnumpy()
        feature = feature.flatten()

        features[ind, :] = feature
        labels[ind,:] = Gt[name.split('/')[-1]]

        print("to {} pic and name is {}".format(ind, name))

    return features,labels


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




data_dir = "/data/mc_data/MC4/"
train_dir = data_dir + 'train/'
# test_dir = data_dir + 'test/'

# clf_model_dir = "/home/momenta/Desktop/ckpt/HeadRidge/"
# name_dir = '/home/mc/Desktop/'
# all_names = ReadFile(name_dir + 'train_names.txt')
# head_gt = ReadGazeTxt(data_dir + "head_label.txt")

# all_names = ReadFile('/Users/momo/Desktop/MoGaze/data/train/head/img.txt')
# head_gt = ReadGazeTxt('/Users/momo/Desktop/MoGaze/data/train/head_label.txt')

ShuffleName = True
Batch = namedtuple('Batch', ['data'])
imgSize = 128
ctx = mx.cpu()
mod = LoadFeaExtractor('./resNet50/zrn_landmark87_ResNet50',9)

# if ShuffleName:
#     np.random.shuffle(all_names)

valid_num = 5000
valid_names = all_names[:valid_num]
train_names = all_names[valid_num:]

train_features, train_labels = GetFeaandGt(train_names, mod, head_gt, 0, 10000)
test_features, test_labels = GetFeaandGt(valid_names, mod, head_gt, 0, 5000)

clf = Ridge(alpha=2.0)
clf.fit(train_features, train_labels)

pre_test_labels = clf.predict(test_features)
clf_mae = mean_absolute_error(pre_test_labels, test_labels, multioutput='raw_values')

pre_train_labels = clf.predict(train_features)
clf_mae_tr = mean_absolute_error(pre_train_labels, clf.predict(train_features),
                                 multioutput='raw_values')

print("clf valid mae: {}".format(clf_mae))
print("clf train mae: {}".format(clf_mae_tr))


save_path = clf_model_dir + 'HeadPose_tr{}.model'.format(train_features.shape[0])
joblib.dump(clf, save_path)

print('finished')
