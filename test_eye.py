import face_alignment as fa
import cv2
import math
import numpy as np
import random
import mxnet as mx
import sys
sys.path.append('/Users/momo/caffe/python')
import caffe
import time
from collections import namedtuple


left_eye_left_cornerID = 39
left_eye_right_cornerID = 45
left_eye_startID = 39
left_eye_endID = 50
# dst_size = 128
# dst_size_w = 64
# dst_size_h = 48
# batch_size = 64
# people_num = 64
# PT_NUM = 17
# assert batch_size % people_num == 0
# batch_size_per_people = batch_size / people_num
#
# label_dim = PT_NUM * 2
ctx = mx.cpu()
Batch = namedtuple('Batch', ['data'])
Anet = fa.AlignmentLoadModel()



def get_data(image):
    # image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    # image = np.expand_dims(image, axis=2)
    image = image.transpose(2, 0, 1)
    #image = image / 255.
    image = np.reshape(image, (1, image.shape[0], image.shape[1], image.shape[2]))
    return image

def get_list(test_list_path):
    # test_img = np.zeros((batch_size, 3, dst_size, dst_size), dtype=np.float32)
    # test_label = np.zeros((batch_size, label_dim), dtype=np.float32)

    test_list_file = open(test_list_path)
    test_list = test_list_file.read()
    test_list = test_list.split('\n')
    return test_list

def flip_image_landmarks96(img, landmarks96):
    landmarks_order = [18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,  # face margin
                       34, 33, 32, 31, 30, 29, 38, 37, 36, 35,  # left eyebrow
                       24, 23, 22, 21, 20, 19, 28, 27, 26, 25,  # right eyebrow
                       57, 56, 55, 54, 53, 52, 51, 62, 61, 60, 59, 58,  # lefteye
                       45, 44, 43, 42, 41, 40, 39, 50, 49, 48, 47, 46,  # right eye
                       74, 73, 72, 71, 70, 69, 68, 67, 66, 65, 64, 63, 75,  # nose
                       82, 81, 80, 79, 78, 77, 76, 87, 86, 85, 84, 83, 92, 91, 90, 89, 88, 95, 94, 93]  # mouth
    img = cv2.flip(img, 1)
    swap_label = np.copy(landmarks96)
    swap_label[:, 0] = img.shape[1] - 1 - swap_label[:, 0]
    landmarks_order_num = len(landmarks_order)
    for label_i in range(landmarks_order_num):
        landmarks96[label_i, :] = swap_label[landmarks_order[label_i], :]

    # for j in range(landmarks96.shape[0]):
    #     cv2.circle(img, (landmarks96[j, 0], landmarks96[j, 1]), 1, (255, 0, 0), 1)
    # cv2.imshow('img', img)
    # cv2.waitKey(0)

    return img, landmarks96

def rotate_img_landmarks(img, landmarks, degree_threshold ):


    rot_d = -180 * (math.atan(1.0 * (landmarks[left_eye_left_cornerID][1] - landmarks[left_eye_right_cornerID][1]) / (
        landmarks[left_eye_right_cornerID][0] - landmarks[left_eye_left_cornerID][0]))) / 3.14

    if(abs(rot_d) > degree_threshold):

        im_width = img.shape[1]
        im_height = img.shape[0]
        # M = cv2.getRotationMatrix2D((center[0], center[1]), rot_d, 1)
        M = cv2.getRotationMatrix2D((im_width / 2, im_height / 2), rot_d, 1)

        for j in range(landmarks.shape[0]):
            landmarks[j, :] = np.dot(M, np.array( [landmarks[j, 0], landmarks[j, 1], 1]).transpose()).transpose()


        if landmarks[:, 0].min() >= 0 and landmarks[:, 0].max() < im_width and landmarks[:,1].min() >= 0 and landmarks[:,1].max() < im_height:
            img = cv2.warpAffine(img, M, (im_width, im_height), None, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT, (0, 0, 0))

        #
        # for j in range(landmarks.shape[0]):
        #     cv2.circle(img, (landmarks[j, 0], landmarks[j, 1]), 1, (0, 255, 255), 1)
        # cv2.imshow('img', img)
        # cv2.waitKey(0)

    return img, landmarks

def random_crop(img, part_landmarks, dst_size_w, dst_size_h):

    width_height_ratio = 1.0 * dst_size_w / dst_size_h

    # bounding rect
    [rect_x, rect_y, rect_w, rect_h] = cv2.boundingRect(part_landmarks)

    # print rect_x,rect_y,rect_w,rect_h

    center_x = rect_x + rect_w / 2
    center_y = rect_y + rect_h / 2
    max_w = max(rect_w, rect_h)
    expand_ratio = random.random() * 0.6 + 1.2
    max_w = int(max_w * expand_ratio)
    max_h = max_w / width_height_ratio

    # random crop between two rects of max rect and min rect
    left_bound = rect_x - (center_x - max_w / 2)
    right_bound = (center_x + max_w / 2) - rect_x - rect_w
    top_bound = rect_y - (center_y - max_h / 2)
    bottom_bound = (center_y + max_h / 2) - rect_y - rect_h
    left_bound = int(left_bound * 0.5)
    right_bound = int(right_bound * 0.5)
    top_bound = int(top_bound * 0.5)
    bottom_bound = int(bottom_bound * 0.5)

    new_x = center_x - max_w / 2 + random.randint(-right_bound, left_bound)
    new_y = center_y - max_h / 2 + random.randint(-bottom_bound, top_bound)

    # add margin for out-of-bounds rect
    if new_x < 0:
        inc_left = -new_x
    else:
        inc_left = 0

    if new_y < 0:
        inc_top = -new_y
    else:
        inc_top = 0

    if new_x + max_w > img.shape[1]:
        inc_right = new_x + max_w - img.shape[1]
    else:
        inc_right = 0

    if new_y + max_h > img.shape[0]:
        inc_bottom = new_y + max_h - img.shape[0]
    else:
        inc_bottom = 0

    img = cv2.copyMakeBorder(img, inc_top, inc_bottom, inc_left, inc_right, cv2.BORDER_CONSTANT)

    # correct new_x new_y minus label
    if new_x < 0:
        new_x = 0
    if new_y < 0:
        new_y = 0

    # correct all label
    new_label = np.copy(part_landmarks)
    new_label[:, 0] = new_label[:, 0] + inc_left
    new_label[:, 1] = new_label[:, 1] + inc_top

    # crop image
    img = img[new_y:new_y + max_h, new_x:new_x + max_w, :]
    scale_ratio_x = float(dst_size_w) / max_w
    scale_ratio_y = float(dst_size_h) / max_h

    # resize image and correct landmarks
    img = cv2.resize(img, (dst_size_w, dst_size_h))
    new_label[:, 0] = (new_label[:, 0] - new_x) * scale_ratio_x
    new_label[:, 1] = (new_label[:, 1] - new_y) * scale_ratio_y

    # for j in range(new_label.shape[0]):
    #     cv2.circle(img, (new_label[j, 0], new_label[j, 1]), 1, (255, 0, 0), 1)
    # cv2.imshow('img', img)
    # cv2.waitKey(0)

    return img, new_label


def get_crop_img(img_ori, width, height):
    # input original image
    # output crop reseized image

    img = img_ori
    label = []
    landmarks_96 = fa.alignment(img, Anet)
    landmarks_96 = landmarks_96[0]
    # for i in range(0,len(landmarks_96)/2):
    #     cv2.circle(img, (int(landmarks_96[i] * img.shape[1]), int(landmarks_96[i + 96] * img.shape[0])), 1, (0, 255, 255))
    # cv2.imshow('ori', img)
    # cv2.waitKey(0)

    # change label format
    for m in range(0, len(landmarks_96) / 2):
        label.append([landmarks_96[m], landmarks_96[m + len(landmarks_96) / 2]])

    # from normalized label to img size label
    for l in range(len(label)):
        label[l][0] = label[l][0] * img.shape[1]
        label[l][0] = np.float32(label[l][0])
        label[l][1] = label[l][1] * img.shape[0]
        label[l][1] = np.float32(label[l][1])

    label = np.array(label)

    # for left and right eye
    for i in range(0, 2):
        if i == 1:
            # left eye is 0, right eye is 1
            # for right eye, flip img and landmarks
            img, label = flip_image_landmarks96(img, label)

        # rotate img
        img, label = rotate_img_landmarks(img, label, 5)

        # bounding rect
        eye_lable = label[left_eye_startID:left_eye_endID + 1, :]

        img, label = random_crop(img, eye_lable, width, height)
        return  img




def load_model(model_prefix,epoch):
    fcnxs, fcnxs_args, fcnxs_auxs = mx.model.load_checkpoint(model_prefix, epoch)
    fcnxs_args = {k: v.as_in_context(ctx) for k, v in fcnxs_args.items()}
    fcnxs_auxs = {k: v.as_in_context(ctx) for k, v in fcnxs_auxs.items()}
    mod = mx.mod.Module(symbol=fcnxs, context=ctx, label_names=('l2_label',))
    del fcnxs_args['data']
    del fcnxs_args['l2_label']
    mod.bind(for_training=False, data_shapes=[('data', (1, 3, 128, 128))], force_rebind=True)
    mod.set_params(fcnxs_args, fcnxs_auxs, allow_missing=True)
    return mod



def test(mod, test_list_path, save_path):
    width = 128
    height = 128

    test_list = get_list(test_list_path)
    for i in range(0, len(test_list)):
        # img_path = '/Users/fuyu/Desktop/test2.jpg'
        # img_ori = cv2.imread(img_path)

        img_ori = cv2.imread(test_list[i])
        img_resize = cv2.resize(img_ori, (width, height))
        img_crop = get_crop_img(img_resize, width, height)
        # cv2.imshow('a', img_crop)
        # cv2.waitKey(0)
        img_test = get_data(img_crop)


        # scale_ratio_x = img_crop.shape[0] * 1.0 / width
        # scale_ratio_y = img_crop.shape[1] * 1.0 / height

        start_time = time.time()
        mod.forward(Batch([mx.nd.array(img_test)]))
        print("--- %s seconds ---" % (time.time() - start_time))

        output_outer = mod.get_outputs()[0].asnumpy()

        output_outer = np.reshape(output_outer, (2, -1))

        for j in range(output_outer.shape[1]):
            cv2.circle(img_crop, (int(output_outer[0, j] * width), int(output_outer[1, j] * height)), 1, (0, 255, 0), 2)
        #
        # cv2.imshow('img', img_crop)
        # cv2.waitKey(0)
        # print i

        save_name = save_path + str(i) + ".jpg"
        cv2.imwrite(save_name, img_crop)

def compare_caffe_mxnet():
    caffe_prototxt = '/Users/fuyu/work/Beauty/186points/face_alignment/model/get_186pt_symbol_eye.prototxt'
    caffe_model = '/Users/fuyu/work/Beauty/186points/face_alignment/model/get_186pt_symbol_eye-0396.caffemodel'
    ANet = caffe.Net(caffe_prototxt, caffe_model, caffe.TEST)
    mod = load_model('/Users/fuyu/work/Beauty/186points/face_alignment/model/get_186pt_symbol_eye', 396)
    test_list_path = '/Users/fuyu/work/Beauty/186points/face_alignment/TestImgList.txt'

    test_list = get_list(test_list_path)
    for i in range(0, len(test_list)):
        img_ori = cv2.imread(test_list[i])
        img_crop = get_crop_img(img_ori)
        img_test = get_data(img_crop)
        scale_ratio = img_crop.shape[0] * 1.0 / 128
        mod.forward(Batch([mx.nd.array(img_test)]))
        output_outer = mod.get_outputs()[0].asnumpy()
        output_outer = output_outer * float(128)
        landmarks1 = np.reshape(output_outer, (2, -1))

        ANet.blobs['data'].data[...] = img_test
        out = ANet.forward()
        landmarks2 = out['fullyconnected1']
        landmarks2 *= float(128)








if __name__ == '__main__':

    # mod = load_model(sys.arglsv[1],sys.argv[2])
    # test(mod, sys.argv[3], sys.argv[4])

    mod = load_model('/Users/fuyu/work/Beauty/186points/face_alignment/model/eye/128*128/get_186pt_symbol_eye', 396)
    file_list_path = '/Users/fuyu/work/Beauty/186points/FileList.txt'
    save_img_path = '/Users/fuyu/work/Beauty/186points/face_alignment/test_result/0314_128*128_two_eye3/'
    test(mod, file_list_path, save_img_path )
    # test_64_48(mod, '/Users/fuyu/Desktop/test.txt', '/Users/fuyu/work/Beauty/186points/face_alignment/test_result/0312/' )

    # compare_caffe_mxnet()


