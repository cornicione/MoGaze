#coding = utf-8

import os
import cv2
import mxnet as mx
import face_recognition
import numpy as np
from numpy import *
from collections import namedtuple


def get_5checkpoins(path):
    img =face_recognition.load_image_file(path)
    face_location=face_recognition.face_locations(img)
    face_landmarks=face_recognition.face_landmarks(img)
    chin=face_landmarks[0]['chin']
    left_eyebrow=face_landmarks[0]['left_eyebrow']
    right_eyebrow=face_landmarks[0]['right_eyebrow']
    nose_bridge=face_landmarks[0]['nose_bridge']
    nose_tip=face_landmarks[0]['nose_tip']
    left_eye=face_landmarks[0]['left_eye']
    right_eye=face_landmarks[0]['right_eye']
    top_lip=face_landmarks[0]['top_lip']
    bottom_lip=face_landmarks[0]['bottom_lip']
    list_facelandmark=left_eyebrow+right_eyebrow+nose_bridge+nose_tip+top_lip+bottom_lip+left_eye+right_eye
    x_sum=0
    y_sum=0
    for left_eye_point in left_eye:
            x_sum+=left_eye_point[0]
            y_sum+=left_eye_point[1]
    left_eye_center=(int(x_sum/len(left_eye)),int(y_sum/len(left_eye)))
    x_sum=0
    y_sum=0
    for right_eye_point in right_eye:
            x_sum+=right_eye_point[0]
            y_sum+=right_eye_point[1]
    right_eye_center=(int(x_sum/len(right_eye)),int(y_sum/len(right_eye)))

    nose=nose_bridge+nose_tip
    x_sum=0
    y_sum=0
    for nose_point in nose:
            x_sum+=nose_point[0]
            y_sum+=nose_point[1]
    #nose_center=(int(x_sum/len(nose)),int(y_sum/len(nose)))
    lip_left=(int((top_lip[0][0]+bottom_lip[6][0])/2),int((top_lip[0][1]+bottom_lip[6][1])/2))
    lip_right=(int((top_lip[6][0]+bottom_lip[0][0])/2),int((top_lip[6][1]+bottom_lip[0][1])/2))
    points=[]
    points.append(left_eye_center)
    points.append(right_eye_center)
    points.append((int((lip_left[0]+lip_right[0])/2),int((lip_left[1]+lip_right[1])/2)))
    return points,list_facelandmark



def get_AffineMatrix(path):
    crop_size=(200,200)
    benchmark_180=[
        (67,83),
        (133,83),
        (100,144),
       # (72,144),
       # (128,144)
    ]
    check_points,face_landmark=get_5checkpoins(path)
    src_matrix_src=np.zeros((3,2),dtype='float32')
    dst_matrix_src=np.zeros((3,2),dtype='float32')
    i=0
    while i<3:
        src_matrix_src[i][0]=check_points[i][0]
        dst_matrix_src[i][0]=benchmark_180[i][0]/float(200)*200
        src_matrix_src[i][1] = check_points[i][1]
        dst_matrix_src[i][1] = benchmark_180[i][1] / float(200) * 200
        i+=1
    src_matrix=mat(src_matrix_src)
    dst_matrix=mat(dst_matrix_src)

    warp_matrix=cv2.getAffineTransform(src_matrix,dst_matrix)
    return warp_matrix,face_landmark


def get_AffineMatrix_result(img_path):
    affine_matrix, face_landmark=get_AffineMatrix(img_path)
    src_img=cv2.imread(img_path)
    dst_img=cv2.warpAffine(src_img,affine_matrix,(src_img.shape[1], src_img.shape[0]))
    check_points=np.zeros((3,len(face_landmark)),dtype='int')
    index=0
    for points in face_landmark:
        check_points[0][index]=points[0]
        check_points[1][index]=points[1]
        check_points[2][index]=1
        index+=1

    # cou = 0
    # check_points_result = np.dot(affine_matrix, check_points)
    # check_points_int = check_points_result.astype(int)
    # for i in range (0,len(check_points_int[0])):
    #     cou+= 1
    #     cv2.circle(dst_img,(check_points_int[0][i],check_points_int[1][i]), 1, (0, 255, 0), 1)
    #     cv2.putText(dst_img, str(cou),(check_points_int[0][i],check_points_int[1][i]), cv2.FONT_HERSHEY_COMPLEX,
    #                 0.4, (200, 255, 255), 1)

    inverse_affine=cv2.invertAffineTransform(affine_matrix)


    # cou = 0
    # check_points_result=np.row_stack((check_points_result,np.ones((1,len(face_landmark)),dtype='float32')))
    # check_points_source = np.dot(inverse_affine,check_points_result)
    # check_points_source_int = check_points_source.astype(int)
    # for i in range(0, len(check_points_int[0])):
    #     cou += 1
    #     cv2.circle(src_img, (check_points_source_int[0][i], check_points_source_int[1][i]),
    #                1, (255, 255, 0), 1)
    #     cv2.putText(src_img, str(cou), (check_points_source_int[0][i], check_points_source_int[1][i]),
    #                 cv2.FONT_HERSHEY_COMPLEX, 0.4, (200, 255, 255), 1)

    dst_img=dst_img[0:int(200),0:int(200)]
    # cv2.imshow('source',src_img)
    # cv2.imshow('affine',dst_img)
    # cv2.waitKey()

    # file_split=path.split('/')
    # filename_name=(os.path.splitext(file_split[len(file_split)-1]))[0]
    # cv2.imwrite(save_dir + filename_name + '_crop.jpg', dst_img)

    return dst_img,inverse_affine,affine_matrix


def get_data(image):
    image = image.transpose(2, 0, 1)
    image = np.reshape(image, (1, image.shape[0], image.shape[1], image.shape[2]))
    return image

def LoadMxModel(model_prefix,epoch):
    ctx = mx.cpu()
    fcnxs, fcnxs_args, fcnxs_auxs = mx.model.load_checkpoint(model_prefix, epoch)
    fcnxs_args = {k: v.as_in_context(ctx) for k, v in fcnxs_args.items()}
    fcnxs_auxs = {k: v.as_in_context(ctx) for k, v in fcnxs_auxs.items()}
    mod = mx.mod.Module(symbol=fcnxs, context=ctx, label_names=('l2_label',))
    # mx.viz.plot_network(fcnxs).view()
    del fcnxs_args['data']
    del fcnxs_args['l2_label']
    mod.bind(for_training=False,
             data_shapes=[('data', (1, 3, imgSize, imgSize))],
             force_rebind=True)
    mod.set_params(fcnxs_args, fcnxs_auxs, allow_missing=True)

    return mod


count=0
pro_dir = '/Users/momo/Desktop/MoGaze/'
data_dir = pro_dir + 'data/'
train_dir = data_dir + 'train/'
save_dir = train_dir + 'head_crop/'

Batch = namedtuple('Batch', ['data'])
f=open(train_dir+'head/img.txt')
filenames=f.readlines()

imgSize = 128
model_prefix = './resNet50/zrn_landmark87_ResNet50'
mod = LoadMxModel(model_prefix, epoch=9)


for filename in filenames:
    filename = filename.strip()
    count+=1
    if not filename.startswith('.DS_Store'):
        crop_img, inverse_matrix, affine_matrix = get_AffineMatrix_result(filename)

        img = cv2.imread(filename)
        scale_ratio=img.shape[0]*1.0/imgSize
        input_img_1 = cv2.resize(img, (imgSize, imgSize))

        image = get_data(input_img_1)
        mod.forward(Batch([mx.nd.array(image)]))
        output_outer = mod.get_outputs()[0].asnumpy()
        output_outer = output_outer * float(imgSize)
        output_outer = np.reshape(output_outer, (2, -1))
        print output_outer[:10]

        check_points=np.ones((3,87),dtype='float32')
        for i in range(output_outer.shape[1]):
            check_points[0][i]=float(output_outer[0, i]*scale_ratio)
            check_points[1][i]=float(output_outer[1, i]*scale_ratio)
        temp=np.dot(inverse_matrix,check_points)
        temp=temp.astype(int)

        print temp.shape
        print temp[:10]

        i=0
        while i<87:
            cv2.circle(img, (temp[0][i],temp[1][i]), 1,
                       (0, 255, 0), 1)
            i+=1
        cv2.imshow("show",img)
        cv2.waitKey()

    print("one pic done")


print('finished')
