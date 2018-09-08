#coding=utf8

import sys, os
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
import numpy as np
import mxnet as mx
import cv2
from collections import namedtuple
Batch = namedtuple('Batch', ['data'])
epoch = 9
imgSize = 128
model_prefix = './resNet50/zrn_landmark87_ResNet50'
ctx = mx.cpu()
path='./crop_img/image.txt'
# path_inverse = './checkpoints/inverse.txt'


def get_data(image):
    image = image.transpose(2, 0, 1)
    image = np.reshape(image, (1, image.shape[0], image.shape[1], image.shape[2]))
    return image

def main_Image(path,model_prefix,epoch):
    fcnxs, fcnxs_args, fcnxs_auxs = mx.model.load_checkpoint(model_prefix, epoch)
    fcnxs_args = {k: v.as_in_context(ctx) for k, v in fcnxs_args.items()}
    fcnxs_auxs = {k: v.as_in_context(ctx) for k, v in fcnxs_auxs.items()}
    mod = mx.mod.Module(symbol=fcnxs, context=ctx, label_names=('l2_label',))
    mx.viz.plot_network(mod).view()
    del fcnxs_args['data']
    del fcnxs_args['l2_label']
    mod.bind(for_training=False, data_shapes=[('data', (1, 3, imgSize, imgSize))], force_rebind=True)
    mod.set_params(fcnxs_args, fcnxs_auxs, allow_missing=True)
    # lines = os.listdir(path)
    f = open(path)
    lines = f.readlines()
    count=0
    for line in lines:
        line = line.strip()
        ful_name=line
        if ful_name.startswith('.DS_Store'):
            continue
        path_inverse = ful_name.replace('crop_img','checkpoints')
        path_inverse = path_inverse.replace('_crop.jpg', '.inverse')
        fr = open(path_inverse)
        liness = fr.readlines()
        img = cv2.imread(ful_name)
        inverse_matrix=np.zeros((2,3),dtype='float32')
        inverse_matrix[0][0] = float(liness[0][0:-1])
        inverse_matrix[0][1] = float(liness[1][0:-1])
        inverse_matrix[0][2] = float(liness[2][0:-1])
        inverse_matrix[1][0] = float(liness[3][0:-1])
        inverse_matrix[1][1] = float(liness[4][0:-1])
        inverse_matrix[1][2] = float(liness[5][0:-1])
        scale_ratio=img.shape[0]*1.0/imgSize
        input_img_1 = cv2.resize(img, (imgSize, imgSize))

        image = get_data(input_img_1)

        mod.forward(Batch([mx.nd.array(image)]))
        output_outer = mod.get_outputs()[0].asnumpy()
        output_outer = output_outer * float(imgSize)
        output_outer = np.reshape(output_outer, (2, -1))
        check_points=np.ones((3,87),dtype='float32')
        for i in range(output_outer.shape[1]):
            check_points[0][i]=float(output_outer[0, i]*scale_ratio)
            check_points[1][i]=float(output_outer[1, i]*scale_ratio)
        temp=np.dot(inverse_matrix,check_points)
        temp=temp.astype(int)
        #show_img=cv2.imread('./source_img/'+os.path.splitext(ful_name)[0]+'.jpg')
        a = os.path.split(ful_name)[1]
        a = a.replace('_crop.jpg','.txt')
        fw=open('../face_data_deal/test_img/'+ a,'w')
        for index in range(0,87):
             fw.write(str(temp[0][index])+'\n')
             fw.write(str(temp[1][index])+'\n')
             i+=1
        fw.close()
        '''
        i=0
        while i<87:
                cv2.circle(show_img, (temp[0][i],temp[1][i]), 1,
                           (0, 255, 0), 1)
                i+=1
        '''
        #cv2.imshow(ful_name, show_img)
        #cv2.waitKey(2000)
        count+=1
       # cv2.imwrite('./result_img/'+os.path.splitext(ful_name)[0]+'.jpg',show_img)
        if count%100==0:
            print('已经处理了'+str(count/len(lines)*100)+'%'+'\n')

# /data/test/faceImg/list.txt model/get_186pt_symbol 27
if __name__ == "__main__":
    main_Image(path=path,model_prefix=model_prefix,epoch=epoch)
