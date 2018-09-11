import sys
sys.path.append("/Users/momo/caffe/python")
import caffe
import os
from sklearn.externals import joblib


def ComparePara(net_1, net_2):
    for k in net_1.params:
        if k in net_2.params:
            weight_1 = net_1.params[k][0].data.flatten()
            bias_1 = net_1.params[k][1].data.flatten()
            weight_2 = net_2.params[k][0].data.flatten()
            bias_2 = net_2.params[k][1].data.flatten()

            for w in range(weight_2.shape[0]):
                if not (weight_1[w] == weight_2[w]):
                    break
            print k + ' weight not same!'

            for b in range(bias_2.shape[0]):
                if not (bias_1[b] == bias_2[b]):
                    break
            print k + ' bias not same!'
        else:
            print k + " is not in dst net!"
        print k + " done!"

    print "Compare Para done!"


pro_dir = '/Users/momo/Desktop/Euler/'
model_dir = pro_dir + 'model/'


old_prototxt = model_dir + 'stable_facealignment_v1_new.prototxt'
old_caffemodel = old_prototxt.replace('.prototxt','.caffemodel')
new_prototxt = model_dir + 'stable_facealignment_v1_euler_816.prototxt'
new_caffemodel = new_prototxt.replace('.prototxt','.caffemodel')
net_old = caffe.Net(old_prototxt, old_caffemodel, caffe.TEST)
net_new = caffe.Net(new_prototxt, caffe.TEST)


#'''
# function: copy from clf
clf = joblib.load(model_dir + '96pt_ori2w_pitch4k_yaw1w5_v1.model')
w = clf.coef_
b = clf.intercept_
for k in net_new.params:
    if k in net_old.params and k != 'poselayer':
        for i in range(len(net_old.params[k])):
            net_new.params[k][i].data[...] = net_old.params[k][i].data
        print 'copy from '+ k

print net_new.params['poselayer'][0].data.shape, net_new.params['poselayer'][1].data.shape
net_new.params['poselayer'][0].data[...] = w
net_new.params['poselayer'][1].data[...] = b
net_new.save(new_caffemodel)
#'''



'''
# function : copy multi model


for root,dirs,files in os.walk(model_dir):
    for file in files:
        new_caffemodel = model_dir + file
        dst_caffemodel = new_caffemodel
        net_new = caffe.Net(dst_prototxt, new_caffemodel, caffe.TEST)
        print new_caffemodel
# ********** for two more nets **************
        for k in net_new.params:
            if k in net_ori.params:
                for i in range(len(net_ori.params[k])):
                    net_new.params[k][i].data[...] = net_ori.params[k][i].data
                    print 'copy from ori net '+ k
            elif k in net_new.params:
                for i in range(len(net_new.params[k])):
                    net_new.params[k][i].data[...] = net_new.params[k][i].data
                    print 'copy from new net '+ k
# ********************************************
        net_new.save(dst_caffemodel)
'''

# for k in net_new.params:
#     if k in net_lr.params:
#         for i in range(len(net_lr.params[k])):
#             net_new.params[k][i].data[...] = net_lr.params[k][i].data
#         print 'copy from '+ k
#     else:
#         for i in range(len(net_pose.params[k])):
#             net_new.params[k][i].data[...] = 0
#         print 'do not have this layer : ' + k
# net_new.save(dst_caffemodel)

