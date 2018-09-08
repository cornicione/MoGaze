import os
import cv2
import face_recognition
import numpy as np
from numpy import *
path='./source_img/'
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
    #points.append(lip_left)
    #points.append(lip_right)
    return points,list_facelandmark
"""
img_show=cv2.imread(path)
count=0
for temp_point in points:
    count+=1
    cv2.circle(img_show,temp_point,1,(0,255,0),1)
    cv2.putText(img_show,str(count),temp_point,cv2.FONT_HERSHEY_COMPLEX,0.4,(200, 255, 255),1,cv2.LINE_AA)
cv2.imshow('hha',img_show)
cv2.imwrite('./hah.jpg',img_show)
cv2.waitKey(0)
"""
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
    #print(src_matrix)
    #print(dst_matrix)
    #print(src_matrix_src)
    warp_matrix=cv2.getAffineTransform(src_matrix,dst_matrix)
    return warp_matrix,face_landmark
def get_AffineMatrix_result(path):
    affine_matrix,face_landmark=get_AffineMatrix(path)
    src_img=cv2.imread(path)
    #print(affine_matrix)
    #dst_img=np.copy(src_img)
    #dst_img=mat(dst)
    dst_img=cv2.warpAffine(src_img,affine_matrix,(src_img.shape[1],src_img.shape[0]))
    check_points=np.zeros((3,len(face_landmark)),dtype='int')
    index=0
    for points in face_landmark:
        check_points[0][index]=points[0]
        check_points[1][index]=points[1]
        check_points[2][index]=1
        index+=1
    check_points_result=np.dot(affine_matrix,check_points)
    check_points_int=check_points_result.astype(int)
   # print(check_points_int)
    """
    for i in range (0,len(check_points_int[0])):
        count += 1
        cv2.circle(dst_img,(check_points_int[0][i],check_points_int[1][i]), 1, (0, 255, 0), 1)
        cv2.putText(dst_img,str(count),(check_points_int[0][i],check_points_int[1][i]), cv2.FONT_HERSHEY_COMPLEX, 0.4, (200, 255, 255), 1, cv2.LINE_AA)
    """
    ##inverse_affine=np.zeros((3,2),'float32')
    inverse_affine=cv2.invertAffineTransform(affine_matrix)
    #a=np.zeros((1,len(face_landmark)),dtype='float32')
    check_points_result=np.row_stack((check_points_result,np.ones((1,len(face_landmark)),dtype='float32')))
    check_points_source = np.dot(inverse_affine,check_points_result)
    check_points_source_int = check_points_source.astype(int)
    """
    for i in range(0, len(check_points_int[0])):
        count += 1
        cv2.circle(src_img, (check_points_source_int[0][i], check_points_source_int[1][i]), 1, (0, 255, 0), 1)
        cv2.putText(src_img, str(count), (check_points_source_int[0][i], check_points_source_int[1][i]),
                    cv2.FONT_HERSHEY_COMPLEX, 0.4, (200, 255, 255), 1, cv2.LINE_AA)
    """
    dst_img=dst_img[0:int(200),0:int(200)]
   # cv2.imshow('source',src_img)
   # cv2.imshow('affine',dst_img)

    cv2.waitKey(20)
    file_split=path.split('/')
    filename_name=(os.path.splitext(file_split[len(file_split)-1]))[0]
    cv2.imwrite('./test_img/'+filename_name+'_crop.jpg',dst_img)
    cv2.imwrite('./test_img/' + filename_name + '.jpg', src_img)
    affin_matrix_file=open('./test_img/'+filename_name+'.affine','w')
    inverse_affin_file = open('./test_img/' + filename_name + '.inverse', 'w')
    col=0
    while col<2:
        row=0
        while row<3:
            affin_matrix_file.write(str(affine_matrix[col][row])+'\n')
            row+=1
        col+=1
    affin_matrix_file.close()
    col=0
    while col<2:
        row=0
        while row<3:
            inverse_affin_file.write(str(inverse_affine[col][row])+'\n')
            row+=1
        col+=1
    inverse_affin_file.close()

if __name__=='__main__':
    count=0
    f=open('/Users/momo/Downloads/data_deal/face_data_deal/source_img/img.txt')
    filenames=f.readlines()

    for filename in filenames:
        filename = filename.strip()
        count+=1
        if not filename.startswith('.DS_Store'):
            get_AffineMatrix_result(filename)
        if count%100==0:
            print('已经完成了'+str(float(count/len(filenames)))+'\n')
    print('finished')

