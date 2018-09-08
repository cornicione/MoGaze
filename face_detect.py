import face_recognition        #用于人脸识别
import os
from PIL import Image
from get_path_file import get_absolute_path
import shutil
def get_checkpointsAndcrop(file_list):
    scale =1.7
    count_success=0
    count_fail=0
    sum = len(file_list)
    count = 0
    for filename in file_list:
        if not filename.startswith('.DS_Store'):
            img = face_recognition.load_image_file(filename)   #读入图片
            h,w,channel=img.shape
            face_locations = face_recognition.face_locations(img)
            if len(face_locations)<=0 or len(face_locations)>=2:
                print('没有检测到人脸,越过')
                os.remove(filename)
                count+=1
                continue
            name_path=filename.split('/')
            name=name_path[len(name_path)-1]
            name=os.path.splitext(name)[0]
            f=open('./keypointx/'+name+'.txt','w')
            top,right,bottom,left=face_locations[0]
            new_w = right - left
            new_h = bottom - top
            top=top*0.9
            right=right
            x_middle = int((left+right)/2)
            y_middle = int((bottom + top)/2)
            if new_w>new_h:
                x1 = int(x_middle -int( new_w*scale/2))
                x2 = int(x_middle + int(new_w*scale /2))
                y1 = int(y_middle - int(new_w*scale/2))
                y2 = int(y_middle + int(new_w *scale/2))
                if x1>=0 and y1>=0  and x2<=w and y2<=h:
                   count_success+=1
                   #face_image = img[y1:y2, x1:x2]
                  # pil_image = Image.fromarray(face_image)
                 #  pil_image.save('./crop_img/'+name+'.jpg')
                else:
                    if x1<0 and y1>=0 and y2<=h :
                        x1 = 0
                        x2 -= x1
                        count_success += 1
                    #    face_image = img[y1:y2, x1:x2]
                     #   pil_image = Image.fromarray(face_image)
                     #   pil_image.save('./crop_img/'+name+'.jpg')
                  #      print(face_locations)
                    else:
                        if x2>w and y1>=0 and y2<=h:
                            x2=w
                            x1 -= (w - x2)
                            count_success += 1
                      #      face_image = img[y1:y2, x1:x2]
                       #     pil_image = Image.fromarray(face_image)
                       #     pil_image.save('./crop_img/'+name+'.jpg')
                       #     print(face_locations)
                        else:
                            os.remove(filename)
                            count_fail+=1
            else:
                x1 = int(x_middle - new_h * scale / 2)
                x2 = int(x_middle + new_h * scale / 2)
                y1 = int(y_middle - new_h * scale / 2)
                y2 = int(y_middle + new_h * scale / 2)
                if x1 >= 0 and y1 >= 0  and x2 <= w and y2 <=h:
                    count_success += 1
                   # face_image = img[y1:y2, x1:x2]
                   # pil_image = Image.fromarray(face_image)
                   # pil_image.save('./crop_img/'+name+'.jpg')
                   # print(face_locations)
                else:
                    if x1<0 and y1>=0 and y2<=h:
                        x1=0
                        x2-=x1
                        count_success += 1
                    #    face_image = img[y1:y2, x1:x2]
                     #   pil_image = Image.fromarray(face_image)
                     #   pil_image.save('./crop_img/'+name+'.jpg')
                     #   print(face_locations)
                    else:
                        if x2>w and y1>=0 and y2<=h:
                            x1 -= (w-x2)
                            x2 = w
                            count_success += 1
                       #     face_image = img[y1:y2, x1:x2]
                       #     pil_image = Image.fromarray(face_image)
                        #    pil_image.save('./crop_img/'+name+'.jpg')
                        #    print(face_locations)
                        else:
                            os.remove(filename)
                            count_fail += 1
            pos=[x1,y1,x2,y2]
            for pos_temp in pos:
                f.write(str(pos_temp)+'\n')
            f.close()
            count+=1
            if(count%1000==0):
                print('已经处理了'+str(float(count/sum * 100))+'%的图片\n')
def copy_img(file_list):
    count=50939;
    for filename in file_list:
        shutil.move(filename,'./img/'+str(count)+'.jpg')
        count+=1;
if __name__=='__main__':
    path='./img/'
    file_list=get_absolute_path(path)
    #copy_img(file_list=file_list)
    get_checkpointsAndcrop(file_list=file_list)
    print('finished')