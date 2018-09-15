import cv2,sys,os



names = [i for i in os.listdir('./data/train/l_eye/') if i.endswith('.png')]

for n in names:
    img=cv2.imread('./data/train/l_eye/'+n)

    mid_y,mid_x = img.shape[0]//2, img.shape[1]//2
    h = img.shape[0]
    w = img.shape[1]

    crop_img = img[0:int(0.8*h), int(0.1*w):int(0.9*w),...]

    cv2.imshow("crop",crop_img)
    cv2.waitKey()

