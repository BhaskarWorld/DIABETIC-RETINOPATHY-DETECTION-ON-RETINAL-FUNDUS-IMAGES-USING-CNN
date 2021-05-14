import numpy as np
import cv2
from preprocessinglib import crop_resize,grey_clahe,maxmin
import os
import random




size = (256,256)
offset = 4

clipLimitVal = 8.0
tileGridSizeVal = (8,8)

minVal = 0
maxVal = 1

for i in range(5):
    path = 'H:/project/CategorizedData/class_'+str(i)+'/'
    sav_path = 'H:/project/preprocessed_data/p'+str(i)+'/'
    for img in os.listdir(path):
        if img.endswith('.jpeg'):
            fn,fext = os.path.splitext(img)
            img = cv2.imread(path+img)
            print(fn,"\r")

            imgcr = crop_resize(img,size,offset)
            imggc = grey_clahe(imgcr, clipLimitVal, tileGridSizeVal)
            if(imggc.shape[0]>=256 & imggc.shape[1]>=256):
                imgo = maxmin(imggc,minVal,maxVal)
                cv2.imwrite(sav_path+'{}_pre{}'.format(fn,fext), imgo)

print('done')









##cv2.imshow('d',imgo)
##cv2.imwrite('hello.jpeg',imgo)
##
##imgappend = np.reshape(imgo, (1,np.product(imgo.shape)))[0]
##
##imgsav = np.vstack([imgload,imgappend])
##
##np.save('test3.npy', imgsav)
##print('done')




