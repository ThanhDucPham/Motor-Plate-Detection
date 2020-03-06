import cv2
import numpy as np
import random
import os
import glob

root = './image/charTrainset'
labels = os.listdir(root)

train_images = []

for label in labels[:2] :
    for i,path in enumerate(glob.glob('./image/charTrainset/'+label+'/*')) :
        for j in range(10) :
            img = cv2.imread(path)
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _,img_thresh = cv2.threshold(img_gray,0,255,cv2.THRESH_OTSU)
            rows,cols = img_thresh.shape
            degre = random.uniform(-10,10)
            M = cv2.getRotationMatrix2D((cols/2,rows/2),degre,1)
            img_train = cv2.warpAffine(img_thresh,M,(cols,rows))
            # train_images.append(img_train)
            cv2.imwrite('./image/charTrainset/'+label+'/'+'img_'+str(i)+'_'+str(j)+'.jpg', img_train)


# print(img.shape)


# cv2.imshow('image', img_train)
# cv2.waitKey(0)