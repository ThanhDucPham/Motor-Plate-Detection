import cv2
import numpy as np
from skimage.feature import hog
# from sklearn.svm import LinearSVC
from sklearn import svm, metrics
from keras.datasets import mnist
from sklearn.metrics import accuracy_score
from skimage import io
import glob
import os
from mnist import MNIST
from joblib import dump, load

train_images = []
train_labels = []

root = './image/charTrainset'
labels = os.listdir(root)

for label in labels :
    for path in glob.glob('./image/charTrainset/'+label+'/*') :
        train_labels.append(label)
        img = cv2.imread(path)
        img = cv2.resize(img, (28,28), interpolation=cv2.INTER_AREA)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _,img_thresh = cv2.threshold(img_gray,0,255,cv2.THRESH_OTSU)
        # img_train = img_thresh.reshape(784, )
        train_images.append(img_thresh)

train_feature = []
for i in range(len(train_images)):
    feature = hog(train_images[i],orientations=9,pixels_per_cell=(3.5,3.5),cells_per_block=(2,2),block_norm="L2")
    train_feature.append(feature)
train_feature = np.array(train_feature,dtype = np.float32)


clf = svm.SVC(C=100)
clf.fit(train_feature, train_labels)

dump(clf, 'trainHOGModel.joblib')
# clf = load('trainModel.joblib')
# test_images = train_images[480:580]
# test_labels = train_labels[480:580]

# predict = clf.predict(test_images)
# ac_score = metrics.accuracy_score(test_labels, predict)
# print(ac_score)

# cv2.imshow('t',train_images[200])
# cv2.waitKey(0)
# train_images1 = np.array(train_images1)/255
# predict = clf.predict(train_images1)
# print(predict)

